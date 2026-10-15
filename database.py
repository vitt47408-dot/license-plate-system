from __future__ import annotations

import math
import os
import re
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "license_plate.db"
DATABASE_URL = ""


class PostgresConnection:
    def __init__(self, connection):
        self._connection = connection

    def execute(self, sql: str, parameters=()):
        pragma_match = re.fullmatch(r"\s*PRAGMA\s+table_info\((\w+)\)\s*", sql, re.IGNORECASE)
        if pragma_match:
            table_name = pragma_match.group(1)
            cursor = self._connection.execute(
                """
                SELECT column_name AS name, data_type AS type,
                       CASE WHEN is_nullable = 'NO' THEN 1 ELSE 0 END AS notnull,
                       column_default AS dflt_value,
                       CASE WHEN column_name = (
                           SELECT kcu.column_name
                           FROM information_schema.table_constraints tc
                           JOIN information_schema.key_column_usage kcu
                             ON tc.constraint_name = kcu.constraint_name
                            AND tc.table_schema = kcu.table_schema
                           WHERE tc.constraint_type = 'PRIMARY KEY'
                             AND tc.table_schema = current_schema()
                             AND tc.table_name = %s
                           LIMIT 1
                       ) THEN 1 ELSE 0 END AS pk
                FROM information_schema.columns
                WHERE table_schema = current_schema() AND table_name = %s
                ORDER BY ordinal_position
                """,
                (table_name, table_name),
            )
            return PostgresCursor(cursor)

        rewritten = re.sub(r"datetime\(([\w.]+)\)", r"\1", sql)
        rewritten = rewritten.replace(
            "date(event_time) = date(?)",
            "substring(event_time, 1, 10) = ?",
        )
        rewritten = rewritten.replace(
            "date(event_time) BETWEEN date(?) AND date(?)",
            "substring(event_time, 1, 10) BETWEEN ? AND ?",
        )
        rewritten = rewritten.replace("time(event_time) >= time(?)", "substring(event_time, 12, 8) >= ?")
        rewritten = rewritten.replace("time(event_time) <= time(?)", "substring(event_time, 12, 8) <= ?")
        rewritten = re.sub(
            r"INTEGER PRIMARY KEY AUTOINCREMENT",
            "BIGSERIAL PRIMARY KEY",
            rewritten,
            flags=re.IGNORECASE,
        )
        rewritten = re.sub(
            r"(TEXT(?: NOT NULL)? DEFAULT) CURRENT_TIMESTAMP",
            r"\1 CURRENT_TIMESTAMP::timestamp(0)::text",
            rewritten,
            flags=re.IGNORECASE,
        )
        rewritten = re.sub(
            r"=\s*CURRENT_TIMESTAMP(?!::timestamp)",
            "= CURRENT_TIMESTAMP::timestamp(0)::text",
            rewritten,
            flags=re.IGNORECASE,
        )
        rewritten = rewritten.replace("?", "%s")
        wants_id = bool(
            re.match(
                r"\s*INSERT\s+INTO\s+(vehicles|access_events|ocr_samples)\b",
                rewritten,
                re.IGNORECASE,
            )
        )
        if wants_id and " returning " not in rewritten.lower():
            rewritten = rewritten.rstrip().rstrip(";") + " RETURNING id"
        try:
            cursor = self._connection.execute(rewritten, parameters)
        except Exception as exc:
            self._raise_database_error(exc)
            raise
        lastrowid = None
        if wants_id:
            row = cursor.fetchone()
            lastrowid = row["id"] if row else None
        return PostgresCursor(cursor, lastrowid)

    def commit(self):
        try:
            self._connection.commit()
        except Exception as exc:
            self._raise_database_error(exc)
            raise

    def close(self):
        self._connection.close()

    @staticmethod
    def _raise_database_error(exc):
        try:
            import psycopg
        except ImportError:
            return
        if isinstance(exc, psycopg.Error):
            raise sqlite3.OperationalError(str(exc)) from exc


class PostgresCursor:
    def __init__(self, cursor, lastrowid=None):
        self._cursor = cursor
        self.lastrowid = lastrowid

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()


def configure_database(database_url: str = "") -> None:
    global DATABASE_URL
    DATABASE_URL = database_url.strip()


def database_backend_configured() -> bool:
    return bool(DATABASE_URL)


def normalize_plate(value: str) -> str:
    if value is None:
        return ""
    cleaned = "".join(ch for ch in str(value).upper().replace(" ", "") if ch.isalnum())
    return cleaned[:15]


def get_connection():
    if DATABASE_URL:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL đã được cấu hình nhưng thiếu gói psycopg. "
                "Cài dependencies từ requirements.txt rồi khởi động lại."
            ) from exc
        try:
            connection = psycopg.connect(DATABASE_URL, row_factory=dict_row)
        except psycopg.Error as exc:
            raise sqlite3.OperationalError(str(exc)) from exc
        return PostgresConnection(connection)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT UNIQUE NOT NULL,
            vehicle_type TEXT,
            color TEXT,
            owner_name TEXT,
            phone_number TEXT,
            notes TEXT,
            is_allowed INTEGER DEFAULT 0,
            is_registered INTEGER DEFAULT 0,
            is_restricted INTEGER NOT NULL DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS access_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            direction TEXT CHECK(direction IN ('IN', 'OUT')) NOT NULL,
            event_time TEXT NOT NULL,
            source_type TEXT,
            image_path TEXT,
            confidence REAL DEFAULT 0,
            status TEXT DEFAULT 'confirmed',
            review_reason TEXT,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    vehicle_columns = {row["name"] for row in conn.execute("PRAGMA table_info(vehicles)").fetchall()}
    for name, definition in {
        "phone_number": "TEXT",
        "is_restricted": "INTEGER NOT NULL DEFAULT 0",
    }.items():
        if name not in vehicle_columns:
            conn.execute(f"ALTER TABLE vehicles ADD COLUMN {name} {definition}")

    event_columns = {row["name"] for row in conn.execute("PRAGMA table_info(access_events)").fetchall()}
    if "review_reason" not in event_columns:
        conn.execute("ALTER TABLE access_events ADD COLUMN review_reason TEXT")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id TEXT,
            details TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ocr_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            predicted_text TEXT NOT NULL DEFAULT '',
            confirmed_text TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0,
            image_path TEXT NOT NULL,
            source_type TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def set_setting(key: str, value: str) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )
    conn.commit()
    conn.close()


def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    conn = get_connection()
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    conn.close()
    if row is None:
        return default
    return row["value"]


def add_vehicle(data: Dict[str, Any]) -> Dict[str, Any]:
    return upsert_vehicle(data)


def upsert_vehicle(data: Dict[str, Any]) -> Dict[str, Any]:
    conn = get_connection()
    plate = normalize_plate(data.get("plate_number", ""))
    if not plate:
        conn.close()
        raise ValueError("Biển số không được để trống.")
    row = conn.execute("SELECT * FROM vehicles WHERE plate_number = ?", (plate,)).fetchone()
    is_registered = bool(data.get("is_registered", bool(row["is_registered"]) if row else False))
    is_allowed = bool(data.get("is_allowed", bool(row["is_allowed"]) if row else False))
    is_restricted = bool(data.get("is_restricted", bool(row["is_restricted"]) if row else False))
    if is_restricted and is_allowed:
        conn.close()
        raise ValueError("Xe đang bị hạn chế không thể đồng thời được cấp quyền ra vào.")
    if is_allowed and not is_registered:
        conn.close()
        raise ValueError("Chỉ phương tiện đã đăng ký mới được cấp quyền ra vào.")
    if row is None:
        cur = conn.execute(
            """
            INSERT INTO vehicles (plate_number, vehicle_type, color, owner_name, phone_number, notes, is_allowed, is_registered, is_restricted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plate,
                data.get("vehicle_type", ""),
                data.get("color", ""),
                data.get("owner_name", ""),
                data.get("phone_number", ""),
                data.get("notes", ""),
                int(is_allowed),
                int(is_registered),
                int(is_restricted),
            ),
        )
        veh_id = cur.lastrowid
        conn.commit()
        result = conn.execute("SELECT * FROM vehicles WHERE id = ?", (veh_id,)).fetchone()
    else:
        conn.execute(
            """
            UPDATE vehicles
            SET vehicle_type = ?, color = ?, owner_name = ?, phone_number = ?, notes = ?, is_allowed = ?, is_registered = ?, is_restricted = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                data.get("vehicle_type", row["vehicle_type"]),
                data.get("color", row["color"]),
                data.get("owner_name", row["owner_name"]),
                data.get("phone_number", row["phone_number"]),
                data.get("notes", row["notes"]),
                int(is_allowed),
                int(is_registered),
                int(is_restricted),
                row["id"],
            ),
        )
        conn.commit()
        result = conn.execute("SELECT * FROM vehicles WHERE id = ?", (row["id"],)).fetchone()
    conn.close()
    return dict(result)


def get_vehicle_by_plate(plate_number: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM vehicles WHERE plate_number = ?", (normalize_plate(plate_number),)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_latest_confirmed_event(plate_number: str) -> Optional[Dict[str, Any]]:
    plate = normalize_plate(plate_number)
    if not plate:
        return None
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT direction, event_time
            FROM access_events
            WHERE plate_number = ? AND status = 'confirmed'
            ORDER BY datetime(event_time) DESC, id DESC
            LIMIT 1
            """,
            (plate,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_vehicles(search: str = "") -> List[Dict[str, Any]]:
    conn = get_connection()
    query = "SELECT * FROM vehicles WHERE plate_number LIKE ? OR owner_name LIKE ? OR vehicle_type LIKE ? OR color LIKE ? OR phone_number LIKE ? ORDER BY updated_at DESC"
    keyword = f"%{search.strip()}%"
    rows = conn.execute(query, (keyword, keyword, keyword, keyword, keyword)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def delete_vehicle(vehicle_id: int) -> None:
    conn = get_connection()
    conn.execute("DELETE FROM vehicles WHERE id = ?", (vehicle_id,))
    conn.commit()
    conn.close()


def add_event(
    plate_number: str,
    direction: str,
    event_time: str,
    source_type: str,
    image_path: str = "",
    confidence: float = 0.0,
    notes: str = "",
    status: str = "confirmed",
    review_reason: str = "",
) -> Dict[str, Any]:
    plate = normalize_plate(plate_number)
    if not plate:
        raise ValueError("Biển số không hợp lệ để lưu lịch sử.")
    direction = direction.upper()
    if direction not in {"IN", "OUT"}:
        raise ValueError("Hướng di chuyển phải là IN hoặc OUT.")
    if status not in {"confirmed", "needs_review"}:
        raise ValueError("Trạng thái sự kiện không hợp lệ.")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Độ tin cậy phải nằm trong khoảng từ 0 đến 1.")
    conn = get_connection()
    cur = conn.execute(
        """
        INSERT INTO access_events (plate_number, direction, event_time, source_type, image_path, confidence, status, review_reason, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (plate, direction, event_time, source_type, image_path, confidence, status, review_reason, notes),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM access_events WHERE id = ?", (cur.lastrowid,)).fetchone()
    conn.close()
    return dict(row)


def is_dup_event_recent(plate_number: str, direction: str, seconds: int = 25) -> bool:
    conn = get_connection()
    try:
        cutoff = (datetime.now() - timedelta(seconds=max(0, seconds))).strftime("%Y-%m-%d %H:%M:%S")
        row = conn.execute(
            """
            SELECT id FROM access_events
            WHERE plate_number = ? AND direction = ? AND event_time >= ?
            ORDER BY event_time DESC
            LIMIT 1
            """,
            (normalize_plate(plate_number), direction.upper(), cutoff),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def get_recent_events(limit: int = 15) -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM access_events ORDER BY event_time DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def list_events(plate_number: str = "", date: str = "", direction: str = "") -> List[Dict[str, Any]]:
    conn = get_connection()
    sql = "SELECT * FROM access_events WHERE 1=1"
    params: List[str] = []
    if plate_number:
        sql += " AND plate_number LIKE ?"
        params.append(f"%{normalize_plate(plate_number)}%")
    if date:
        sql += " AND date(event_time) = date(?)"
        params.append(date)
    if direction:
        sql += " AND direction = ?"
        params.append(direction.upper())
    sql += " ORDER BY event_time DESC"
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def list_events_range(
    date_from: str,
    date_to: str,
    plate_number: str = "",
    direction: str = "",
    time_from: str = "",
    time_to: str = "",
    status: str = "",
) -> List[Dict[str, Any]]:
    conn = get_connection()
    sql = "SELECT * FROM access_events WHERE date(event_time) BETWEEN date(?) AND date(?)"
    params: List[str] = [date_from, date_to]
    if plate_number:
        sql += " AND plate_number LIKE ?"
        params.append(f"%{normalize_plate(plate_number)}%")
    if direction:
        sql += " AND direction = ?"
        params.append(direction.upper())
    if time_from:
        sql += " AND time(event_time) >= time(?)"
        params.append(time_from)
    if time_to:
        sql += " AND time(event_time) <= time(?)"
        params.append(time_to)
    if status:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY event_time DESC"
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_dashboard_stats() -> Dict[str, Any]:
    conn = get_connection()
    total_in = conn.execute("SELECT COUNT(*) AS c FROM access_events WHERE direction = 'IN' AND status = 'confirmed'").fetchone()["c"]
    total_out = conn.execute("SELECT COUNT(*) AS c FROM access_events WHERE direction = 'OUT' AND status = 'confirmed'").fetchone()["c"]
    active_vehicles = conn.execute(
        """
        SELECT COUNT(*) AS c FROM (
            SELECT latest.plate_number, latest.direction
            FROM access_events latest
            WHERE latest.status = 'confirmed'
              AND latest.id = (
                  SELECT candidate.id FROM access_events candidate
                  WHERE candidate.plate_number = latest.plate_number
                    AND candidate.status = 'confirmed'
                  ORDER BY datetime(candidate.event_time) DESC, candidate.id DESC
                  LIMIT 1
              )
        )
        WHERE direction = 'IN'
        """
    ).fetchone()["c"]
    unregistered = conn.execute("SELECT COUNT(*) AS c FROM vehicles WHERE is_registered = 0 AND is_restricted = 0").fetchone()["c"]
    allowed = conn.execute("SELECT COUNT(*) AS c FROM vehicles WHERE is_allowed = 1").fetchone()["c"]
    registered = conn.execute("SELECT COUNT(*) AS c FROM vehicles WHERE is_registered = 1").fetchone()["c"]
    restricted = conn.execute("SELECT COUNT(*) AS c FROM vehicles WHERE is_restricted = 1").fetchone()["c"]
    conn.close()
    return {
        "total_in": total_in,
        "total_out": total_out,
        "active_vehicles": active_vehicles,
        "unregistered": unregistered,
        "registered": registered,
        "allowed": allowed,
        "restricted": restricted,
    }


def list_active_vehicles() -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT e.plate_number, e.event_time AS last_event_time, e.image_path,
                   v.vehicle_type, v.color, v.owner_name, v.phone_number,
                   v.is_registered, v.is_allowed, v.is_restricted
            FROM access_events e
            LEFT JOIN vehicles v ON v.plate_number = e.plate_number
            WHERE e.direction = 'IN'
              AND e.status = 'confirmed'
              AND e.id = (
                  SELECT latest.id FROM access_events latest
                  WHERE latest.plate_number = e.plate_number
                    AND latest.status = 'confirmed'
                  ORDER BY datetime(latest.event_time) DESC, latest.id DESC
                  LIMIT 1
              )
            ORDER BY e.event_time ASC
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def list_attention_events(limit: int = 100) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT e.*, v.is_registered, v.is_allowed, v.is_restricted
            FROM access_events e
            LEFT JOIN vehicles v ON v.plate_number = e.plate_number
            WHERE e.status != 'confirmed'
               OR (e.source_type != 'manual' AND e.confidence < 0.55)
               OR v.plate_number IS NULL
               OR COALESCE(e.image_path, '') = ''
            ORDER BY e.event_time DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def review_event(event_id: int, new_plate: str, reviewer: str = "operator") -> Dict[str, Any]:
    plate = normalize_plate(new_plate)
    if not plate:
        raise ValueError("Biển số xác nhận không được để trống.")
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM access_events WHERE id = ?", (event_id,)).fetchone()
        if row is None:
            raise ValueError("Không tìm thấy sự kiện cần xác minh.")
        conn.execute(
            "UPDATE access_events SET plate_number = ?, status = 'confirmed', review_reason = NULL WHERE id = ?",
            (plate, event_id),
        )
        conn.execute(
            "INSERT INTO audit_logs(action, entity_type, entity_id, details) VALUES (?, ?, ?, ?)",
            ("event_reviewed", "access_event", str(event_id), f"{reviewer}: {row['plate_number']} -> {plate}"),
        )
        conn.commit()
        result = conn.execute("SELECT * FROM access_events WHERE id = ?", (event_id,)).fetchone()
        return dict(result)
    finally:
        conn.close()


def log_audit(action: str, entity_type: str, entity_id: str = "", details: str = "") -> None:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO audit_logs(action, entity_type, entity_id, details) VALUES (?, ?, ?, ?)",
            (action, entity_type, entity_id, details),
        )
        conn.commit()
    finally:
        conn.close()


def list_audit_logs(limit: int = 100) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM audit_logs ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def add_ocr_sample(
    predicted_text: str,
    confirmed_text: str,
    confidence: float,
    image_path: str,
    source_type: str,
) -> Dict[str, Any]:
    confirmed = normalize_plate(confirmed_text)
    if not confirmed:
        raise ValueError("Biển số đã xác nhận không được để trống.")
    confidence_value = float(confidence)
    if not math.isfinite(confidence_value) or not 0 <= confidence_value <= 1:
        raise ValueError("Độ tin cậy OCR phải nằm trong khoảng từ 0 đến 1.")
    if not image_path:
        raise ValueError("Cần có ảnh cắt biển số để lưu mẫu OCR.")
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO ocr_samples (predicted_text, confirmed_text, confidence, image_path, source_type)
            VALUES (?, ?, ?, ?, ?)
            """,
            (normalize_plate(predicted_text), confirmed, confidence_value, image_path, source_type),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM ocr_samples WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return dict(row)
    finally:
        conn.close()


def list_ocr_samples(limit: int = 500) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM ocr_samples ORDER BY created_at DESC, id DESC LIMIT ?",
            (max(1, min(int(limit), 5000)),),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def backup_database(output_path: str) -> str:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if DATABASE_URL:
        source = get_connection()
        backup = sqlite3.connect(target)
        try:
            for table in ("vehicles", "access_events", "settings", "audit_logs", "ocr_samples"):
                columns = source.execute(f"PRAGMA table_info({table})").fetchall()
                if not columns:
                    continue
                definitions = []
                for column in columns:
                    data_type = str(column["type"]).lower()
                    sqlite_type = "REAL" if data_type in {"real", "double precision", "numeric"} else (
                        "INTEGER" if data_type in {"integer", "bigint", "smallint"} else "TEXT"
                    )
                    primary = " PRIMARY KEY" if column["pk"] else ""
                    definitions.append(f'"{column["name"]}" {sqlite_type}{primary}')
                backup.execute(f'CREATE TABLE "{table}" ({", ".join(definitions)})')
                rows = source.execute(f'SELECT * FROM "{table}"').fetchall()
                names = [column["name"] for column in columns]
                quoted_names = ", ".join(f'"{name}"' for name in names)
                placeholders = ", ".join("?" for _ in names)
                backup.executemany(
                    f'INSERT INTO "{table}" ({quoted_names}) VALUES ({placeholders})',
                    [tuple(row[name] for name in names) for row in rows],
                )
            backup.commit()
        finally:
            backup.close()
            source.close()
        return str(target)
    source = get_connection()
    backup = sqlite3.connect(target)
    try:
        source.backup(backup)
    finally:
        backup.close()
        source.close()
    return str(target)


def validate_database_backup(source_path: str) -> None:
    source_file = Path(source_path)
    if not source_file.is_file():
        raise ValueError("Không tìm thấy file database trong bản sao lưu.")

    conn = sqlite3.connect(source_file)
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise ValueError("Database trong bản sao lưu bị lỗi hoặc không toàn vẹn.")
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if not {"vehicles", "access_events"}.issubset(tables):
            raise ValueError("File được chọn không phải database của ứng dụng.")
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"Không thể đọc database trong bản sao lưu: {exc}") from exc
    finally:
        conn.close()


def _remap_restored_image_paths(conn, evidence_root: str | Path) -> None:
    for table in ("access_events", "ocr_samples"):
        columns = set()
        for column in conn.execute(f"PRAGMA table_info({table})").fetchall():
            columns.add(column["name"] if hasattr(column, "keys") else column[1])
        if "image_path" not in columns:
            continue
        rows = conn.execute(
            f"SELECT id, image_path FROM {table} WHERE image_path IS NOT NULL AND image_path != ''"
        ).fetchall()
        for row in rows:
            if hasattr(row, "keys"):
                row_id, image_path = row["id"], row["image_path"]
            else:
                row_id, image_path = row[0], row[1]
            normalized = str(image_path).replace("\\", "/")
            marker = "/evidence/"
            if marker not in normalized.lower():
                continue
            relative_path = normalized[normalized.lower().rfind(marker) + len(marker) :]
            relative_parts = relative_path.split("/")
            if any(part in {"", ".", ".."} for part in relative_parts):
                continue
            if isinstance(evidence_root, str) and evidence_root.startswith("s3://"):
                target_path = f"{evidence_root.rstrip('/')}/{'/'.join(relative_parts)}"
                conn.execute(
                    f"UPDATE {table} SET image_path = ? WHERE id = ?",
                    (target_path, row_id),
                )
                continue
            target = Path(evidence_root).joinpath(*relative_parts)
            if target.is_file():
                conn.execute(
                    f"UPDATE {table} SET image_path = ? WHERE id = ?",
                    (str(target), row_id),
                )


def _restore_postgres_database(source_path: str, evidence_root: Optional[str]) -> str:
    init_db()
    source = sqlite3.connect(source_path)
    source.row_factory = sqlite3.Row
    target = get_connection()
    tables = ("vehicles", "access_events", "settings", "audit_logs", "ocr_samples")
    try:
        source_tables = {
            row[0] for row in source.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        for table in tables:
            if table not in source_tables:
                continue
            target.execute(f'DELETE FROM "{table}"')
            columns = [row["name"] for row in source.execute(f'PRAGMA table_info({table})').fetchall()]
            if not columns:
                continue
            placeholders = ", ".join("?" for _ in columns)
            column_sql = ", ".join(f'"{column}"' for column in columns)
            rows = source.execute(f'SELECT * FROM "{table}"').fetchall()
            for row in rows:
                target.execute(
                    f'INSERT INTO "{table}" ({column_sql}) VALUES ({placeholders})',
                    tuple(row[column] for column in columns),
                )
            if "id" in columns:
                target.execute(
                    f"""
                    SELECT setval(
                        pg_get_serial_sequence(?, 'id'),
                        COALESCE(MAX(id), 1),
                        MAX(id) IS NOT NULL
                    ) FROM "{table}"
                    """,
                    (table,),
                )
        if evidence_root:
            _remap_restored_image_paths(target, evidence_root)
        target.commit()
    except Exception:
        target._connection.rollback()
        raise
    finally:
        source.close()
        target.close()
    return "PostgreSQL database"


def restore_database(source_path: str, evidence_root: Optional[str] = None) -> str:
    validate_database_backup(source_path)
    if DATABASE_URL:
        return _restore_postgres_database(source_path, evidence_root)
    target = DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    staging_path = target.with_name(f".{target.name}.{uuid.uuid4().hex}.restore")
    source = sqlite3.connect(source_path)
    staging = None
    try:
        staging = sqlite3.connect(staging_path)
        source.backup(staging)
        if evidence_root:
            _remap_restored_image_paths(staging, evidence_root)
        staging.commit()
        integrity = staging.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise ValueError("Không thể khôi phục vì database bản sao lưu không toàn vẹn.")
        staging.close()
        staging = None
        os.replace(staging_path, target)
    finally:
        source.close()
        if staging is not None:
            staging.close()
        if staging_path.exists():
            staging_path.unlink()
    return str(target)


def export_history_csv(date_from: str, date_to: str, output_path: str) -> str:
    import pandas as pd

    df = pd.DataFrame(list_events_range(date_from, date_to))
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def export_history_excel(date_from: str, date_to: str, output_path: str) -> str:
    import pandas as pd

    df = pd.DataFrame(list_events_range(date_from, date_to))
    df.to_excel(output_path, index=False)
    return output_path
