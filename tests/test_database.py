import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import database


class DatabaseBackupTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.database_path = self.root / "license_plate.db"
        self.db_path_patch = patch.object(database, "DB_PATH", self.database_path)
        self.db_path_patch.start()
        database.init_db()

    def tearDown(self):
        self.db_path_patch.stop()
        self.temp_dir.cleanup()

    def test_restore_database_and_remap_evidence_paths(self):
        database.upsert_vehicle({"plate_number": "51A12345"})
        old_evidence = self.root / "evidence"
        old_evidence.mkdir()
        old_image = old_evidence / "capture.jpg"
        old_image.write_bytes(b"evidence")
        database.add_event(
            "51A12345",
            "IN",
            "2026-10-13 10:00:00",
            "manual",
            str(old_image),
        )

        backup_path = self.root / "backup.db"
        database.backup_database(str(backup_path))
        database.upsert_vehicle({"plate_number": "30B67890"})

        restored_evidence = self.root / "restored" / "evidence"
        restored_evidence.mkdir(parents=True)
        (restored_evidence / "capture.jpg").write_bytes(b"evidence")
        database.restore_database(str(backup_path), str(restored_evidence))
        database.init_db()

        self.assertIsNotNone(database.get_vehicle_by_plate("51A12345"))
        self.assertIsNone(database.get_vehicle_by_plate("30B67890"))
        event = database.get_recent_events(1)[0]
        self.assertEqual(event["image_path"], str(restored_evidence / "capture.jpg"))

    def test_invalid_backup_does_not_replace_current_database(self):
        database.upsert_vehicle({"plate_number": "51A12345"})
        invalid_backup = self.root / "invalid.db"
        invalid_backup.write_bytes(b"not a sqlite database")

        with self.assertRaises(ValueError):
            database.restore_database(str(invalid_backup))

        self.assertIsNotNone(database.get_vehicle_by_plate("51A12345"))

    def test_restore_remaps_evidence_to_configured_object_storage(self):
        database.add_event(
            "51A12345",
            "IN",
            "2026-10-13 10:00:00",
            "manual",
            r"C:\old-app\evidence\capture.jpg",
        )
        backup_path = self.root / "backup.db"
        database.backup_database(str(backup_path))

        database.restore_database(
            str(backup_path),
            "s3://plate-data/app-data/evidence",
        )

        event = database.get_recent_events(1)[0]
        self.assertEqual(
            event["image_path"],
            "s3://plate-data/app-data/evidence/capture.jpg",
        )


class PostgresAdapterTests(unittest.TestCase):
    class Cursor:
        def fetchone(self):
            return {"id": 41}

        def fetchall(self):
            return []

    class Connection:
        def __init__(self):
            self.sql = None
            self.parameters = None

        def execute(self, sql, parameters=()):
            self.sql = sql
            self.parameters = parameters
            return PostgresAdapterTests.Cursor()

    def test_translates_sqlite_schema_and_query_syntax(self):
        connection = self.Connection()
        adapter = database.PostgresConnection(connection)
        adapter.execute(
            """
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.assertIn("BIGSERIAL PRIMARY KEY", connection.sql)
        self.assertIn("CURRENT_TIMESTAMP::timestamp(0)::text", connection.sql)

        adapter.execute(
            "SELECT * FROM access_events WHERE date(event_time) = date(?) "
            "ORDER BY datetime(event_time) DESC LIMIT ?",
            ("2026-10-14", 10),
        )
        self.assertIn("substring(event_time, 1, 10) = %s", connection.sql)
        self.assertIn("ORDER BY event_time DESC LIMIT %s", connection.sql)
        self.assertEqual(connection.parameters, ("2026-10-14", 10))
        adapter.execute(
            "UPDATE vehicles SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (5,),
        )
        self.assertIn("updated_at = CURRENT_TIMESTAMP::timestamp(0)::text", connection.sql)

    def test_returns_inserted_id_for_existing_lastrowid_callers(self):
        connection = self.Connection()
        cursor = database.PostgresConnection(connection).execute(
            "INSERT INTO vehicles (plate_number) VALUES (?)",
            ("51A12345",),
        )
        self.assertIn("RETURNING id", connection.sql)
        self.assertEqual(cursor.lastrowid, 41)

    def test_translates_postgres_errors_for_existing_ui_handlers(self):
        import psycopg

        class BrokenConnection:
            def execute(self, sql, parameters=()):
                raise psycopg.OperationalError("database unavailable")

        with self.assertRaisesRegex(database.sqlite3.OperationalError, "database unavailable"):
            database.PostgresConnection(BrokenConnection()).execute("SELECT 1")


if __name__ == "__main__":
    unittest.main()
