from __future__ import annotations

import hashlib
import hmac
import base64
import io
import shutil
import sqlite3
import stat
import tempfile
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from camera_relay import camera_relay
from browser_camera_component import browser_camera
from database import (
    add_event,
    add_ocr_sample,
    backup_database,
    configure_database,
    database_backend_configured,
    DB_PATH,
    delete_vehicle,
    export_history_csv,
    export_history_excel,
    get_dashboard_stats,
    get_latest_confirmed_event,
    get_recent_events,
    get_vehicle_by_plate,
    init_db,
    is_dup_event_recent,
    list_active_vehicles,
    list_attention_events,
    list_audit_logs,
    list_events,
    list_events_range,
    list_ocr_samples,
    list_vehicles,
    log_audit,
    normalize_plate,
    get_setting,
    restore_database,
    review_event,
    set_setting,
    upsert_vehicle,
    validate_database_backup,
)
from ocr_engine import OCRService
from plate_detector import PlateDetector
from live_camera import AutomaticPlateProcessor
from vehicle_detector import VehicleDetector
from storage import (
    configure_object_storage,
    current_prefix_uri,
    delete_uri,
    exists as stored_file_exists,
    is_object_uri,
    objects as stored_objects,
    put_bytes,
    read_bytes,
    object_storage_enabled,
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
EVIDENCE_DIR = BASE_DIR / "evidence"
OCR_SAMPLE_DIR = EVIDENCE_DIR / "ocr_samples"
EXPORT_DIR = BASE_DIR / "exports"
MODEL_DIR = BASE_DIR / "models"
MAX_BACKUP_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
MAX_VIDEO_SCAN_RESULTS = 25
MAX_VIDEO_SCAN_PREVIEW_SIZE = 1280

PAGE_DETAILS = {
    "Giám sát trực tiếp": ("Giám sát trực tiếp", "Camera, nhận diện biển số và xác nhận sự kiện tại cổng."),
    "Lịch sử ra vào": ("Lịch sử ra vào", "Tra cứu lại các lượt phương tiện theo biển số, ngày và hướng."),
    "Quản lý xe": ("Quản lý xe", "Hồ sơ phương tiện, chủ xe và quyền ra vào."),
    "Báo cáo": ("Báo cáo", "Thống kê lượt xe và xuất lịch sử ra vào."),
    "Cấu hình": ("Cấu hình", "Thiết bị camera và mô hình nhận diện."),
    "Huấn luyện model": ("Huấn luyện model", "Fine-tune YOLO plate detector trên dataset của bạn."),
}


MAX_BROWSER_FRAME_BYTES = 1_500_000


def decode_browser_camera_frame(payload: dict) -> np.ndarray:
    frame_data = payload.get("frame")
    if not isinstance(frame_data, str) or not frame_data:
        raise ValueError("Camera chưa gửi khung hình.")
    if len(frame_data) > MAX_BROWSER_FRAME_BYTES * 4 // 3 + 8:
        raise ValueError("Khung hình camera vượt giới hạn 1.5 MB.")
    try:
        encoded = base64.b64decode(frame_data, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("Dữ liệu ảnh camera không hợp lệ.") from exc
    if not encoded or len(encoded) > MAX_BROWSER_FRAME_BYTES:
        raise ValueError("Khung hình camera rỗng hoặc vượt giới hạn 1.5 MB.")
    frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("Không giải mã được khung hình camera.")
    return frame


def apply_app_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #e7eef7;
            --muted: #91a4b9;
            --brand: #39c6bd;
            --brand-dark: #138e90;
            --surface: #142235;
            --surface-raised: #192a3e;
            --line: #26384d;
            --canvas: #0b1420;
        }
        html, body, [class*="css"] {
            font-family: 'Segoe UI', 'Inter', sans-serif;
        }
        .stApp, [data-testid="stAppViewContainer"] {
            background: var(--canvas); color: var(--ink);
        }
        [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li { color: #d5e0ed; }
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
            color: var(--muted);
        }
        h1, h2, h3, h4, label, [data-testid="stWidgetLabel"] p {
            color: var(--ink);
        }
        [data-testid="stHeader"] {
            background: rgba(11,20,32,.94);
            border-bottom: 1px solid rgba(38,56,77,.75);
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 1600px;
            padding: 1.55rem 1.8rem 2.5rem;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #101c2b 0%, #0d1724 100%);
            border-right: 0;
            min-width: 12rem !important;
            max-width: 12rem !important;
        }
        section[data-testid="stSidebar"] * { color: #dbe6f1; }
        section[data-testid="stSidebar"] [data-testid="stRadio"] > label {
            color: #9db3c8;
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .11em;
            text-transform: uppercase;
            padding: 0 .55rem;
            margin: .25rem 0 .55rem;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
            gap: .2rem;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
            border: 1px solid transparent;
            border-radius: 10px;
            padding: .48rem .62rem;
            transition: background .18s ease, border-color .18s ease;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
            display: none;
        }
        section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child {
            display: none !important;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(100deg,rgba(48,190,183,.19),rgba(48,190,183,.07));
            border-color: rgba(57,198,189,.24);
            box-shadow: inset 3px 0 #39c6bd;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
            background: rgba(255,255,255,.055);
            border-color: rgba(255,255,255,.07);
        }
        .brand-card {
            padding: .88rem;
            margin: .2rem 0 1.35rem;
            border: 1px solid rgba(192,218,237,.11);
            border-radius: 12px;
            background: linear-gradient(145deg,rgba(255,255,255,.065),rgba(255,255,255,.018));
            box-shadow: 0 10px 25px rgba(0,0,0,.12);
        }
        .brand-mark {
            display: inline-flex; align-items: center; justify-content: center;
            width: 40px; height: 40px; border-radius: 12px;
            background: linear-gradient(145deg,#56ded0,#168e91);
            color: #08222c; font-size: 14px; font-weight: 900;
            letter-spacing: -.04em; margin-bottom: .8rem;
        }
        .brand-title { font-size: .94rem; font-weight: 800; color: white; letter-spacing: .045em; }
        .brand-subtitle { color: #91a6bc; font-size: .73rem; margin-top: .35rem; line-height: 1.55; }
        .page-heading {
            display: flex; align-items: flex-end; justify-content: space-between;
            gap: 1rem; padding: .15rem 0 1rem; margin-bottom: .9rem;
            border-bottom: 1px solid var(--line);
        }
        .page-heading h1 {
            color: var(--ink); font-size: clamp(1.65rem, 2.4vw, 2.25rem);
            letter-spacing: -.05em; font-weight: 800; margin: 0;
        }
        .page-heading p { color: var(--muted); margin: .45rem 0 0; font-size: .91rem; }
        .page-heading.dashboard-heading {
            align-items: center; padding: .1rem 0 .8rem; margin-bottom: .8rem;
        }
        .dashboard-heading h1 {
            font-size: 1.08rem; letter-spacing: -.015em; font-weight: 750;
        }
        .dashboard-heading p { margin-top: .2rem; font-size: .78rem; }
        .live-pill {
            white-space: nowrap; color: #8de3d5; background: rgba(35,154,148,.12);
            border: 1px solid rgba(57,198,189,.24); border-radius: 999px;
            padding: .43rem .78rem; font-size: .7rem; font-weight: 800;
            letter-spacing: .055em;
        }
        [data-testid="stMetric"] {
            background: var(--surface); border: 1px solid var(--line);
            border-radius: 11px; padding: .9rem 1rem;
            box-shadow: 0 6px 18px rgba(0,0,0,.12);
            transition: transform .18s ease, box-shadow .18s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: #35506a;
            box-shadow: 0 10px 24px rgba(0,0,0,.2);
        }
        [data-testid="stMetricLabel"] { color: var(--muted); font-size: .82rem; font-weight: 650; }
        [data-testid="stMetricLabel"] > div { white-space: normal; }
        [data-testid="stMetricValue"] { color: #f1f6fb; font-weight: 800; letter-spacing: -.035em; }
        [data-testid="stMetricDelta"] { color: #65d3b0; }
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {
            border: 1px solid var(--line); border-radius: 12px; overflow: hidden;
            background: var(--surface); box-shadow: 0 6px 18px rgba(0,0,0,.12);
        }
        .stButton > button, .stFormSubmitButton > button {
            border-radius: 9px; font-weight: 700; min-height: 2.6rem;
            transition: transform .15s ease, box-shadow .15s ease, background .15s ease;
            background: #1b2b40;
            border: 1px solid #33485f;
            color: #e7eef7;
        }
        .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
            background: linear-gradient(135deg, #28a9a6, #14777e);
            border: 0; color: #f7ffff;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover {
            transform: translateY(-1px); border-color: #42c5bb;
            box-shadow: 0 6px 16px rgba(17,97,109,.2);
        }
        [data-testid="stFileUploader"] section {
            border: 1px dashed #3d566d; border-radius: 12px; background: #111e2d;
        }
        [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            border-radius: 9px; background: #101d2c; color: #e7eef7;
            border-color: #33475d;
        }
        [data-testid="stTextInput"] input:focus,
        [data-testid="stTextArea"] textarea:focus {
            border-color: #39c6bd;
            box-shadow: 0 0 0 1px #39c6bd;
        }
        [data-testid="stAlert"] {
            border-radius: 11px;
            border: 1px solid rgba(129,157,184,.16);
        }
        [data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 11px;
            background: var(--surface);
        }
        .dashboard-hero {
            position: relative; overflow: hidden;
            padding: 1.1rem 1.25rem; margin: .1rem 0 1rem;
            border: 1px solid rgba(72,199,188,.18); border-radius: 12px;
            color: white;
            background: linear-gradient(105deg,#152b3d 0%,#163746 70%,#17464b 100%);
            box-shadow: 0 10px 26px rgba(0,0,0,.15);
        }
        .dashboard-hero:after {
            content: ""; position: absolute; width: 210px; height: 210px;
            right: -58px; top: -112px; border-radius: 50%;
            border: 1px solid rgba(116,228,216,.14);
            box-shadow: 0 0 0 23px rgba(116,228,216,.025), 0 0 0 48px rgba(116,228,216,.018);
        }
        .dashboard-hero-title { font-size: 1.08rem; font-weight: 800; letter-spacing: -.015em; }
        .dashboard-hero-copy { opacity: .82; font-size: .86rem; margin-top: .38rem; }
        .monitor-panel-title {
            color: #f1f6fb; font-size: 1rem; font-weight: 800;
            letter-spacing: -.02em; margin: .1rem 0 .25rem;
        }
        .monitor-panel-copy { color: #91a4b9; font-size: .8rem; margin-bottom: .8rem; }
        .plate-ticket {
            display: inline-block; padding: .48rem .9rem; margin: .15rem 0 .65rem;
            border: 2px solid #dce6ed; border-radius: 7px; background: #f5f7f8;
            color: #172331; font-family: 'Consolas', monospace; font-size: 1.65rem;
            font-weight: 900; letter-spacing: .12em; box-shadow: 0 3px 12px #0003;
        }
        .event-state {
            display: inline-block; padding: .25rem .58rem; border-radius: 999px;
            color: #91e0c2; background: #1d493b; font-size: .72rem; font-weight: 800;
        }
        .event-state.review { color: #ffd18b; background: #4a3925; }
        [data-testid="stCameraInput"] {
            border: 1px solid var(--line); border-radius: 12px;
            background: var(--surface); padding: .4rem;
        }
        .stTabs [data-baseweb="tab-list"] { gap: .35rem; border-bottom-color: var(--line); }
        .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; color: var(--muted); }
        .stTabs [aria-selected="true"] { color: #6de0d1 !important; }
        h2, h3 {
            color: var(--ink); letter-spacing: -.03em; font-weight: 750;
            padding-top: .25rem;
        }
        hr { border-color: var(--line); margin: 1.3rem 0; }
        [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {
            background: #17283b !important; color: #e7eef7 !important;
            border-color: #33485e !important;
        }
        [role="option"]:hover { background: #233a50 !important; }
        [data-testid="stDateInput"] input, [data-testid="stTimeInput"] input,
        [data-testid="stNumberInput"] input {
            background: #101d2c; color: #e7eef7; border-color: #33475d;
        }
        [data-testid="stProgress"] > div > div { background: #38bdb3; }
        [data-testid="stStatusWidget"] { color: #91a4b9; }
        @media (max-width: 700px) {
            [data-testid="stMainBlockContainer"] { padding: 1rem .8rem 1.8rem; }
            .page-heading { align-items: flex-start; flex-direction: column; }
            .live-pill { align-self: flex-start; }
            .dashboard-hero { padding: 1.15rem; }
        }
        @media (max-width: 850px) {
            [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
            [data-testid="stHorizontalBlock"] [data-testid="stColumn"] {
                flex: 1 1 100% !important;
                width: 100% !important;
                min-width: 100% !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def require_app_password() -> None:
    try:
        configured_password = str(st.secrets["APP_PASSWORD"])
    except (KeyError, FileNotFoundError):
        st.error("Chưa cấu hình APP_PASSWORD trong Streamlit Secrets. Hãy cấu hình bí mật trước khi sử dụng app.")
        st.stop()

    if len(configured_password) < 12:
        st.error("APP_PASSWORD phải có ít nhất 12 ký tự.")
        st.stop()

    if st.session_state.get("authenticated"):
        return

    st.title("Đăng nhập hệ thống")
    with st.form("app_login"):
        entered_password = st.text_input("Mật khẩu", type="password")
        submitted = st.form_submit_button("Đăng nhập", type="primary")
    if submitted:
        if hmac.compare_digest(entered_password, configured_password):
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Mật khẩu không đúng.")
    st.stop()


def ensure_dirs() -> None:
    for path in [UPLOAD_DIR, EVIDENCE_DIR, OCR_SAMPLE_DIR, EXPORT_DIR, MODEL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


@st.cache_resource(show_spinner=False)
def get_plate_detector() -> PlateDetector:
    return PlateDetector()


@st.cache_resource(show_spinner=False)
def get_ocr_service() -> OCRService:
    return OCRService()


@st.cache_resource(show_spinner=False)
def get_vehicle_detector() -> VehicleDetector:
    return VehicleDetector()


def get_live_camera_models() -> tuple[PlateDetector, OCRService, VehicleDetector]:
    import torch

    torch.set_num_threads(1)
    return get_plate_detector(), get_ocr_service(), get_vehicle_detector()


def save_uploaded_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    content = uploaded_file.getvalue()
    signature = hashlib.sha256(content).hexdigest()
    cache_key = f"saved_upload_{signature}"
    cached_path = st.session_state.get(cache_key)
    if cached_path and Path(cached_path).is_file():
        return cached_path
    safe_name = Path(uploaded_file.name).name.replace(" ", "_")
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{safe_name}"
    target = UPLOAD_DIR / name
    with open(target, "wb") as f:
        f.write(content)
    if object_storage_enabled():
        put_bytes(f"uploads/{name}", content, uploaded_file.type or "application/octet-stream")
    st.session_state[cache_key] = str(target)
    return str(target)


def save_evidence_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    safe_name = Path(uploaded_file.name).name.replace(" ", "_")
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{safe_name}"
    content = uploaded_file.getvalue()
    if object_storage_enabled():
        return put_bytes(f"evidence/{name}", content, uploaded_file.type or "application/octet-stream")
    target = EVIDENCE_DIR / name
    target.write_bytes(content)
    return str(target)


def save_frame_evidence(frame: np.ndarray) -> str:
    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        raise ValueError("Không thể mã hóa ảnh minh chứng.")
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_capture.jpg"
    content = encoded.tobytes()
    if object_storage_enabled():
        return put_bytes(f"evidence/{name}", content, "image/jpeg")
    target = EVIDENCE_DIR / name
    target.write_bytes(content)
    return str(target)


def save_confirmed_ocr_sample(
    crop: np.ndarray,
    predicted_text: str,
    confirmed_text: str,
    confidence: float,
    source_type: str,
) -> dict:
    if crop is None or crop.size == 0:
        raise ValueError("Vùng ảnh biển số rỗng, không thể lưu mẫu OCR.")
    success, encoded = cv2.imencode(".jpg", crop)
    if not success:
        raise ValueError("Không thể mã hóa ảnh cắt biển số.")
    name = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_ocr.jpg"
    if object_storage_enabled():
        target_path = put_bytes(f"evidence/ocr_samples/{name}", encoded.tobytes(), "image/jpeg")
    else:
        target = OCR_SAMPLE_DIR / name
        target.write_bytes(encoded.tobytes())
        target_path = str(target)
    try:
        return add_ocr_sample(
            predicted_text,
            confirmed_text,
            confidence,
            target_path,
            source_type,
        )
    except (OSError, ValueError, sqlite3.Error):
        if is_object_uri(target_path):
            delete_uri(target_path)
        else:
            Path(target_path).unlink(missing_ok=True)
        raise


def image_exists(image_path: str) -> bool:
    return bool(image_path) and stored_file_exists(image_path)


def image_content(image_path: str):
    return read_bytes(image_path) if is_object_uri(image_path) else image_path


def read_frame_from_video(video_path: str, frame_number: int = 0):
    cap = cv2.VideoCapture(video_path)
    if frame_number > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Không đọc được khung hình số {frame_number} từ video.")
    return frame


def resize_video_scan_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, MAX_VIDEO_SCAN_PREVIEW_SIZE / max(height, width))
    if scale < 1.0:
        return cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return frame


def encode_video_scan_preview(frame: np.ndarray) -> bytes:
    frame = resize_video_scan_frame(frame)
    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 80],
    )
    if not success:
        raise ValueError("Không thể mã hóa ảnh xem trước của kết quả quét video.")
    return encoded.tobytes()


def decode_video_scan_preview(image_bytes: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("Không thể đọc lại ảnh của kết quả quét video.")
    return frame


def detect_video_frame(detector: PlateDetector, frame: np.ndarray) -> list[dict]:
    try:
        return detector.detect(frame, imgsz=416)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc) or "imgsz" not in str(exc):
            raise
        return detector.detect(frame)


def scan_video_for_plates(
    video_path: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[dict], bool]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError("Không mở được video để quét.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if frame_count <= 0 or not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Video không có thông tin thời lượng/tốc độ khung hình hợp lệ.")

    frame_stride = max(1, round(fps / 2), (frame_count + 299) // 300)
    sample_count = (frame_count + frame_stride - 1) // frame_stride
    detector = get_plate_detector()
    ocr_service = get_ocr_service()
    if not ocr_service._loaded:
        cap.release()
        raise RuntimeError(
            f"OCR chưa sẵn sàng. {ocr_service.load_error or 'Kiểm tra EasyOCR và quyền tải model.'}"
        )

    found: dict[str, dict] = {}
    results_truncated = False
    frame_number = 0
    try:
        for sample_index in range(sample_count):
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                break

            scan_frame = resize_video_scan_frame(frame)
            for detection in detect_video_frame(detector, scan_frame):
                crop = detector.crop_plate(scan_frame, detection)
                result = ocr_service.recognize(crop)
                text = str(result.get("text") or "").strip()
                plate = normalize_plate(text)
                confidence = float(result.get("confidence") or 0)
                plate_like = bool(result.get("plate_like"))
                if len(plate) >= 5 and plate_like:
                    rank = (plate_like, confidence)
                    previous = found.get(plate)
                    if previous is not None and rank <= previous["rank"]:
                        continue
                    if previous is None and len(found) >= MAX_VIDEO_SCAN_RESULTS:
                        lowest_plate = min(found, key=lambda key: found[key]["rank"])
                        if rank <= found[lowest_plate]["rank"]:
                            results_truncated = True
                            continue
                        del found[lowest_plate]
                        results_truncated = True

                    found[plate] = {
                        "plate": plate,
                        "confidence": confidence,
                        "plate_like": plate_like,
                        "frame_number": frame_number,
                        "frame_jpeg": encode_video_scan_preview(scan_frame),
                        "rank": rank,
                    }

            if progress_callback and (sample_index % 5 == 0 or sample_index + 1 == sample_count):
                progress_callback(sample_index + 1, sample_count)

            frames_to_skip = min(frame_stride - 1, frame_count - frame_number - 1)
            for _ in range(frames_to_skip):
                if not cap.grab():
                    break
            frame_number += frame_stride
    finally:
        cap.release()

    results = sorted(found.values(), key=lambda item: item["rank"], reverse=True)
    return results, results_truncated


def capture_local_camera(camera_index: int) -> np.ndarray:
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    last_error = ""
    for backend in backends:
        cap = cv2.VideoCapture(camera_index, backend)
        try:
            if not cap.isOpened():
                last_error = f"OpenCV không mở được camera số {camera_index}."
                continue
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            frame = None
            for _ in range(8):
                ok, candidate = cap.read()
                if ok and candidate is not None and candidate.size:
                    frame = candidate
            if frame is not None:
                return frame
            last_error = "Camera đã mở nhưng không trả về được khung hình."
        finally:
            cap.release()
    raise RuntimeError(
        f"{last_error} Đóng ứng dụng khác đang dùng webcam, kiểm tra quyền camera của Windows "
        "trong Settings > Privacy & security > Camera, rồi thử lại."
    )


def process_confirmed_event(plate: str, direction: str, frame: np.ndarray, source_type: str, confidence: float = 0.0) -> dict:
    normalized_plate = normalize_plate(plate)
    if not normalized_plate:
        raise ValueError("Vui lòng xác nhận biển số trước khi ghi nhận.")
    if is_dup_event_recent(normalized_plate, direction, seconds=25):
        raise ValueError("Xe này vừa được ghi nhận cùng hướng trong 25 giây gần đây.")
    image_path = save_frame_evidence(frame)
    return add_event(
        normalized_plate,
        direction,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        source_type,
        image_path,
        confidence,
    )


def record_live_camera_event(
    plate: str,
    direction: str,
    frame: np.ndarray,
    confidence: float,
) -> str:
    if is_dup_event_recent(plate, direction, seconds=25):
        return f"{plate}: lượt trùng trong 25 giây đã được lọc."

    event = process_confirmed_event(
        plate,
        direction,
        frame,
        "live_camera",
        confidence,
    )
    log_audit(
        "access_event_created",
        "access_event",
        str(event["id"]),
        f"{event['plate_number']} direction={direction} "
        f"source=live_camera confidence={confidence:.3f} auto=true",
    )
    return (
        f"ĐÃ TỰ GHI NHẬN {event['plate_number']} · "
        f"{'XE VÀO' if direction == 'IN' else 'XE RA'} · OCR {confidence:.0%}"
    )


@st.fragment
def render_live_camera(direction: str, pair_code: str | None = None) -> None:
    st.caption(
        "Bật camera trình duyệt bằng nút bên dưới. App nhận khung hình qua HTTPS mỗi khoảng 1 giây, "
        "tự lưu biển số đúng định dạng khi OCR đạt từ 65% trong ít nhất 2 lần đọc; "
        "các lượt cùng biển số/hướng trong 25 giây được lọc trùng."
    )
    st.caption(
        "Khung xanh XE khoanh thân xe; khung xanh BIEN SO khoanh biển số. "
        "Video truyền qua kết nối HTTPS của ứng dụng, không sử dụng WebRTC, STUN hoặc TURN."
    )

    camera_key = f"{pair_code or 'local'}_{direction}"
    processor_key = f"browser_camera_processor_{camera_key}"
    processor = st.session_state.get(processor_key)
    if processor is None:
        frame_publisher = (
            (lambda frame: camera_relay.publish(pair_code, frame))
            if pair_code
            else None
        )
        processor = AutomaticPlateProcessor(
            detector=None,
            ocr_service=None,
            direction=direction,
            record_event=record_live_camera_event,
            require_vehicle_detection=True,
            background_processing=True,
            frame_publisher=frame_publisher,
            model_loader=get_live_camera_models,
        )
        st.session_state[processor_key] = processor

    capture = browser_camera(
        key=f"browser_camera_{camera_key}",
        camera_facing_mode="environment",
        max_width=640,
        capture_interval_ms=1100,
    )
    if capture and capture.get("error"):
        st.error(f"Không mở được camera: {capture['error']}")
    elif capture and capture.get("frame"):
        frame_key = f"browser_camera_last_timestamp_{camera_key}"
        timestamp = capture.get("timestamp")
        if timestamp != st.session_state.get(frame_key):
            try:
                frame = decode_browser_camera_frame(capture)
                annotated = processor.process(frame)
            except (ValueError, cv2.error) as exc:
                st.error(f"Không xử lý được ảnh camera: {exc}")
                return
            st.session_state[frame_key] = timestamp
            st.session_state[f"browser_camera_last_image_{camera_key}"] = annotated
        image = st.session_state.get(f"browser_camera_last_image_{camera_key}")
        if image is not None:
            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                caption="Hình camera đã nhận qua kết nối HTTPS",
                use_container_width=True,
            )
        st.success(
            "Camera đang truyền — lần đầu có thể cần thời gian tải mô hình; "
            "biển số hợp lệ sẽ tự ghi vào lịch sử."
        )
        if processor.messages:
            st.caption(" · ".join(processor.messages))
    else:
        st.info("Bấm **Bật camera** bên trên và cho phép trình duyệt truy cập camera.")


def render_remote_camera_link() -> None:
    st.markdown("### Liên kết camera điện thoại với máy tính")
    st.caption(
        "Điện thoại gửi ảnh camera đến ứng dụng qua HTTPS; máy tính dùng cùng mã để xem hình. "
        "Không cần kết nối WebRTC, STUN hoặc TURN. Cả hai thiết bị cần mở ứng dụng và đăng nhập."
    )
    role = st.radio(
        "Thiết bị này sẽ",
        ["Phát camera từ điện thoại", "Xem camera trên máy tính"],
        horizontal=True,
        key="remote_camera_role",
    )

    if role == "Phát camera từ điện thoại":
        pair_code = st.session_state.get("remote_camera_pair_code")
        if not pair_code:
            if st.button(
                "Tạo mã liên kết camera",
                type="primary",
                use_container_width=True,
                key="create_remote_camera_pair",
            ):
                try:
                    pair_code = camera_relay.create_pairing_code()
                except RuntimeError as exc:
                    st.error(f"Không tạo được liên kết camera: {exc}")
                else:
                    st.session_state["remote_camera_pair_code"] = pair_code
                    st.rerun()
            return

        st.success("Mã liên kết đã sẵn sàng. Nhập mã này trên máy tính để xem camera.")
        st.code(pair_code, language="text")
        direction = st.radio(
            "Hướng của camera",
            ["IN", "OUT"],
            horizontal=True,
            format_func=lambda value: "Chỉ ghi XE VÀO" if value == "IN" else "Chỉ ghi XE RA",
            key="remote_automatic_camera_direction",
        )
        if st.button("Thu hồi mã liên kết", key="revoke_remote_camera_pair"):
            camera_relay.revoke_pairing_code(pair_code)
            st.session_state.pop("remote_camera_pair_code", None)
            st.rerun()
        render_live_camera(direction, pair_code=pair_code)
        return

    with st.form("remote_camera_viewer_form"):
        entered_code = st.text_input(
            "Mã liên kết từ điện thoại",
            max_chars=camera_relay.CODE_LENGTH,
            help="Nhập mã đang hiển thị trên điện thoại.",
        ).strip().upper()
        connect = st.form_submit_button("Kết nối camera", type="primary")
    if connect:
        if not camera_relay.has_pairing_code(entered_code):
            st.error("Mã không hợp lệ hoặc đã hết hạn. Tạo mã mới trên điện thoại.")
        else:
            st.session_state["remote_camera_viewer_code"] = entered_code

    pair_code = st.session_state.get("remote_camera_viewer_code")
    if pair_code:
        if not camera_relay.has_pairing_code(pair_code):
            st.session_state.pop("remote_camera_viewer_code", None)
            st.warning("Mã liên kết đã hết hạn. Tạo mã mới trên điện thoại.")
        else:
            @st.fragment(run_every=1)
            def render_remote_camera_frame() -> None:
                if not camera_relay.has_pairing_code(pair_code):
                    st.warning("Mã liên kết đã hết hạn. Tạo mã mới trên điện thoại.")
                    return
                frame, is_live = camera_relay.latest_frame(pair_code)
                if frame is None:
                    st.info("Đang chờ điện thoại bật camera và gửi khung hình...")
                elif is_live:
                    st.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        caption="Camera điện thoại · truyền qua HTTPS",
                        use_container_width=True,
                    )
                else:
                    st.warning("Không nhận được khung hình mới. Kiểm tra camera trên điện thoại.")

            render_remote_camera_frame()


def reset_live_capture() -> None:
    st.session_state.pop("dashboard_capture_bytes", None)
    st.session_state.pop("dashboard_scan_requested", None)
    st.session_state.pop("dashboard_upload_key", None)
    st.session_state.pop("dashboard_analysis", None)


def analyze_frame(frame: np.ndarray, fingerprint: str) -> dict:
    cached = st.session_state.get("dashboard_analysis")
    if cached and cached.get("fingerprint") == fingerprint:
        return cached

    detector = get_plate_detector()
    detections = detector.detect(frame)
    analysis = {
        "fingerprint": fingerprint,
        "detections": detections,
        "annotated": detector.draw_boxes(frame, detections),
        "results": [],
        "detector_error": detector.model_error,
        "ocr_error": "",
    }
    if detections:
        ocr_service = get_ocr_service()
        if not ocr_service._loaded:
            analysis["ocr_error"] = (
                ocr_service.load_error
                or "EasyOCR chưa sẵn sàng. Kiểm tra cài đặt và quyền tải trọng số."
            )
        else:
            for detection in detections:
                crop = detector.crop_plate(frame, detection)
                result = ocr_service.recognize(crop)
                analysis["results"].append({
                    "crop": crop,
                    "text": str(result.get("text") or ""),
                    "confidence": float(result.get("confidence") or 0),
                    "plate_like": bool(result.get("plate_like")),
                    "error": str(result.get("error") or ""),
                })
    st.session_state["dashboard_analysis"] = analysis
    return analysis


def render_overview() -> None:
    stats = get_dashboard_stats()
    today_events = list_events(date=datetime.now().strftime("%Y-%m-%d"))
    capacity_setting = get_setting("parking_capacity")
    try:
        capacity_value = int(capacity_setting) if capacity_setting else 0
        capacity = capacity_value if capacity_value > 0 else None
    except ValueError as exc:
        st.error(f"Giá trị sức chứa đã lưu không hợp lệ: {exc}")
        capacity = None
    metric_columns = st.columns(3)
    metric_columns[0].metric("Tổng xe trong bãi", stats["active_vehicles"])
    metric_columns[1].metric(
        "Xe vào hôm nay",
        sum(event["direction"] == "IN" and event["status"] == "confirmed" for event in today_events),
    )
    if capacity is None:
        metric_columns[2].metric("Chỗ trống", "—")
        metric_columns[2].caption("Cấu hình sức chứa để tính")
    else:
        metric_columns[2].metric(
            "Chỗ trống",
            max(capacity - stats["active_vehicles"], 0),
            delta=f"Sức chứa {capacity}",
            delta_color="off",
        )

    input_column, history_column, detail_column = st.columns([0.9, 1.05, 0.78], gap="medium")
    recent_events = get_recent_events(12)
    vehicles = list_vehicles()
    vehicle_by_plate = {vehicle["plate_number"]: vehicle for vehicle in vehicles}

    upload_hash = ""
    with input_column:
        st.markdown('<div class="monitor-panel-title">Camera giám sát</div>', unsafe_allow_html=True)
        input_source = st.radio(
            "Nguồn hình ảnh",
            [
                "Camera trực tiếp (tự động)",
                "Liên kết camera từ xa",
                "Ảnh/video (thủ công)",
            ],
            horizontal=True,
            key="live_input_source",
            on_change=reset_live_capture,
        )
        live_camera_mode = input_source == "Camera trực tiếp (tự động)"
        if live_camera_mode:
            live_direction = st.radio(
                "Hướng của camera",
                ["IN", "OUT"],
                horizontal=True,
                format_func=lambda value: "Chỉ ghi XE VÀO" if value == "IN" else "Chỉ ghi XE RA",
                key="automatic_camera_direction",
            )
            render_live_camera(live_direction)
        elif input_source == "Liên kết camera từ xa":
            render_remote_camera_link()
        else:
            uploaded_file = st.file_uploader(
                "Chọn JPG/PNG hoặc MP4/AVI",
                type=["jpg", "jpeg", "png", "mp4", "avi"],
                key="live_image_video_upload",
            )
            if uploaded_file is not None:
                try:
                    upload_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
                    frame_number = None
                    if uploaded_file.type.lower().startswith("video"):
                        video_path = save_uploaded_file(uploaded_file)
                        cap = cv2.VideoCapture(video_path)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                        if frame_count <= 0:
                            raise ValueError("Không xác định được số khung hình trong video.")
                        frame_number = st.slider(
                            "Khung hình video",
                            min_value=0,
                            max_value=frame_count - 1,
                            value=0,
                            key=f"live_video_frame_{upload_hash}",
                        )
                        if st.button(
                            "Quét tìm biển số trong toàn video",
                            key=f"scan_video_{upload_hash}",
                            use_container_width=True,
                        ):
                            progress = st.progress(0, text="Đang quét video...")
                            try:
                                scan_results, scan_results_truncated = scan_video_for_plates(
                                    video_path,
                                    lambda current, total: progress.progress(
                                        current / total,
                                        text=f"Đang quét khung hình {current}/{total}...",
                                    ),
                                )
                                st.session_state["dashboard_video_scan"] = {
                                    "signature": upload_hash,
                                    "results": scan_results,
                                    "truncated": scan_results_truncated,
                                }
                                st.session_state.pop("dashboard_video_scan_message", None)
                            except (cv2.error, MemoryError, OSError, RuntimeError, ValueError) as exc:
                                st.error(f"Không quét được video: {exc}")
                            finally:
                                progress.empty()
                        uploaded_frame = read_frame_from_video(video_path, frame_number)
                        capture_source = "uploaded_video"
                    else:
                        uploaded_rgb = np.array(
                            Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
                        )
                        uploaded_frame = cv2.cvtColor(uploaded_rgb, cv2.COLOR_RGB2BGR)
                        capture_source = "uploaded_image"
                    upload_key = (upload_hash, frame_number)
                    if st.session_state.get("dashboard_upload_key") != upload_key:
                        success, encoded = cv2.imencode(".jpg", uploaded_frame)
                        if not success:
                            raise ValueError("Không mã hóa được ảnh đầu vào.")
                        st.session_state["dashboard_capture_bytes"] = encoded.tobytes()
                        st.session_state["dashboard_capture_source"] = capture_source
                        st.session_state["dashboard_scan_requested"] = True
                        st.session_state["dashboard_upload_key"] = upload_key
                        st.session_state.pop("dashboard_analysis", None)
                except (cv2.error, OSError, ValueError) as exc:
                    st.error(f"Không đọc được tệp đầu vào: {exc}")

    pending_capture = st.session_state.get("dashboard_capture_bytes")
    current_frame = None
    current_analysis = None
    current_result_index = 0
    current_plate = ""
    current_confidence = 0.0
    current_plate_like = False
    current_crop = None
    current_fingerprint = ""
    if pending_capture and st.session_state.get("dashboard_scan_requested"):
        current_rgb = np.array(Image.open(io.BytesIO(pending_capture)).convert("RGB"))
        current_frame = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2BGR)
        current_fingerprint = hashlib.sha256(current_frame.tobytes()).hexdigest()
        with st.spinner("Đang phát hiện và đọc biển số..."):
            current_analysis = analyze_frame(current_frame, current_fingerprint)

    with input_column:
        if current_frame is not None and current_analysis:
            st.image(
                cv2.cvtColor(current_analysis["annotated"], cv2.COLOR_BGR2RGB),
                caption="Camera giám sát · vùng phát hiện được khoanh",
                use_container_width=True,
            )
            if current_analysis["detector_error"]:
                st.caption(f"Trạng thái mô hình: {current_analysis['detector_error']}")
            if current_analysis["ocr_error"]:
                st.error(current_analysis["ocr_error"])
            elif current_analysis["detections"]:
                ranking = sorted(
                    range(len(current_analysis["results"])),
                    key=lambda index: (
                        current_analysis["results"][index]["plate_like"],
                        current_analysis["results"][index]["confidence"],
                    ),
                    reverse=True,
                )
                current_result_index = ranking[0]
                if len(ranking) > 1:
                    current_result_index = st.selectbox(
                        "Chọn vùng biển số",
                        ranking,
                        format_func=lambda index: (
                            current_analysis["results"][index]["text"]
                            or f"Vùng {index + 1}"
                        ),
                        key=f"live_detection_{current_fingerprint}",
                    )
                result = current_analysis["results"][current_result_index]
                current_plate = result["text"]
                current_confidence = result["confidence"]
                current_plate_like = result["plate_like"]
                current_crop = result["crop"]
                crop_height, crop_width = current_crop.shape[:2]
                scale = min(8.0, max(2.0, 240.0 / max(1, crop_height)))
                zoom_crop = cv2.resize(
                    current_crop,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
                st.markdown("**Ảnh zoom biển số**")
                st.image(
                    cv2.cvtColor(zoom_crop, cv2.COLOR_BGR2RGB),
                    caption=f"Vùng gốc {crop_width}×{crop_height} px",
                    use_container_width=True,
                )
            else:
                st.warning("Không phát hiện được biển số trong ảnh. Hãy căn gần và chụp lại.")
        elif not live_camera_mode:
            latest_image = next(
                (
                    event for event in recent_events
                    if image_exists(event.get("image_path", ""))
                ),
                None,
            )
            if latest_image:
                st.image(
                    image_content(latest_image["image_path"]),
                    caption=f"Ảnh gần nhất · {latest_image['plate_number']}",
                    use_container_width=True,
                )
            else:
                st.info("Chọn camera hoặc tải ảnh/video để bắt đầu giám sát.")

    with history_column:
        st.markdown('<div class="monitor-panel-title">Lịch sử xe ra vào vừa qua</div>', unsafe_allow_html=True)
        if live_camera_mode:
            @st.fragment(run_every=3)
            def render_live_history() -> None:
                live_events = get_recent_events(12)
                if live_events:
                    history_rows = [{
                        "Biển số xe": event["plate_number"],
                        "Thời gian": event["event_time"],
                        "Sự kiện": "Xe vào" if event["direction"] == "IN" else "Xe ra",
                        "Độ tin cậy OCR": f"{float(event.get('confidence') or 0):.0%}",
                        "Trạng thái": "Đã ghi nhận" if event["status"] == "confirmed" else "Chờ xác minh",
                    } for event in live_events]
                    st.dataframe(
                        pd.DataFrame(history_rows),
                        use_container_width=True,
                        hide_index=True,
                        height=330,
                    )
                else:
                    st.info("Chưa có sự kiện xe ra vào được ghi nhận.")

            render_live_history()
            selected_event = recent_events[0] if recent_events else None
        elif recent_events:
            history_rows = [{
                "Biển số xe": event["plate_number"],
                "Thời gian": event["event_time"],
                "Sự kiện": "Xe vào" if event["direction"] == "IN" else "Xe ra",
                "Độ tin cậy OCR": f"{float(event.get('confidence') or 0):.0%}",
                "Trạng thái": "Đã ghi nhận" if event.get("status") == "confirmed" else "Chờ xác minh",
            } for event in recent_events]
            st.dataframe(
                pd.DataFrame(history_rows),
                use_container_width=True,
                hide_index=True,
                height=330,
            )
            event_options = {event["id"]: event for event in recent_events}
            new_event_id = st.session_state.get("dashboard_new_event_id")
            if new_event_id in event_options and st.session_state.get("dashboard_auto_selected_id") != new_event_id:
                st.session_state["dashboard_selected_event_id"] = new_event_id
                st.session_state["dashboard_auto_selected_id"] = new_event_id
            if st.session_state.get("dashboard_selected_event_id") not in event_options:
                st.session_state["dashboard_selected_event_id"] = recent_events[0]["id"]
            selected_event_id = st.selectbox(
                "Xem chi tiết sự kiện",
                list(event_options),
                format_func=lambda event_id: (
                    f"{event_options[event_id]['plate_number']} · "
                    f"{'Xe vào' if event_options[event_id]['direction'] == 'IN' else 'Xe ra'} · "
                    f"{event_options[event_id]['event_time']}"
                ),
                key="dashboard_selected_event_id",
                label_visibility="collapsed",
            )
            selected_event = event_options[selected_event_id]
        else:
            selected_event = None
            st.info("Chưa có sự kiện xe ra vào được ghi nhận.")

    with detail_column:
        st.markdown('<div class="monitor-panel-title">Chi tiết sự kiện</div>', unsafe_allow_html=True)
        if live_camera_mode:
            st.success(
                "Camera quét mỗi 1 giây và tự ghi nhận; không cần bấm xác nhận. "
                "Biển số dưới ngưỡng OCR sẽ không được lưu."
            )
            st.caption(
                "Hướng camera: XE VÀO" if live_direction == "IN" else "Hướng camera: XE RA"
            )

            @st.fragment(run_every=3)
            def render_live_camera_events() -> None:
                live_events = [
                    event
                    for event in get_recent_events(100)
                    if event.get("source_type") == "live_camera"
                ][:5]
                if not live_events:
                    st.info("Sự kiện nhận diện tự động sẽ xuất hiện ở đây.")
                    return
                st.dataframe(
                    pd.DataFrame([{
                        "Biển số": event["plate_number"],
                        "Hướng": "Xe vào" if event["direction"] == "IN" else "Xe ra",
                        "Thời gian": event["event_time"],
                        "OCR": f"{float(event.get('confidence') or 0):.0%}",
                    } for event in live_events]),
                    use_container_width=True,
                    hide_index=True,
                )

            render_live_camera_events()
        elif current_analysis and current_analysis["ocr_error"]:
            st.warning("OCR chưa sẵn sàng; kiểm tra trạng thái mô hình ở Cấu hình.")
        elif current_analysis and current_analysis["detections"] and current_crop is not None:
            plate_key = f"live_confirmed_plate_{current_fingerprint}_{current_result_index}"
            if plate_key not in st.session_state:
                st.session_state[plate_key] = (
                    current_plate if current_plate_like and current_confidence >= 0.25 else ""
                )
            corrected_plate = st.text_input("Biển số", key=plate_key)
            vehicle = get_vehicle_by_plate(corrected_plate) if corrected_plate else None
            movement = get_latest_confirmed_event(corrected_plate) if corrected_plate else None
            st.caption("THỜI GIAN")
            st.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            st.caption("ĐỘ TIN CẬY OCR")
            st.write(f"{current_confidence:.0%}")
            st.caption("ĐĂNG KÝ")
            st.write(
                "Đã đăng ký" if vehicle and vehicle.get("is_registered")
                else "Chưa đăng ký" if vehicle
                else "Chưa có hồ sơ"
            )
            if vehicle:
                st.caption("THÔNG TIN XE")
                st.write(
                    f"{vehicle.get('vehicle_type') or 'Loại xe chưa cập nhật'} · "
                    f"{vehicle.get('color') or 'Màu chưa cập nhật'}"
                )
                st.write(vehicle.get("owner_name") or "Chủ xe chưa cập nhật")
            if movement:
                st.caption(
                    "ĐANG TRONG KHU VỰC"
                    if movement["direction"] == "IN"
                    else "ĐÃ RA KHỎI KHU VỰC"
                )
            if not current_plate_like:
                st.warning("OCR chưa đủ căn cứ; hãy sửa biển số và đối chiếu ảnh.")
            elif current_confidence < 0.55:
                st.warning("Độ tin cậy thấp; cần người vận hành kiểm tra.")
            confirmed = st.checkbox(
                "Đã kiểm tra biển số",
                key=f"live_confirm_operator_{current_fingerprint}_{current_result_index}",
            )
            event_direction = st.radio(
                "Hướng di chuyển",
                ["IN", "OUT"],
                horizontal=True,
                format_func=lambda value: "XE VÀO" if value == "IN" else "XE RA",
                key="live_event_direction",
            )
            if st.button(
                "Ghi nhận sự kiện",
                type="primary",
                key=f"live_record_{current_fingerprint}_{current_result_index}",
                disabled=not confirmed or not normalize_plate(corrected_plate),
                use_container_width=True,
            ):
                try:
                    event = process_confirmed_event(
                        corrected_plate,
                        event_direction,
                        current_frame,
                        st.session_state.get("dashboard_capture_source", "local_camera"),
                        current_confidence,
                    )
                    log_audit(
                        "access_event_created",
                        "access_event",
                        str(event["id"]),
                        f"{event['plate_number']} direction={event_direction} "
                        f"source={event.get('source_type')} confidence={current_confidence:.3f}",
                    )
                    st.session_state["dashboard_new_event_id"] = event["id"]
                    st.success("Đã ghi nhận sự kiện và lưu ảnh bằng chứng.")
                    st.rerun()
                except ValueError as exc:
                    st.warning(str(exc))
                except (OSError, sqlite3.Error) as exc:
                    st.error(f"Không lưu được sự kiện/ảnh bằng chứng: {exc}")
        elif selected_event:
            vehicle = vehicle_by_plate.get(selected_event["plate_number"])
            st.markdown(
                f'<div class="plate-ticket">{selected_event["plate_number"]}</div>',
                unsafe_allow_html=True,
            )
            st.caption("THỜI GIAN")
            st.write(selected_event["event_time"])
            st.caption("SỰ KIỆN")
            st.write("XE VÀO" if selected_event["direction"] == "IN" else "XE RA")
            st.caption("TRẠNG THÁI")
            st.write("Đã ghi nhận" if selected_event["status"] == "confirmed" else "Chờ xác minh")
            st.caption("ĐĂNG KÝ")
            st.write(
                "Đã đăng ký" if vehicle and vehicle.get("is_registered")
                else "Chưa đăng ký" if vehicle
                else "Chưa có hồ sơ"
            )
            if vehicle:
                st.caption("THÔNG TIN XE")
                st.write(
                    f"{vehicle.get('vehicle_type') or 'Loại xe chưa cập nhật'} · "
                    f"{vehicle.get('color') or 'Màu chưa cập nhật'}"
                )
                st.write(vehicle.get("owner_name") or "Chủ xe chưa cập nhật")
            image_path = selected_event.get("image_path")
            if image_exists(image_path):
                st.image(image_content(image_path), caption="Ảnh sự kiện", use_container_width=True)
            if selected_event.get("status") != "confirmed":
                corrected_plate = st.text_input(
                    "Xác nhận biển số",
                    value=selected_event["plate_number"],
                    key=f"dashboard_review_plate_{selected_event['id']}",
                )
                if st.button("Xác nhận sự kiện", type="primary", key=f"dashboard_review_{selected_event['id']}"):
                    try:
                        review_event(selected_event["id"], corrected_plate)
                        st.success("Đã xác nhận sự kiện.")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
        else:
            st.info("Thông tin sự kiện sẽ hiển thị tại đây sau khi có lượt xe.")

    video_scan = st.session_state.get("dashboard_video_scan")
    if video_scan and upload_hash and video_scan.get("signature") == upload_hash:
        scan_results = video_scan.get("results", [])
        st.divider()
        st.subheader("Kết quả quét toàn bộ video")
        st.caption(
            "Ứng dụng lấy mẫu video tối đa 2 khung hình/giây (tối đa 300 khung hình), "
            f"chạy detector ở 416 px, gộp kết quả trùng và giữ tối đa "
            f"{MAX_VIDEO_SCAN_RESULTS} ảnh xem trước đã nén. Chỉ hiển thị OCR có định dạng "
            "giống biển số; vẫn cần kiểm tra ký tự, nhất là khi confidence thấp."
        )
        if video_scan.get("truncated"):
            st.warning(
                f"Video có nhiều kết quả OCR khác nhau; chỉ hiển thị "
                f"{MAX_VIDEO_SCAN_RESULTS} kết quả có độ tin cậy cao nhất để giảm tải bộ nhớ."
            )
        if not scan_results:
            st.info("Không tìm thấy biển số đọc được trong video đã quét.")
        else:
            st.write(
                f"Tìm thấy {len(scan_results)} biển số khác nhau. "
                "Sửa biển số nếu cần, rồi tích “Đã kiểm tra biển số này” cho từng xe muốn lưu."
            )
            with st.form(key=f"confirm_video_scan_{video_scan['signature']}"):
                for index, item in enumerate(scan_results):
                    with st.container(border=True):
                        preview_column, fields_column = st.columns([1, 1.5])
                        preview_column.image(
                            item["frame_jpeg"],
                            caption=f"Khung hình {item['frame_number']}",
                            use_container_width=True,
                        )
                        with fields_column:
                            corrected_plate = st.text_input(
                                f"Biển số {index + 1}",
                                value=item["plate"],
                                key=f"video_scan_plate_{video_scan['signature']}_{index}",
                            )
                            st.caption(
                                f"OCR {item['confidence']:.0%} · "
                                f"{'Định dạng phù hợp' if item['plate_like'] else 'Cần kiểm tra'}"
                            )
                            st.checkbox(
                                "Đã kiểm tra biển số này",
                                key=f"video_scan_confirm_{video_scan['signature']}_{index}",
                            )
                            st.selectbox(
                                "Hướng di chuyển",
                                ["IN", "OUT"],
                                format_func=lambda value: "XE VÀO" if value == "IN" else "XE RA",
                                key=f"video_scan_direction_{video_scan['signature']}_{index}",
                            )

                submitted = st.form_submit_button(
                    "Ghi nhận các biển số đã xác nhận",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:
                recorded = []
                skipped = []
                errors = []
                selected_count = sum(
                    bool(
                        st.session_state.get(
                            f"video_scan_confirm_{video_scan['signature']}_{index}",
                            False,
                        )
                    )
                    for index in range(len(scan_results))
                )
                if selected_count == 0:
                    st.session_state["dashboard_video_scan_message"] = {
                        "recorded": [],
                        "skipped": [],
                        "errors": [],
                        "selection_required": True,
                    }
                    st.rerun()

                for index, item in enumerate(scan_results):
                    if not st.session_state.get(
                        f"video_scan_confirm_{video_scan['signature']}_{index}",
                        False,
                    ):
                        continue
                    plate = normalize_plate(
                        st.session_state.get(
                            f"video_scan_plate_{video_scan['signature']}_{index}",
                            "",
                        )
                    )
                    direction = st.session_state.get(
                        f"video_scan_direction_{video_scan['signature']}_{index}",
                        "IN",
                    )
                    if not plate:
                        errors.append(f"Biển số thứ {index + 1} chưa hợp lệ.")
                        continue
                    if is_dup_event_recent(plate, direction, seconds=25):
                        skipped.append(f"{plate}: vừa được ghi nhận cùng hướng trong 25 giây.")
                        continue
                    try:
                        frame = decode_video_scan_preview(item["frame_jpeg"])
                        event = process_confirmed_event(
                            plate,
                            direction,
                            frame,
                            "uploaded_video",
                            item["confidence"],
                        )
                        log_audit(
                            "access_event_created",
                            "access_event",
                            str(event["id"]),
                            f"{event['plate_number']} direction={direction} "
                            f"source=uploaded_video confidence={item['confidence']:.3f}",
                        )
                        recorded.append(event["plate_number"])
                    except (cv2.error, OSError, sqlite3.Error, ValueError) as exc:
                        errors.append(f"{plate}: {exc}")

                st.session_state["dashboard_video_scan_message"] = {
                    "recorded": recorded,
                    "skipped": skipped,
                    "errors": errors,
                    "selection_required": False,
                }

            message = st.session_state.get("dashboard_video_scan_message")
            if message:
                if message.get("selection_required"):
                    st.warning(
                        "Chưa chọn biển số nào. Hãy tích “Đã kiểm tra biển số này” "
                        "cho ít nhất một xe rồi bấm ghi nhận."
                    )
                if message["recorded"]:
                    st.success(f"Đã ghi nhận: {', '.join(message['recorded'])}.")
                if message["skipped"]:
                    st.warning("Không ghi nhận một số biển số: " + " ".join(message["skipped"]))
                if message["errors"]:
                    st.error("Có lỗi khi lưu: " + " ".join(message["errors"]))
                if (
                    not message.get("selection_required")
                    and not message["recorded"]
                    and not message["skipped"]
                    and not message["errors"]
                ):
                    st.info("Chưa có sự kiện nào được ghi nhận.")

def render_plate_recognition(
    frame_override: np.ndarray | None = None,
    source_type_override: str = "",
) -> None:
    frame = frame_override
    source_type = source_type_override or ""
    if frame is not None:
        st.markdown("## Kết quả nhận diện")
    else:
        source = st.radio(
            "Nguồn đầu vào",
            ["Tải ảnh / video", "Camera máy tính"],
            horizontal=True,
            key="recognition_source",
        )
    if frame is None and source == "Tải ảnh / video":
        source_type = "uploaded_image"
        uploaded_file = st.file_uploader(
            "Chọn ảnh JPG/PNG hoặc video MP4/AVI",
            type=["jpg", "jpeg", "png", "mp4", "avi"],
            key="plate_file_upload",
        )
        if uploaded_file is None:
            st.info("Tải ảnh hoặc video để bắt đầu.")
            return
        if uploaded_file.type.lower().startswith("video"):
            source_type = "uploaded_video"
            saved_path = save_uploaded_file(uploaded_file)
            cap = cv2.VideoCapture(saved_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if frame_count <= 0:
                st.error("Không xác định được số khung hình trong video.")
                return
            frame_number = st.slider(
                "Chọn khung hình để nhận diện",
                min_value=0,
                max_value=frame_count - 1,
                value=0,
                key=f"video_frame_{Path(uploaded_file.name).name}",
            )
            try:
                frame = read_frame_from_video(saved_path, frame_number)
            except ValueError as exc:
                st.error(str(exc))
                return
        elif frame is None:
            source_type = "uploaded_image"
            image_rgb = np.array(Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB"))
            frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    elif frame is None:
        source_type = "local_camera"
        camera_index = st.number_input(
            "Camera trên máy tính",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            key="server_camera_index",
        )
        if not st.button("Bật camera và chụp", type="primary", key="capture_server_camera"):
            st.info("Camera được mở trực tiếp bằng OpenCV trên máy đang chạy ứng dụng.")
            return
        try:
            frame = capture_local_camera(int(camera_index))
        except RuntimeError as exc:
            st.error(str(exc))
            return

    if frame is None or frame.size == 0:
        st.error("Không đọc được ảnh từ nguồn đã chọn.")
        return

    detector = get_plate_detector()
    detections = detector.detect(frame)
    if not detections:
        st.warning("Không phát hiện được biển số. Hãy thử chụp gần hơn, giữ camera ổn định và đảm bảo đủ ánh sáng.")
        return

    if detector.model_error:
        st.warning(f"Không nạp được mô hình detector; đang dùng OpenCV contour. Chi tiết: {detector.model_error}")
    elif detector.mode != "YOLO plate detector":
        st.info("Chưa có trọng số YOLO chuyên phát hiện biển số; kết quả định vị hiện dùng OpenCV contour.")

    annotated = detector.draw_boxes(frame, detections)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_img = Image.fromarray(image_rgb)
    annotated_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    with st.spinner("Đang khởi tạo OCR (lần đầu có thể cần tải trọng số)..."):
        ocr_service = get_ocr_service()
    if not ocr_service._loaded:
        st.error(f"OCR chưa sẵn sàng. {ocr_service.load_error or 'Kiểm tra EasyOCR và quyền tải model.'}")
        if st.button("Thử tải lại mô hình OCR", key="retry_ocr_load"):
            get_ocr_service.clear()
            st.rerun()
        return

    result_rows = []
    for detection in detections:
        crop = detector.crop_plate(frame, detection)
        result = ocr_service.recognize(crop)
        crop_height, crop_width = crop.shape[:2]
        result_rows.append({
            "x1": detection["x1"],
            "y1": detection["y1"],
            "x2": detection["x2"],
            "y2": detection["y2"],
            "Kích thước vùng": f"{crop_width}×{crop_height}",
            "text": result.get("text", ""),
            "OCR confidence": result.get("confidence", 0),
            "Định dạng biển số": "Có vẻ phù hợp" if result.get("plate_like") else "Cần kiểm tra",
            "Detector confidence": detection.get("confidence"),
            "error": result.get("error", ""),
            "method": detection.get("method", detector.mode),
        })

    recognition_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    def detection_rank(index: int) -> tuple[float, float]:
        row = result_rows[index]
        detector_conf = row["Detector confidence"]
        detector_score = float(detector_conf) if detector_conf is not None else 0.0
        ocr_score = float(row["OCR confidence"] or 0.0)
        format_score = 1.0 if row["Định dạng biển số"] == "Có vẻ phù hợp" else 0.0
        return format_score, 0.6 * ocr_score + 0.4 * detector_score

    ranked_detection_indices = sorted(
        range(len(result_rows)),
        key=detection_rank,
        reverse=True,
    )
    plausible_detections = [
        index
        for index in ranked_detection_indices
        if result_rows[index]["Định dạng biển số"] == "Có vẻ phù hợp"
    ]
    best_result_index = plausible_detections[0] if plausible_detections else ranked_detection_indices[0]
    best_plate = (
        str(result_rows[best_result_index]["text"])
        if (
            plausible_detections
            and float(result_rows[best_result_index]["OCR confidence"] or 0) >= 0.25
        )
        else ""
    )

    if len(detections) > 1:
        selected_detection_index = st.selectbox(
            "Chọn biển số cần ghi nhận",
            list(range(len(detections))),
            index=best_result_index,
            format_func=lambda index: (
                f"Biển số {index + 1} · {result_rows[index]['text'] or 'chưa đọc được'}"
            ),
            key=f"selected_ocr_detection_{hashlib.sha256(frame.tobytes()).hexdigest()}",
        )
    else:
        selected_detection_index = 0

    selected_result = result_rows[selected_detection_index]
    selected_ocr_text = str(selected_result["text"]).strip()
    selected_confidence = float(selected_result["OCR confidence"] or 0)
    selected_plate_like = selected_result["Định dạng biển số"] == "Có vẻ phù hợp"
    fingerprint = hashlib.sha256(frame.tobytes()).hexdigest()
    plate_input_key = f"confirmed_plate_{fingerprint}_{selected_detection_index}"
    if plate_input_key not in st.session_state:
        st.session_state[plate_input_key] = best_plate if selected_detection_index in plausible_detections else ""
    selected_crop = detector.crop_plate(frame, detections[selected_detection_index])
    crop_height, crop_width = selected_crop.shape[:2]
    preview_scale = min(8.0, max(2.0, 240.0 / max(1, crop_height)))
    enlarged_crop = cv2.resize(
        selected_crop,
        None,
        fx=preview_scale,
        fy=preview_scale,
        interpolation=cv2.INTER_CUBIC,
    )

    detail_column, evidence_column = st.columns([1, 1.15], gap="large")
    best_confidence = selected_confidence
    with detail_column:
        st.markdown("### Chi tiết sự kiện")
        with st.container(border=True):
            st.caption("BIỂN SỐ XE")
            st.markdown(
                f'<div class="plate-ticket">{selected_ocr_text or "CHƯA ĐỌC ĐƯỢC"}</div>',
                unsafe_allow_html=True,
            )
            direction = st.selectbox(
                "Loại sự kiện",
                ["IN", "OUT"],
                format_func=lambda value: "XE VÀO" if value == "IN" else "XE RA",
                key="recognized_event_direction",
            )
            result_metrics = st.columns(3)
            result_metrics[0].metric("Thời gian", recognition_time)
            result_metrics[1].metric("Loại sự kiện", "Xe vào" if direction == "IN" else "Xe ra")
            result_metrics[2].metric(
                "Tin cậy OCR",
                f"{selected_confidence:.0%}",
            )
            result_metrics[0].markdown(
                f"**Định dạng**  \n{'Có vẻ phù hợp' if selected_plate_like else 'Cần xác minh'}"
            )
            corrected = st.text_input("Xác nhận hoặc sửa biển số", key=plate_input_key)
            vehicle_status = get_vehicle_by_plate(corrected) if corrected else None
            latest_movement = get_latest_confirmed_event(corrected) if corrected else None
            registration_label = (
                "Đã đăng ký" if vehicle_status and vehicle_status.get("is_registered")
                else "Chưa đăng ký" if vehicle_status
                else "Chưa có hồ sơ"
            )
            result_metrics[1].markdown(f"**Đăng ký**  \n{registration_label}")
            if vehicle_status:
                info_columns = st.columns(2)
                info_columns[0].markdown(f"**Loại xe**  \n{vehicle_status.get('vehicle_type') or 'Chưa cập nhật'}")
                info_columns[1].markdown(f"**Màu xe**  \n{vehicle_status.get('color') or 'Chưa cập nhật'}")
                info_columns[0].markdown(f"**Chủ phương tiện**  \n{vehicle_status.get('owner_name') or 'Chưa cập nhật'}")
                info_columns[1].markdown(f"**Số điện thoại**  \n{vehicle_status.get('phone_number') or 'Chưa cập nhật'}")
                if vehicle_status.get("is_restricted"):
                    st.warning("Phương tiện đang bị hạn chế ra vào — cần người quản lý kiểm tra.")
                elif vehicle_status.get("is_registered") and vehicle_status.get("is_allowed"):
                    st.success("Đã đăng ký và được phép ra vào.")
                elif vehicle_status.get("is_registered"):
                    st.warning("Đã đăng ký nhưng chưa được cấp quyền ra vào.")
                else:
                    st.info("Chưa có hồ sơ đăng ký.")
            elif corrected:
                st.info("Chưa có hồ sơ phương tiện cho biển số này.")
            if latest_movement:
                movement_label = (
                    "ĐANG TRONG KHU VỰC"
                    if latest_movement["direction"] == "IN"
                    else "ĐÃ RA KHỎI KHU VỰC"
                )
                st.caption(f"Trạng thái hiện tại: {movement_label} · {latest_movement['event_time']}")
            if not selected_plate_like:
                st.warning("Kết quả OCR chưa đủ căn cứ xác nhận biển số; hãy kiểm tra ảnh và sửa nếu cần.")
            elif selected_confidence < 0.55:
                st.warning("Độ tin cậy thấp; hãy đối chiếu ký tự với ảnh biển số.")
            if crop_width < 80 or crop_height < 32:
                st.warning(
                    f"Vùng biển số chỉ có {crop_width}×{crop_height} px; ảnh gốc có thể chưa đủ chi tiết."
                )
            confirmed_by_operator = st.checkbox(
                "Tôi đã kiểm tra ảnh và xác nhận biển số chính xác",
                key=f"operator_confirm_{fingerprint}_{selected_detection_index}",
            )
            if st.button(
                "Xác nhận và ghi nhận",
                type="primary",
                key="record_recognized_event",
                disabled=not confirmed_by_operator or not normalize_plate(corrected),
                use_container_width=True,
            ):
                try:
                    event = process_confirmed_event(corrected, direction, frame, source_type, best_confidence)
                    log_audit(
                        "access_event_created",
                        "access_event",
                        str(event["id"]),
                        f"{event['plate_number']} direction={direction} source={source_type} confidence={best_confidence:.3f}",
                    )
                    st.session_state["last_recognition_event"] = {
                        "frame_fingerprint": fingerprint,
                        "plate_number": event["plate_number"],
                        "event_time": event["event_time"],
                        "direction": direction,
                        "confidence": best_confidence,
                        "image_path": event.get("image_path"),
                    }
                    st.success(
                        f"Đã ghi nhận {event['plate_number']} — "
                        f"{'XE VÀO' if direction == 'IN' else 'XE RA'} lúc {event['event_time']}."
                    )
                except ValueError as exc:
                    st.warning(str(exc))
                except OSError as exc:
                    st.error(f"Không lưu được ảnh minh chứng: {exc}")

    with evidence_column:
        st.markdown("### Ảnh bằng chứng")
        st.image(
            annotated_img,
            caption="Khung phát hiện trên ảnh đầu vào",
            use_container_width=True,
        )
        st.image(
            cv2.cvtColor(enlarged_crop, cv2.COLOR_BGR2RGB),
            caption=f"Ảnh biển số đã cắt · ảnh gốc {crop_width}×{crop_height} px",
            use_container_width=True,
        )
        with st.expander("Xem ảnh gốc"):
            st.image(original_img, use_container_width=True)
        st.caption("Khung phát hiện không đảm bảo OCR đọc đúng; hãy đối chiếu ảnh trước khi xác nhận.")

    saved_event = st.session_state.get("last_recognition_event")
    if (
        saved_event
        and saved_event.get("frame_fingerprint") == fingerprint
        and saved_event.get("plate_number") == normalize_plate(corrected)
    ):
        st.markdown("### Thông tin xe ra vào")
        with st.container(border=True):
            event_detail, event_evidence = st.columns([1, 1.1], gap="large")
            with event_detail:
                st.markdown(
                    f'<div class="plate-ticket">{saved_event["plate_number"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Thời gian:** {saved_event['event_time']}")
                st.markdown(
                    f"**Loại sự kiện:** {'Xe vào' if saved_event['direction'] == 'IN' else 'Xe ra'}"
                )
                st.markdown(f"**Độ tin cậy OCR:** {saved_event['confidence']:.0%}")
                st.markdown(
                    f"**Trạng thái:** {'Đang trong khu vực' if saved_event['direction'] == 'IN' else 'Đã ra khỏi khu vực'}"
                )
                st.markdown(
                    f"**Đăng ký:** {'Đã đăng ký' if vehicle_status and vehicle_status.get('is_registered') else 'Chưa đăng ký' if vehicle_status else 'Chưa có hồ sơ'}"
                )
            with event_evidence:
                image_path = saved_event.get("image_path")
                if image_exists(image_path):
                    st.image(image_content(image_path), caption="Ảnh bằng chứng sự kiện", use_container_width=True)
                else:
                    st.info("Không có ảnh bằng chứng.")

def render_access_log() -> None:
    st.subheader("Tạo sự kiện mới")
    with st.container(border=True):
        direction = st.radio(
            "Hướng di chuyển",
            ["IN", "OUT"],
            format_func=lambda value: "XE VÀO" if value == "IN" else "XE RA",
            horizontal=True,
            key="manual_event_direction",
        )
        plate_number = st.text_input("Biển số đã xác nhận", placeholder="Ví dụ: 51A12345", key="manual_event_plate")
        plate = normalize_plate(plate_number)
        vehicle = get_vehicle_by_plate(plate) if plate else None
        if plate and vehicle:
            if vehicle.get("is_restricted"):
                st.warning("Hồ sơ đang bị đánh dấu hạn chế ra vào. Cần người quản lý kiểm tra; đây không phải kết luận vi phạm.")
            elif vehicle["is_allowed"]:
                st.success(f"Phương tiện đã đăng ký và được phép ra vào — {vehicle.get('owner_name') or 'chưa có tên chủ xe'}.")
            else:
                st.warning("Xe đã có hồ sơ nhưng chưa được cấp quyền ra vào.")
        elif plate:
            st.info("Biển số này chưa có trong hồ sơ phương tiện.")
        event_status = st.radio(
            "Trạng thái xác minh",
            ["confirmed", "needs_review"],
            format_func=lambda value: "Đã xác nhận" if value == "confirmed" else "Đánh dấu cần xác minh",
            horizontal=True,
            key="manual_event_status",
        )
        if event_status == "needs_review":
            st.warning("Sự kiện sẽ được lưu chờ xác minh và không ảnh hưởng tới số xe đang trong khu vực.")

        evidence_mode = st.radio(
            "Ảnh minh chứng",
            ["Không đính kèm", "Tải ảnh lên", "Chụp bằng camera"],
            horizontal=True,
            key="manual_event_evidence_mode",
        )
        evidence_file = None
        if evidence_mode == "Tải ảnh lên":
            evidence_file = st.file_uploader(
                "Ảnh minh chứng JPG/PNG",
                type=["jpg", "jpeg", "png"],
                key="manual_event_evidence_file",
            )
        elif evidence_mode == "Chụp bằng camera":
            evidence_file = st.camera_input("Chụp ảnh minh chứng", key="manual_event_camera")

        notes = st.text_area("Ghi chú", key="manual_event_notes")
        if st.button("Lưu sự kiện ra/vào", type="primary", key="save_manual_event"):
            if not plate:
                st.error("Không lưu được sự kiện nếu chưa xác nhận biển số.")
            elif is_dup_event_recent(plate, direction, seconds=25):
                st.warning("Xe này vừa được ghi nhận cùng hướng trong 25 giây gần đây; đã ngăn ghi lặp.")
            else:
                try:
                    image_path = save_evidence_file(evidence_file) if evidence_file else ""
                    event = add_event(
                        plate,
                        direction,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "manual",
                        image_path,
                        0.0,
                        notes,
                        status=event_status,
                        review_reason="Operator marked for review" if event_status == "needs_review" else "",
                    )
                    log_audit(
                        "access_event_created",
                        "access_event",
                        str(event["id"]),
                        f"{plate} direction={direction} source=manual status={event_status}",
                    )
                    st.session_state["last_manual_event_id"] = event["id"]
                    st.success(f"Đã ghi nhận {plate} — {'XE VÀO' if direction == 'IN' else 'XE RA'}.")
                except (OSError, ValueError) as exc:
                    st.error(f"Không thể lưu sự kiện: {exc}")

    recent_events = get_recent_events(30)
    vehicle_by_plate = {vehicle["plate_number"]: vehicle for vehicle in list_vehicles()}
    last_manual_event_id = st.session_state.get("last_manual_event_id")
    last_manual_event = next(
        (event for event in recent_events if event["id"] == last_manual_event_id),
        None,
    )
    if last_manual_event:
        event_vehicle = vehicle_by_plate.get(last_manual_event["plate_number"])
        with st.container(border=True):
            st.markdown("### Thông tin xe ra vào vừa ghi nhận")
            event_detail, event_evidence = st.columns([1, 1.1], gap="large")
            with event_detail:
                st.markdown(
                    f'<div class="plate-ticket">{last_manual_event["plate_number"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Thời gian:** {last_manual_event['event_time']}")
                st.markdown(
                    f"**Loại sự kiện:** {'Xe vào' if last_manual_event['direction'] == 'IN' else 'Xe ra'}"
                )
                st.markdown(
                    f"**Trạng thái:** {'Đã xác nhận' if last_manual_event['status'] == 'confirmed' else 'Chờ xác minh'}"
                )
                st.markdown(
                    f"**Đăng ký:** {'Đã đăng ký' if event_vehicle and event_vehicle.get('is_registered') else 'Chưa đăng ký' if event_vehicle else 'Chưa có hồ sơ'}"
                )
                st.markdown(f"**Ghi chú:** {last_manual_event.get('notes') or '—'}")
            with event_evidence:
                image_path = last_manual_event.get("image_path")
                if image_exists(image_path):
                    st.image(image_content(image_path), caption="Ảnh bằng chứng sự kiện", use_container_width=True)
                else:
                    st.info("Sự kiện này chưa có ảnh bằng chứng.")

    st.divider()
    st.subheader("Thông tin xe ra vào")
    active_vehicles = list_active_vehicles()
    summary = st.columns(3)
    summary[0].metric("Xe đang trong khu vực", len(active_vehicles))
    summary[1].metric("Lượt ghi nhận gần đây", len(recent_events))
    summary[2].metric(
        "Sự kiện chờ xác minh",
        sum(event.get("status") != "confirmed" for event in list_attention_events()),
    )

    recent_tab, active_tab, review_tab = st.tabs(
        ["Lượt xe mới nhất", "Xe đang trong khu vực", "Chờ xác minh"]
    )
    with recent_tab:
        if recent_events:
            event_rows = []
            for event in recent_events:
                vehicle = vehicle_by_plate.get(event["plate_number"])
                event_rows.append({
                    "Biển số": event["plate_number"],
                    "Thời gian": event["event_time"],
                    "Hướng": "Xe vào" if event["direction"] == "IN" else "Xe ra",
                    "Loại xe": vehicle.get("vehicle_type") or "Chưa cập nhật" if vehicle else "Chưa có hồ sơ",
                    "Chủ xe": vehicle.get("owner_name") or "Chưa cập nhật" if vehicle else "Chưa có hồ sơ",
                    "Đăng ký": (
                        "Đã đăng ký" if vehicle and vehicle.get("is_registered")
                        else "Chưa đăng ký" if vehicle
                        else "Chưa có hồ sơ"
                    ),
                    "Tin cậy OCR": f"{float(event.get('confidence') or 0):.0%}",
                    "Xác minh": "Đã xác nhận" if event.get("status") == "confirmed" else "Cần xác minh",
                })
            st.dataframe(pd.DataFrame(event_rows), use_container_width=True, hide_index=True)
            with st.expander("Ảnh bằng chứng của các lượt xe gần đây"):
                images_shown = False
                for event in recent_events:
                    image_path = event.get("image_path")
                    if image_exists(image_path):
                        st.image(
                            image_content(image_path),
                            caption=(
                                f"{event['plate_number']} · "
                                f"{'Xe vào' if event['direction'] == 'IN' else 'Xe ra'} · "
                                f"{event['event_time']}"
                            ),
                            width=420,
                        )
                        images_shown = True
                if not images_shown:
                    st.info("Các lượt xe gần đây chưa có ảnh bằng chứng.")
        else:
            st.info("Chưa có lượt xe nào được ghi nhận.")

    with active_tab:
        if active_vehicles:
            active_rows = [{
                "Biển số": vehicle["plate_number"],
                "Loại xe": vehicle.get("vehicle_type") or "Chưa cập nhật",
                "Màu xe": vehicle.get("color") or "Chưa cập nhật",
                "Chủ xe": vehicle.get("owner_name") or "Chưa cập nhật",
                "Số điện thoại": vehicle.get("phone_number") or "Chưa cập nhật",
                "Thời gian vào gần nhất": vehicle["last_event_time"],
                "Đăng ký": "Đã đăng ký" if vehicle.get("is_registered") else "Chưa đăng ký",
            } for vehicle in active_vehicles]
            st.dataframe(pd.DataFrame(active_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Hiện không có xe nào đang ở trong khu vực theo các lượt đã xác minh.")

    with review_tab:
        pending_events = [
            event for event in list_attention_events()
            if event.get("status") != "confirmed"
        ]
        if pending_events:
            pending_rows = [{
                "Biển số cần kiểm tra": event["plate_number"],
                "Thời gian": event["event_time"],
                "Hướng": "Xe vào" if event["direction"] == "IN" else "Xe ra",
                "Ghi chú xác minh": event.get("review_reason") or "Chờ người quản lý xác nhận",
            } for event in pending_events]
            st.dataframe(pd.DataFrame(pending_rows), use_container_width=True, hide_index=True)
            st.caption("Các sự kiện này chưa được tính vào danh sách xe đang trong khu vực.")
        else:
            st.success("Không có sự kiện nào đang chờ xác minh.")


def render_vehicle_management() -> None:
    st.subheader("Quản lý phương tiện")
    search_term = st.text_input("Tìm kiếm phương tiện", value="")
    vehicles = list_vehicles(search_term)
    vehicle_by_id = {item["id"]: item for item in vehicles}
    selection = st.selectbox(
        "Chọn hồ sơ để sửa hoặc tạo mới",
        [None, *vehicle_by_id.keys()],
        format_func=lambda item_id: "＋ Tạo hồ sơ mới" if item_id is None else f"{vehicle_by_id[item_id]['plate_number']} · {vehicle_by_id[item_id].get('owner_name') or 'chưa có chủ xe'}",
        key="vehicle_edit_selection",
    )
    selected = vehicle_by_id.get(selection) if selection is not None else None
    suffix = str(selection) if selection is not None else "new"
    with st.form("vehicle_form"):
        plate_number = st.text_input("Biển số", value=selected["plate_number"] if selected else "", key=f"vehicle_plate_{suffix}")
        vehicle_type = st.text_input("Loại xe", value=selected.get("vehicle_type") or "" if selected else "", key=f"vehicle_type_{suffix}")
        color = st.text_input("Màu xe", value=selected.get("color") or "" if selected else "", key=f"vehicle_color_{suffix}")
        owner_name = st.text_input("Chủ phương tiện", value=selected.get("owner_name") or "" if selected else "", key=f"vehicle_owner_{suffix}")
        phone_number = st.text_input("Số điện thoại (không bắt buộc)", value=selected.get("phone_number") or "" if selected else "", key=f"vehicle_phone_{suffix}")
        notes = st.text_area("Ghi chú", value=selected.get("notes") or "" if selected else "", key=f"vehicle_notes_{suffix}")
        is_allowed = st.checkbox("Được phép ra vào", value=bool(selected["is_allowed"]) if selected else False, key=f"vehicle_allowed_{suffix}")
        is_registered = st.checkbox("Đã đăng ký", value=bool(selected["is_registered"]) if selected else False, key=f"vehicle_registered_{suffix}")
        is_restricted = st.checkbox("Hạn chế ra vào", value=bool(selected.get("is_restricted")) if selected else False, key=f"vehicle_restricted_{suffix}")
        submitted = st.form_submit_button("Lưu thay đổi" if selected else "Thêm phương tiện", type="primary")

        if submitted:
            plate = normalize_plate(plate_number)
            if not plate:
                st.error("Biển số không được để trống.")
            else:
                try:
                    upsert_vehicle({
                        "plate_number": plate,
                        "vehicle_type": vehicle_type,
                        "color": color,
                        "owner_name": owner_name,
                        "phone_number": phone_number,
                        "notes": notes,
                        "is_allowed": is_allowed,
                        "is_registered": is_registered,
                        "is_restricted": is_restricted,
                    })
                    log_audit(
                        "vehicle_updated" if selected else "vehicle_created",
                        "vehicle",
                        plate,
                        f"type={vehicle_type}; registered={is_registered}; allowed={is_allowed}; restricted={is_restricted}",
                    )
                    st.success(f"Đã lưu hồ sơ phương tiện {plate}.")
                    st.rerun()
                except (ValueError, OSError) as exc:
                    st.error(f"Không thể lưu hồ sơ xe: {exc}")

    if vehicles:
        st.dataframe(pd.DataFrame(vehicles), use_container_width=True, hide_index=True)
        if selected:
            delete_confirmed = st.checkbox(
                f"Xác nhận xóa hồ sơ {selected['plate_number']}",
                key=f"confirm_delete_{selected['id']}",
            )
            if st.button("Xóa hồ sơ đã chọn", disabled=not delete_confirmed, key="delete_selected_vehicle"):
                delete_vehicle(selected["id"])
                log_audit("vehicle_deleted", "vehicle", selected["plate_number"], "Deleted vehicle profile")
                st.success(f"Đã xóa hồ sơ {selected['plate_number']}.")
                st.rerun()
    else:
        st.info("Không tìm thấy phương tiện nào.")


def render_history() -> None:
    st.subheader("Lịch sử ra vào")
    today = date.today()
    date_range = st.date_input(
        "Khoảng ngày",
        value=(today - timedelta(days=30), today),
        key="history_date_range",
    )
    filters = st.columns([1.3, 1, 1, 1])
    with filters[0]:
        plate_filter = st.text_input("Biển số", key="history_plate_filter")
    with filters[1]:
        direction_filter = st.selectbox(
            "Hướng",
            ["", "IN", "OUT"],
            format_func=lambda value: "Tất cả" if not value else ("Xe vào" if value == "IN" else "Xe ra"),
            key="history_direction_filter",
        )
    with filters[2]:
        show_evidence = st.checkbox("Hiện ảnh minh chứng", key="history_show_evidence")
    with filters[3]:
        review_filter = st.selectbox(
            "Xác minh",
            ["", "confirmed", "needs_review"],
            format_func=lambda value: "Tất cả" if not value else ("Đã xác minh" if value == "confirmed" else "Cần xác minh"),
            key="history_review_filter",
        )
    if not isinstance(date_range, tuple) or len(date_range) != 2:
        st.info("Chọn đủ ngày bắt đầu và ngày kết thúc.")
        return
    date_from, date_to = date_range
    if date_from > date_to:
        st.error("Ngày bắt đầu phải nhỏ hơn hoặc bằng ngày kết thúc.")
        return

    time_filters = st.columns(2)
    time_from = time_filters[0].time_input("Từ giờ", value=datetime.strptime("00:00", "%H:%M").time())
    time_to = time_filters[1].time_input("Đến giờ", value=datetime.strptime("23:59", "%H:%M").time())
    if time_from > time_to:
        st.error("Giờ bắt đầu phải nhỏ hơn hoặc bằng giờ kết thúc.")
        return
    events = list_events_range(
        date_from.isoformat(),
        date_to.isoformat(),
        plate_filter,
        direction_filter,
        time_from.strftime("%H:%M:%S"),
        time_to.strftime("%H:%M:%S"),
        review_filter,
    )
    if not events:
        st.info("Không có lịch sử phù hợp.")
    else:
        st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)
        if show_evidence:
            with st.expander("Ảnh minh chứng trong kết quả"):
                for event in events:
                    image_path = event.get("image_path")
                    if image_exists(image_path):
                        st.image(
                            image_content(image_path),
                            caption=f"{event['plate_number']} · {'Xe vào' if event['direction'] == 'IN' else 'Xe ra'} · {event['event_time']}",
                            width=480,
                        )
                    elif image_path:
                        st.warning(f"Không tìm thấy ảnh minh chứng: {image_path}")

    review_queue = list_attention_events()
    unresolved = [event for event in review_queue if event.get("status") != "confirmed"]
    if unresolved:
        st.subheader(f"Sự kiện cần xác minh ({len(unresolved)})")
        for event in unresolved:
            with st.expander(f"#{event['id']} · {event['plate_number']} · {event['event_time']}"):
                if image_exists(event.get("image_path", "")):
                    st.image(image_content(event["image_path"]), width=480)
                st.write(f"Độ tin cậy OCR: {event.get('confidence', 0):.3f}")
                corrected_plate = st.text_input(
                    "Biển số người quản lý xác nhận",
                    value=event["plate_number"],
                    key=f"review_plate_{event['id']}",
                )
                if st.button("Xác nhận sự kiện", key=f"review_event_{event['id']}"):
                    try:
                        reviewed = review_event(event["id"], corrected_plate)
                        st.success(f"Đã xác nhận biển số {reviewed['plate_number']}.")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))


def render_reports() -> None:
    st.subheader("Thống kê và báo cáo")
    today = date.today()
    date_range = st.date_input(
        "Khoảng thời gian báo cáo",
        value=(today - timedelta(days=30), today),
        key="report_date_range",
    )
    if not isinstance(date_range, tuple) or len(date_range) != 2:
        st.info("Chọn đủ ngày bắt đầu và ngày kết thúc.")
        return
    date_from, date_to = date_range
    if date_from > date_to:
        st.error("Ngày bắt đầu phải nhỏ hơn hoặc bằng ngày kết thúc.")
        return

    events = list_events_range(date_from.isoformat(), date_to.isoformat())
    df = pd.DataFrame(events)
    if df.empty:
        st.info("Không có sự kiện trong khoảng thời gian này.")
    else:
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
        df = df.dropna(subset=["event_time"])
        df["Ngày"] = df["event_time"].dt.date
        df["Hướng"] = df["direction"].map({"IN": "Xe vào", "OUT": "Xe ra"})
        daily = df.groupby(["Ngày", "Hướng"]).size().unstack(fill_value=0)
        st.bar_chart(daily)
        summary = df.groupby("Hướng").size().rename("Số lượt").reset_index()
        st.dataframe(summary, use_container_width=True, hide_index=True)

    export_name = f"history_{date_from.isoformat()}_to_{date_to.isoformat()}"
    export_key = f"report_export_{date_from.isoformat()}_{date_to.isoformat()}"
    if st.button("Tạo báo cáo CSV và Excel", type="primary", key="generate_report"):
        export_file = EXPORT_DIR / export_name
        try:
            csv_path = export_history_csv(date_from.isoformat(), date_to.isoformat(), str(export_file.with_suffix(".csv")))
            xlsx_path = export_history_excel(date_from.isoformat(), date_to.isoformat(), str(export_file.with_suffix(".xlsx")))
            st.session_state[export_key] = {
                "csv_path": csv_path,
                "xlsx_path": xlsx_path,
            }
            log_audit("report_created", "report", export_name, f"from={date_from}; to={date_to}; rows={len(events)}")
            st.success("Đã tạo báo cáo CSV và Excel.")
        except (OSError, ValueError, ImportError) as exc:
            st.error(f"Không thể tạo báo cáo: {exc}")

    report_files = st.session_state.get(export_key)
    if report_files:
        try:
            with open(report_files["csv_path"], "rb") as csv_file:
                csv_bytes = csv_file.read()
            with open(report_files["xlsx_path"], "rb") as xlsx_file:
                xlsx_bytes = xlsx_file.read()
            downloads = st.columns(2)
            downloads[0].download_button(
                "Tải báo cáo CSV",
                data=csv_bytes,
                file_name=Path(report_files["csv_path"]).name,
                mime="text/csv",
                type="primary",
            )
            downloads[1].download_button(
                "Tải báo cáo Excel",
                data=xlsx_bytes,
                file_name=Path(report_files["xlsx_path"]).name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.caption(f"Bản báo cáo được lưu trong thư mục exports với tiền tố {export_name}.")
        except OSError as exc:
            st.error(f"Không đọc được file báo cáo để tải xuống: {exc}")
            st.session_state.pop(export_key, None)


def render_ai_training() -> None:
    st.subheader("Đánh giá độ chính xác OCR")
    st.write("Tải CSV có hai cột `ground_truth` và `predicted`, mỗi dòng là một ảnh đã được người dùng gán nhãn và kết quả OCR tương ứng.")
    evaluation_file = st.file_uploader("CSV đối chiếu nhãn thật và OCR", type=["csv"], key="ocr_evaluation_csv")
    if evaluation_file is not None:
        try:
            evaluation = pd.read_csv(evaluation_file, dtype=str).fillna("")
            required = {"ground_truth", "predicted"}
            if not required.issubset(evaluation.columns):
                st.error("CSV cần có hai cột bắt buộc: ground_truth và predicted.")
            elif evaluation.empty:
                st.warning("CSV không có dòng dữ liệu để đánh giá.")
            else:
                evaluation["ground_truth"] = evaluation["ground_truth"].map(normalize_plate)
                evaluation["predicted"] = evaluation["predicted"].map(normalize_plate)
                evaluation = evaluation[evaluation["ground_truth"] != ""]
                evaluation["exact_match"] = evaluation["ground_truth"] == evaluation["predicted"]
                tp = int(evaluation["exact_match"].sum())
                fp = int(((~evaluation["exact_match"]) & (evaluation["predicted"] != "")).sum())
                fn = int((evaluation["predicted"] == "").sum())
                accuracy = evaluation["exact_match"].mean() if len(evaluation) else 0.0
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                metric_columns = st.columns(4)
                metric_columns[0].metric("Số mẫu", len(evaluation))
                metric_columns[1].metric("OCR khớp hoàn toàn", f"{accuracy:.1%}")
                metric_columns[2].metric("Precision exact-match", f"{precision:.1%}")
                metric_columns[3].metric("Recall exact-match", f"{recall:.1%}")
                st.dataframe(evaluation, use_container_width=True, hide_index=True)
                st.caption("Đây là chỉ số OCR exact-match trên CSV đã cung cấp, không phải Precision/Recall/mAP phát hiện hộp YOLO.")
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            st.error(f"Không đọc được CSV đánh giá: {exc}")

    st.divider()
    st.subheader("Mẫu ký tự OCR đã xác nhận")
    st.caption(
        "Chỉ gồm mẫu ảnh/nhãn do người vận hành chủ động xác nhận. Có thể tải CSV để đánh giá OCR "
        "hoặc tải ZIP ảnh cắt kèm nhãn; EasyOCR không tự được huấn luyện lại từ các mẫu này."
    )
    ocr_samples = list_ocr_samples(limit=1000)
    if ocr_samples:
        sample_rows = []
        for sample in ocr_samples:
            sample_rows.append({
                "id": sample["id"],
                "ground_truth": sample["confirmed_text"],
                "predicted": sample["predicted_text"],
                "confidence": sample["confidence"],
                "source": sample["source_type"],
                "created_at": sample["created_at"],
                "image_file": sample["image_path"].replace("\\", "/").rsplit("/", 1)[-1],
            })
        samples_df = pd.DataFrame(sample_rows)
        st.metric("Số mẫu đã xác nhận", len(ocr_samples))
        st.dataframe(samples_df, use_container_width=True, hide_index=True)
        download_column, zip_column = st.columns(2)
        download_column.download_button(
            "Tải CSV mẫu OCR",
            data=samples_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"ocr_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_ocr_samples_csv",
        )
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("labels.csv", samples_df.to_csv(index=False).encode("utf-8-sig"))
            for sample in ocr_samples:
                sample_path = sample["image_path"]
                if image_exists(sample_path):
                    suffix = PurePosixPath(sample_path).suffix.lower() or ".jpg"
                    archive.writestr(
                        f"images/{sample['id']}_{sample['confirmed_text']}{suffix}",
                        read_bytes(sample_path),
                    )
        zip_column.download_button(
            "Tải ZIP ảnh và nhãn",
            data=archive_buffer.getvalue(),
            file_name=f"ocr_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            key="download_ocr_samples_zip",
        )
    else:
        st.info("Chưa có mẫu OCR được xác nhận. Hãy lưu mẫu tại trang Nhận diện biển số.")

    st.divider()
    st.subheader("Huấn luyện / đánh giá YOLO")
    active_detector = MODEL_DIR / "plate_detector_yolov8n.pt"
    if active_detector.is_file():
        st.info(
            f"Detector hiện được cấu hình nạp từ `{active_detector}`. "
            "Trọng số này chỉ được thay thế sau khi một lượt huấn luyện mới hoàn tất thành công."
        )
    st.warning(
        "Trong thư mục dự án chưa có dataset được gán nhãn. YOLOv8n pretrained có thể dùng làm điểm khởi tạo để fine-tune, "
        "nhưng bản thân nó không phải detector biển số cho đến khi được huấn luyện trên dữ liệu biển số."
    )
    with st.expander("Cách chuẩn bị dữ liệu và huấn luyện", expanded=True):
        st.markdown(
            """
            **1. Chuẩn bị dataset có bounding box biển số** (không phải ảnh/video chưa gán nhãn). Mỗi ảnh cần file `.txt`
            cùng tên, mỗi dòng theo thứ tự `class_id x_center y_center width height`, tọa độ chuẩn hóa từ 0 đến 1.

            **2. Sắp xếp thư mục** và giữ train/val tách biệt:
            """
        )
        st.code(
            "datasets/plates/\n"
            "  images/train/     # ảnh huấn luyện\n"
            "  images/val/       # ảnh validation riêng\n"
            "  labels/train/     # nhãn .txt trùng tên ảnh\n"
            "  labels/val/       # nhãn .txt trùng tên ảnh\n"
            "  dataset.yaml",
            language="text",
        )
        st.markdown("**3. Tạo `dataset.yaml`**; chỉnh đường dẫn `path` cho đúng máy:")
        st.code(
            "path: C:/Users/Admin/Desktop/license_plate_system/datasets/plates\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: license_plate",
            language="yaml",
        )
        st.markdown(
            """
            **4. Chạy ngay tại trang này:** để trọng số khởi tạo là `yolov8n.pt` (Ultralytics sẽ tải trọng số chính thức
            nếu máy có Internet), kiểm tra đường dẫn `dataset.yaml`, chọn tỷ lệ train/số epoch/kích thước ảnh, xác nhận
            dataset rồi bấm **Bắt đầu huấn luyện trên CPU**. Có thể tải checkpoint `.pt` riêng lên thay cho `yolov8n.pt`.

            Kết quả tốt nhất được lưu thành `models\\plate_detector_yolov8n.pt`; log/checkpoint từng lượt nằm trong
            `runs\\train`. Sau khi huấn luyện xong, đánh giá mAP trên chính dataset có tập `val` để xem Precision,
            Recall và mAP. Tập test độc lập nên chỉ dùng để đánh giá cuối cùng.

            Dataset biển số Việt Nam tham khảo: [Kaggle - Vietnamese license plates](https://www.kaggle.com/datasets/tanhphp/vietnamese-license-plates).
            Kiểm tra điều khoản sử dụng và tải/xuất dataset thành YOLO format trước khi dùng; ảnh đơn thuần không có nhãn
            bounding box thì không thể dùng để train detector.
            """
        )

    initialization_path = MODEL_DIR / "training_init.pt"
    default_weights_value = str(initialization_path) if initialization_path.is_file() else "yolov8n.pt"
    weight_upload = st.file_uploader(
        "Tải checkpoint khởi tạo YOLO lên (tùy chọn, .pt)",
        type=["pt"],
        key="plate_weight_upload",
        help="Nếu không tải checkpoint riêng, dùng yolov8n.pt pretrained chính thức để fine-tune.",
    )
    if weight_upload is not None:
        weight_path = initialization_path
        weight_bytes = weight_upload.getvalue()
        weight_signature = hashlib.sha256(weight_bytes).hexdigest()
        if st.session_state.get("uploaded_training_model_signature") != weight_signature:
            try:
                weight_path.write_bytes(weight_bytes)
                st.session_state["uploaded_training_model_signature"] = weight_signature
                log_audit("training_checkpoint_uploaded", "ai_model", weight_path.name, f"bytes={weight_path.stat().st_size}")
                st.success(f"Đã lưu checkpoint khởi tạo: {weight_path}")
            except OSError as exc:
                st.error(f"Không lưu được trọng số mô hình: {exc}")

    default_weights = MODEL_DIR / "plate_detector_yolov8n.pt"
    default_dataset_yaml = BASE_DIR / "datasets" / "vietnamese-license-plates" / "splits" / "data.yaml"
    with st.form("yolo_train_form"):
        weights_path = st.text_input(
            "Checkpoint khởi tạo",
            value=default_weights_value,
            help="Để yolov8n.pt để Ultralytics tự tải checkpoint pretrained. Đây chỉ là điểm khởi tạo, không phải detector biển số hoàn chỉnh.",
        )
        dataset_yaml = st.text_input(
            "Đường dẫn dataset.yaml đã gán nhãn",
            value=str(default_dataset_yaml) if default_dataset_yaml.is_file() else "",
            placeholder="C:\\duong_dan\\datasets\\plates\\dataset.yaml",
        )
        train_fraction = st.select_slider(
            "Tỷ lệ tập train dùng trong lượt này",
            options=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
            value=0.1,
            format_func=lambda value: f"{value:.0%}",
            help="100% dùng toàn bộ dữ liệu. Trên CPU cũ, bắt đầu với tỷ lệ nhỏ để ước lượng thời gian.",
        )
        epochs = st.number_input("Epoch", min_value=1, max_value=500, value=5, step=1)
        image_size = st.selectbox("Kích thước ảnh", [320, 416, 512, 640], index=0)
        confirmed_dataset = st.checkbox("Tôi xác nhận ảnh train/val có nhãn bounding box biển số hợp lệ")
        train_requested = st.form_submit_button("Bắt đầu huấn luyện trên CPU", type="primary")
    if train_requested:
        weights = Path(weights_path).expanduser()
        data_yaml = Path(dataset_yaml).expanduser()
        official_ultralytics_weights = weights.name.lower() in {
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"
        }
        if not weights.is_file() and not official_ultralytics_weights:
            st.error("Không tìm thấy checkpoint .pt. Hãy kiểm tra đường dẫn hoặc dùng yolov8n.pt để tải pretrained chính thức.")
        elif not data_yaml.is_file() or data_yaml.suffix.lower() not in {".yaml", ".yml"}:
            st.error("Hãy chỉ tới file dataset.yaml hoặc dataset.yml tồn tại trên máy.")
        elif not confirmed_dataset:
            st.error("Cần xác nhận bộ dữ liệu có nhãn biển số trước khi chạy.")
        else:
            try:
                from ultralytics import YOLO

                run_name = datetime.now().strftime("plates_%Y%m%d_%H%M%S")
                with st.spinner(
                    f"YOLO đang huấn luyện trên CPU với {train_fraction:.0%} tập train. "
                    "Thời gian phụ thuộc phần cứng và kích thước dataset..."
                ):
                    model = YOLO(str(weights))
                    model.train(
                        data=str(data_yaml),
                        epochs=int(epochs),
                        imgsz=int(image_size),
                        device="cpu",
                        batch=4,
                        workers=0,
                        pretrained=True,
                        fraction=float(train_fraction),
                        project=str(BASE_DIR / "runs" / "train"),
                        name=run_name,
                    )
                best_weights = Path(model.trainer.best)
                if not best_weights.is_file():
                    raise FileNotFoundError("Huấn luyện xong nhưng không tìm thấy best.pt.")
                destination = MODEL_DIR / "plate_detector_yolov8n.pt"
                if best_weights.resolve() != destination.resolve():
                    shutil.copy2(best_weights, destination)
                get_plate_detector.clear()
                log_audit(
                    "yolo_training_finished",
                    "ai_model",
                    destination.name,
                    f"epochs={epochs}; fraction={train_fraction}; size={image_size}; init={weights}; data={data_yaml}",
                )
                st.success(f"Huấn luyện hoàn tất. Trọng số tốt nhất: {destination}")
                st.caption(f"Log và checkpoint gốc: {best_weights.parent}")
            except (ImportError, OSError, ValueError, RuntimeError, AttributeError, TypeError, KeyError, SyntaxError) as exc:
                st.error(f"Huấn luyện thất bại: {exc}")

    st.subheader("Đánh giá detector YOLO")
    with st.form("yolo_eval_form"):
        eval_weights = st.text_input("Trọng số cần đánh giá", value=str(default_weights), key="yolo_eval_weights")
        eval_yaml = st.text_input("Dataset YAML có tập val", value="", key="yolo_eval_yaml")
        eval_requested = st.form_submit_button("Chạy đánh giá mAP trên CPU")
    if eval_requested:
        weights = Path(eval_weights).expanduser()
        data_yaml = Path(eval_yaml).expanduser()
        if not weights.is_file() or not data_yaml.is_file():
            st.error("Cần chỉ tới file trọng số .pt và dataset YAML tồn tại.")
        else:
            try:
                from ultralytics import YOLO

                with st.spinner("Đang chạy đánh giá YOLO trên tập val..."):
                    model = YOLO(str(weights))
                    metrics = model.val(data=str(data_yaml), device="cpu")
                box_metrics = metrics.box
                score_cols = st.columns(4)
                score_cols[0].metric("Precision", f"{float(box_metrics.mp):.3f}")
                score_cols[1].metric("Recall", f"{float(box_metrics.mr):.3f}")
                score_cols[2].metric("mAP@0.50", f"{float(box_metrics.map50):.3f}")
                score_cols[3].metric("mAP@0.50:0.95", f"{float(box_metrics.map):.3f}")
                log_audit("yolo_validation_finished", "ai_model", weights.name, f"dataset={data_yaml}")
            except (ImportError, OSError, ValueError, RuntimeError, AttributeError, TypeError, KeyError, SyntaxError) as exc:
                st.error(f"Đánh giá YOLO thất bại: {exc}")


def render_train_page() -> None:
    import tempfile
    import yaml
    from train_plate_detector import (
        deploy_best_model,
        prepare_dataset_from_zip,
        train_plate_detector,
    )

    st.markdown(
        """
        Fine-tune detector YOLOv8 trên CPU. Dataset phải có bounding box biển số ở định dạng YOLO,
        với ảnh và nhãn tách riêng cho train/validation.
        """
    )
    st.info(
        """
        **Lưu ý khi chạy trên CPU/Cloud:** huấn luyện có thể mất nhiều thời gian và tiêu tốn tài nguyên.
        Bắt đầu với YOLOv8n, ảnh 416 px, batch 4 và ít epoch; không đóng trang trong khi đang train.
        """
    )
    with st.expander("1. Chuẩn bị dataset", expanded=True):
        st.markdown(
            """
            Tải dataset biển số YOLO, ví dụ [Kaggle – VNLicensePlate](https://www.kaggle.com/datasets/bomaich/vnlicenseplate),
            rồi nén thư mục dataset thành ZIP. ZIP cần có đúng một `data.yaml`, thư mục ảnh train/val
            và nhãn `.txt` tương ứng. Mỗi nhãn theo định dạng YOLO:
            `class_id x_center y_center width height` với tọa độ từ 0 đến 1.
            """
        )
        st.code(
            "dataset.zip\n"
            "├── data.yaml\n"
            "├── images/train/\n"
            "├── images/val/\n"
            "├── labels/train/\n"
            "└── labels/val/",
            language="text",
        )
        st.caption(
            "Ứng dụng sẽ kiểm tra đường dẫn ZIP và thư mục train/val, sau đó tự sửa `path` trong data.yaml "
            "theo vị trí giải nén. Không tải ảnh/video chưa có nhãn bounding box."
        )

    uploaded = st.file_uploader(
        "2. Upload dataset (.zip)",
        type=["zip"],
        key="plate_training_dataset_zip",
    )
    columns = st.columns(3)
    with columns[0]:
        epochs = st.number_input("Số epochs", min_value=1, max_value=100, value=5, step=1)
    with columns[1]:
        image_size = st.selectbox("Kích thước ảnh", [320, 416, 480, 640], index=1)
    with columns[2]:
        batch = st.selectbox("Batch size", [2, 4, 8], index=1)
    model_name = st.selectbox(
        "Model khởi tạo",
        ["yolov8n.pt", "yolov8s.pt"],
        index=0,
        help="yolov8n nhẹ hơn và phù hợp hơn với CPU.",
    )

    trained_artifact = st.session_state.get("trained_plate_model_download")
    if trained_artifact:
        st.download_button(
            "Tải model đã huấn luyện (.pt)",
            data=trained_artifact["content"],
            file_name=trained_artifact["filename"],
            mime="application/octet-stream",
            key="download_trained_plate_model",
            on_click="ignore",
        )

    if not st.button(
        "Bắt đầu huấn luyện",
        type="primary",
        use_container_width=True,
        disabled=uploaded is None,
        key="start_uploaded_plate_training",
    ):
        return

    dataset_bytes = uploaded.getvalue()
    if not dataset_bytes:
        st.error("File dataset ZIP rỗng.")
        return

    runs_dir = BASE_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    progress = st.progress(0)
    progress_text = st.empty()
    run_name = datetime.now().strftime("streamlit_%Y%m%d_%H%M%S")
    try:
        with tempfile.TemporaryDirectory(prefix="plate_dataset_", dir=runs_dir) as temp_dir:
            data_yaml = prepare_dataset_from_zip(
                dataset_bytes,
                Path(temp_dir) / "dataset",
            )
            progress_text.info(f"Đã xác thực dataset: `{data_yaml}`. Đang huấn luyện trên CPU...")

            def update_training_progress(epoch: int, total_epochs: int) -> None:
                progress.progress(min(99, int(epoch * 100 / total_epochs)))
                progress_text.info(f"Đang huấn luyện: epoch {epoch}/{total_epochs}")

            best_weights = train_plate_detector(
                data_yaml=data_yaml,
                epochs=int(epochs),
                imgsz=int(image_size),
                batch=int(batch),
                model_name=model_name,
                project=BASE_DIR / "runs" / "plate_detect",
                name=run_name,
                device="cpu",
                progress_callback=update_training_progress,
            )
        deployed = deploy_best_model(best_weights, MODEL_DIR / "plate_detector_yolov8n.pt")
        get_plate_detector.clear()
        progress.progress(100)
        progress_text.success(f"Huấn luyện hoàn tất. Model đang hoạt động: `{deployed}`")
        st.balloons()
        st.session_state["trained_plate_model_download"] = {
            "content": deployed.read_bytes(),
            "filename": deployed.name,
        }
        st.download_button(
            "Tải model đã huấn luyện (.pt)",
            data=st.session_state["trained_plate_model_download"]["content"],
            file_name=st.session_state["trained_plate_model_download"]["filename"],
            mime="application/octet-stream",
            key=f"download_plate_model_{run_name}",
            on_click="ignore",
        )
        st.caption(
            "Trên máy chủ có filesystem tạm thời, hãy tải model xuống và lưu giữ bản sao bên ngoài app."
        )
        try:
            log_audit(
                "yolo_training_finished",
                "ai_model",
                deployed.name,
                f"epochs={epochs}; imgsz={image_size}; batch={batch}; model={model_name}",
            )
        except sqlite3.Error as exc:
            st.warning(f"Model đã huấn luyện và được cài đặt, nhưng không ghi được nhật ký: {exc}")
    except (
        ImportError,
        OSError,
        ValueError,
        RuntimeError,
        zipfile.BadZipFile,
        yaml.YAMLError,
        AttributeError,
        TypeError,
        KeyError,
    ) as exc:
        progress_text.error(f"Huấn luyện thất bại: {exc}")
        st.exception(exc)


def create_backup_package(output_path: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="plate_backup_", dir=EXPORT_DIR) as temp_dir:
        database_path = Path(temp_dir) / "license_plate.db"
        backup_database(str(database_path))
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(database_path, "license_plate.db")
            archived_names = set()
            for source_dir, archive_root in (
                (EVIDENCE_DIR, "evidence"),
                (UPLOAD_DIR, "uploads"),
            ):
                for source_file in source_dir.rglob("*"):
                    if source_file.is_file():
                        archive_name = PurePosixPath(
                            archive_root,
                            *source_file.relative_to(source_dir).parts,
                        ).as_posix()
                        archive.write(source_file, archive_name)
                        archived_names.add(archive_name)
            for category in ("evidence", "uploads"):
                for relative_name, content in stored_objects(category):
                    archive_name = PurePosixPath(category, relative_name).as_posix()
                    if archive_name not in archived_names:
                        archive.writestr(archive_name, content)
    return str(output_path)


def restore_backup_package(uploaded_file) -> str:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="plate_restore_", dir=EXPORT_DIR) as temp_dir:
        staged_database = Path(temp_dir) / "license_plate.db"
        entries: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
        database_entries = 0
        uncompressed_size = 0

        with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue()), "r") as archive:
            for info in archive.infolist():
                name = info.filename
                if name == "license_plate.db":
                    database_entries += 1
                    if info.is_dir():
                        raise ValueError("File database trong bản sao lưu không hợp lệ.")
                    entries.append((info, PurePosixPath(name)))
                    uncompressed_size += info.file_size
                    if uncompressed_size > MAX_BACKUP_UNCOMPRESSED_BYTES:
                        raise ValueError("Bản sao lưu vượt quá giới hạn giải nén 2 GB.")
                    continue

                path = PurePosixPath(name)
                mode = info.external_attr >> 16
                if info.is_dir() and path.parts in {("evidence",), ("uploads",)}:
                    continue
                if (
                    path.is_absolute()
                    or len(path.parts) < 2
                    or path.parts[0] not in {"evidence", "uploads"}
                    or any(part in {"", ".", ".."} for part in path.parts)
                    or "\\" in name
                    or stat.S_ISLNK(mode)
                ):
                    raise ValueError(f"Đường dẫn file không an toàn trong bản sao lưu: {name}")
                entries.append((info, path))
                uncompressed_size += info.file_size
                if uncompressed_size > MAX_BACKUP_UNCOMPRESSED_BYTES:
                    raise ValueError("Bản sao lưu vượt quá giới hạn giải nén 2 GB.")

            if database_entries != 1:
                raise ValueError("Bản sao lưu phải chứa đúng một file license_plate.db.")

            database_info = next(info for info, path in entries if path.as_posix() == "license_plate.db")
            with archive.open(database_info) as source, staged_database.open("wb") as target:
                shutil.copyfileobj(source, target)
            validate_database_backup(str(staged_database))

            storage_dirs = {"evidence": EVIDENCE_DIR, "uploads": UPLOAD_DIR}
            for info, path in entries:
                if path.as_posix() == "license_plate.db" or info.is_dir():
                    continue
                content = archive.read(info)
                category = path.parts[0]
                relative_name = "/".join(path.parts[1:])
                if object_storage_enabled():
                    put_bytes(f"{category}/{relative_name}", content)
                    if category == "uploads":
                        target = storage_dirs[category].joinpath(*path.parts[1:])
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(content)
                else:
                    target = storage_dirs[category].joinpath(*path.parts[1:])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)

        evidence_root = (
            current_prefix_uri("evidence")
            if object_storage_enabled()
            else str(EVIDENCE_DIR)
        )
        restored_path = restore_database(str(staged_database), evidence_root)

    init_db()
    log_audit(
        "backup_restored",
        "backup",
        Path(uploaded_file.name).name,
        f"database={restored_path}",
    )
    return restored_path


def render_backup_audit() -> None:
    restore_message = st.session_state.pop("backup_restore_message", "")
    if restore_message:
        st.success(restore_message)

    st.subheader("Vị trí lưu dữ liệu")
    if database_backend_configured():
        st.write("Dữ liệu chính được lưu trên PostgreSQL đã cấu hình.")
    else:
        st.write("Dữ liệu chính được lưu trên máy đang chạy ứng dụng.")
    image_location = current_prefix_uri("evidence") if object_storage_enabled() else str(EVIDENCE_DIR)
    upload_location = current_prefix_uri("uploads") if object_storage_enabled() else str(UPLOAD_DIR)
    database_location = "PostgreSQL (DATABASE_URL)" if database_backend_configured() else str(DB_PATH)
    st.code(
        f"Database: {database_location}\nẢnh bằng chứng: {image_location}\nFile tải lên: {upload_location}",
        language="text",
    )
    if database_backend_configured():
        st.caption("Kết nối PostgreSQL đang được sử dụng.")
    elif DB_PATH.is_file():
        st.caption(f"Database SQLite hiện có ({DB_PATH.stat().st_size:,} byte).")
    else:
        st.warning("Chưa thấy file database; file sẽ được tạo khi ứng dụng khởi tạo.")
    st.caption("Thông tin xe/lịch sử nằm trong database; ảnh và file tải lên được lưu riêng.")

    st.divider()
    st.subheader("Sao lưu và khôi phục")
    st.write("Tạo một file ZIP gồm database, ảnh bằng chứng và file đã tải lên để quét.")
    if st.button("Tạo bản sao lưu", type="primary", key="create_backup"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = EXPORT_DIR / f"license_plate_backup_{timestamp}.zip"
        try:
            create_backup_package(backup_path)
            log_audit("backup_created", "backup", timestamp, f"package={backup_path.name}")
            st.session_state["latest_backup"] = str(backup_path)
            st.success("Đã tạo bản sao lưu database, ảnh bằng chứng và file tải lên.")
        except (OSError, sqlite3.Error, zipfile.BadZipFile, ValueError) as exc:
            backup_path.unlink(missing_ok=True)
            st.error(f"Tạo bản sao lưu thất bại: {exc}")

    backup_path = st.session_state.get("latest_backup")
    if backup_path and Path(backup_path).is_file():
        try:
            with open(backup_path, "rb") as backup_file:
                st.download_button(
                    "Tải bản sao lưu ZIP",
                    backup_file.read(),
                    file_name=Path(backup_path).name,
                    mime="application/zip",
                )
        except OSError as exc:
            st.error(f"Không đọc được file backup {backup_path}: {exc}")

    st.markdown("#### Khôi phục từ file ZIP")
    st.warning(
        "Khôi phục sẽ thay database hiện tại và ghi đè file trùng tên trong evidence/uploads. "
        "File khác đang có sẽ được giữ lại. Tạo, tải backup hiện tại trước khi tiếp tục."
    )
    restore_file = st.file_uploader(
        "Chọn bản sao lưu của ứng dụng",
        type=["zip"],
        key="restore_backup_upload",
    )
    confirmed_restore = st.checkbox(
        "Tôi hiểu rằng thao tác này sẽ thay dữ liệu hiện tại",
        key="confirm_backup_restore",
    )
    if st.button(
        "Khôi phục bản sao lưu",
        type="primary",
        key="restore_backup",
        disabled=restore_file is None or not confirmed_restore,
    ):
        try:
            restored_path = restore_backup_package(restore_file)
            st.session_state["backup_restore_message"] = (
                f"Đã khôi phục dữ liệu thành công từ bản sao lưu vào {restored_path}."
            )
            st.rerun()
        except (OSError, sqlite3.Error, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
            st.error(f"Khôi phục bản sao lưu thất bại: {exc}")

    st.divider()
    st.subheader("Nhật ký hệ thống")
    audit = list_audit_logs(200)
    if audit:
        st.dataframe(pd.DataFrame(audit), use_container_width=True, hide_index=True)
        if st.button("Xuất nhật ký CSV", key="export_audit"):
            audit_csv = pd.DataFrame(audit).to_csv(index=False).encode("utf-8-sig")
            log_audit("audit_exported", "audit_log", "", f"rows={len(audit)}")
            st.download_button(
                "Tải CSV nhật ký",
                data=audit_csv,
                file_name=f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_audit",
            )
    else:
        st.info("Chưa có thao tác nào trong nhật ký.")


def render_config() -> None:
    st.subheader("Cấu hình mô hình AI")
    st.markdown("""
    - Detector hiện tại là YOLOv8n đã fine-tune trên dữ liệu biển số Việt Nam; file khởi tạo COCO chỉ là pretrained checkpoint, không phải detector biển số.
    - Đây là checkpoint pilot chạy trên 5% tập train trong 3 epoch do giới hạn tốc độ CPU; cần huấn luyện thêm trên toàn bộ dữ liệu trước khi dùng như hệ thống production.
    - Kết quả kiểm tra pilot được lưu trong `runs\\train\\vietnamese_plates_cpu_pilot_3ep\\test_metrics.json`.
    - OCR chạy bằng EasyOCR trên CPU; trọng số OCR được tải khi khởi tạo lần đầu.
    """)
    model_path = MODEL_DIR / "plate_detector_yolov8n.pt"
    st.write(f"Đường dẫn mô hình hiện tại: {model_path}")
    detector = get_plate_detector()
    if detector.mode == "YOLO plate detector":
        st.success("Đã nạp mô hình YOLO từ file trong thư mục models.")
    else:
        st.warning("Chưa nạp mô hình detector. Ứng dụng dùng contour OpenCV; không tải mô hình YOLO generic để thay thế.")
        if detector.model_error:
            st.error(f"Lỗi nạp trọng số: {detector.model_error}")
    st.code(
        "models\\plate_detector_yolov8n.pt",
        language="text",
    )
    st.caption("Sau khi thay trọng số, khởi động lại ứng dụng để nạp checkpoint mới.")
    st.divider()
    st.subheader("Sức chứa khu vực")
    current_capacity = get_setting("parking_capacity")
    try:
        capacity_value = int(current_capacity) if current_capacity else 0
    except ValueError as exc:
        st.error(f"Giá trị sức chứa đã lưu không hợp lệ: {exc}")
        capacity_value = 0
    with st.form("parking_capacity_form"):
        new_capacity = st.number_input(
            "Sức chứa tối đa (0 = chưa cấu hình)",
            min_value=0,
            max_value=100000,
            value=capacity_value,
            step=1,
            help="Dashboard tính chỗ trống bằng sức chứa trừ số xe đang ở trong khu vực.",
        )
        save_capacity = st.form_submit_button("Lưu sức chứa")
    if save_capacity:
        try:
            set_setting("parking_capacity", str(int(new_capacity)))
            log_audit(
                "parking_capacity_updated",
                "setting",
                "parking_capacity",
                f"capacity={int(new_capacity)}",
            )
            st.success("Đã lưu sức chứa khu vực.")
            st.rerun()
        except (OSError, sqlite3.Error) as exc:
            st.error(f"Không lưu được sức chứa: {exc}")
    st.divider()
    render_backup_audit()


def main() -> None:
    st.set_page_config(
        page_title="Plate Control | Giám sát phương tiện",
        page_icon="LP",
        layout="wide",
        initial_sidebar_state="auto",
    )
    apply_app_styles()
    configure_database(str(st.secrets.get("DATABASE_URL", "")))
    configure_object_storage(st.secrets)
    require_app_password()
    ensure_dirs()
    init_db()
    if st.session_state.get("main_page") not in PAGE_DETAILS:
        st.session_state["main_page"] = "Giám sát trực tiếp"

    with st.sidebar:
        if st.button("Đăng xuất", use_container_width=True):
            st.session_state.pop("authenticated", None)
            st.rerun()
        st.markdown(
            """
            <div class="brand-card">
              <div class="brand-mark">VMS</div>
              <div class="brand-title">VMS</div>
              <div class="brand-subtitle">Hệ thống giám sát xe ra vào &amp; biển số</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Điều hướng",
            [
                "Giám sát trực tiếp",
                "Lịch sử ra vào",
                "Quản lý xe",
                "Báo cáo",
                "Cấu hình",
                "Huấn luyện model",
            ],
            format_func=lambda item: {
                "Giám sát trực tiếp": "▣  Giám sát trực tiếp",
                "Lịch sử ra vào": "◷  Lịch sử ra vào",
                "Quản lý xe": "▱  Quản lý xe",
                "Báo cáo": "▤  Báo cáo",
                "Cấu hình": "⚙  Cấu hình",
                "Huấn luyện model": "🧠  Huấn luyện model",
            }[item],
            key="main_page",
        )

    title, description = PAGE_DETAILS[page]
    heading_class = "page-heading"
    status_label = "TRUNG TÂM ĐIỀU HÀNH"
    if page == "Giám sát trực tiếp":
        title = "Hệ thống Quản lý Giám sát Xe ra vào & Biển số"
        description = "Giám sát trực tiếp"
        heading_class += " dashboard-heading"
        status_label = f"NHÂN VIÊN · ◷ {datetime.now():%H:%M:%S}"
    st.markdown(
        f"""
        <div class="{heading_class}">
          <div><h1>{title}</h1><p>{description}</p></div>
          <div class="live-pill">{status_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if page == "Giám sát trực tiếp":
        render_overview()
    elif page == "Quản lý xe":
        render_vehicle_management()
    elif page == "Lịch sử ra vào":
        render_history()
    elif page == "Báo cáo":
        render_reports()
    elif page == "Huấn luyện model":
        render_train_page()
    else:
        render_config()


if __name__ == "__main__":
    main()
