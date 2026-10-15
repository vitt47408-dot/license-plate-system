"""Train and deploy a YOLO plate detector on CPU."""

from __future__ import annotations

import shutil
import zipfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Callable, Union


PathLike = Union[str, Path]
MAX_DATASET_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
MAX_DATASET_ARCHIVE_BYTES = 512 * 1024 * 1024


def prepare_dataset_from_zip(archive_bytes: bytes, destination: PathLike) -> Path:
    """Safely extract a YOLO dataset ZIP and normalize its data.yaml paths."""
    import io
    import yaml

    if len(archive_bytes) > MAX_DATASET_ARCHIVE_BYTES:
        raise ValueError("Dataset ZIP vượt quá giới hạn tải lên 512 MB.")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    seen_paths = set()
    total_size = 0
    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
        for info in archive.infolist():
            name = info.filename
            path = PurePosixPath(name)
            mode = info.external_attr >> 16
            if (
                not name
                or "\\" in name
                or path.is_absolute()
                or any(
                    part in {"", ".", ".."}
                    or any(character in part for character in '<>:"|?*')
                    or part.endswith((" ", "."))
                    for part in path.parts
                )
                or (mode & 0o170000) == 0o120000
            ):
                raise ValueError(f"Đường dẫn không an toàn trong dataset ZIP: {name}")
            normalized_name = path.as_posix()
            if normalized_name.casefold() in seen_paths:
                raise ValueError(f"Dataset ZIP có đường dẫn lặp: {name}")
            seen_paths.add(normalized_name.casefold())
            if info.flag_bits & 0x1:
                raise ValueError("Dataset ZIP được mã hóa; hãy tải lên ZIP không đặt mật khẩu.")
            if total_size + info.file_size > MAX_DATASET_UNCOMPRESSED_BYTES:
                raise ValueError("Dataset giải nén vượt quá giới hạn 2 GB.")
            target = destination.joinpath(*path.parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f"Đường dẫn vượt khỏi thư mục dataset: {name}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, target.open("xb") as output:
                actual_size = 0
                while chunk := source.read(1024 * 1024):
                    actual_size += len(chunk)
                    if total_size + actual_size > MAX_DATASET_UNCOMPRESSED_BYTES:
                        raise ValueError("Dataset giải nén vượt quá giới hạn 2 GB.")
                    output.write(chunk)
                if actual_size != info.file_size:
                    raise zipfile.BadZipFile(f"Kích thước file ZIP không khớp: {name}")
                total_size += actual_size

    yaml_files = sorted(destination.rglob("data.yaml"))
    if len(yaml_files) != 1:
        raise ValueError("Dataset ZIP phải chứa đúng một file data.yaml.")
    data_yaml = yaml_files[0]
    with data_yaml.open("r", encoding="utf-8") as source:
        dataset_config = yaml.safe_load(source)
    if not isinstance(dataset_config, dict):
        raise ValueError("Nội dung data.yaml phải là một object YAML.")
    if not dataset_config.get("train") or not dataset_config.get("val"):
        raise ValueError("data.yaml cần khai báo cả đường dẫn train và val.")

    data_root = data_yaml.parent.resolve()
    for split_name in ("train", "val"):
        split_values = dataset_config[split_name]
        split_values = split_values if isinstance(split_values, list) else [split_values]
        for split_value in split_values:
            split_path = Path(str(split_value))
            if split_path.is_absolute():
                raise ValueError(f"Đường dẫn {split_name} trong data.yaml phải là đường dẫn tương đối.")
            resolved_split = (data_root / split_path).resolve()
            if not resolved_split.is_relative_to(destination) or not resolved_split.is_dir():
                raise ValueError(
                    f"Không tìm thấy thư mục {split_name} trong dataset ZIP: {split_value}"
                )
            split_parts = split_path.parts
            if split_parts and split_parts[0] == "images":
                label_path = data_root / "labels" / Path(*split_parts[1:])
                if not label_path.is_dir():
                    raise ValueError(f"Không tìm thấy thư mục nhãn YOLO tương ứng: {label_path}")

    dataset_config["path"] = str(data_root)
    with data_yaml.open("w", encoding="utf-8") as output:
        yaml.safe_dump(dataset_config, output, sort_keys=False, allow_unicode=True)
    return data_yaml


def train_plate_detector(
    data_yaml: PathLike,
    epochs: int = 40,
    imgsz: int = 416,
    batch: int = 4,
    model_name: str = "yolov8n.pt",
    project: PathLike = "runs/plate_detect",
    name: str = "train",
    device: str = "cpu",
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Fine-tune YOLOv8 and return the best checkpoint path."""
    data_yaml = Path(data_yaml).resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Không tìm thấy data.yaml: {data_yaml}")
    if epochs < 1 or imgsz < 1 or batch < 1:
        raise ValueError("epochs, imgsz và batch phải là số nguyên dương.")

    from ultralytics import YOLO

    model = YOLO(str(model_name))
    if progress_callback is not None:
        model.add_callback(
            "on_train_epoch_end",
            lambda trainer: progress_callback(int(trainer.epoch) + 1, int(epochs)),
        )
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        exist_ok=True,
        patience=15,
        save=True,
        plots=True,
        verbose=True,
        workers=0,
        amp=False,
    )

    trainer = getattr(model, "trainer", None)
    if trainer is None:
        raise RuntimeError("YOLO kết thúc nhưng không trả về trainer để xác định best.pt.")
    best_pt = Path(trainer.save_dir) / "weights" / "best.pt"
    if not best_pt.is_file():
        raise FileNotFoundError(f"Không tìm thấy best.pt tại {best_pt}")
    return best_pt


def deploy_best_model(best_pt: PathLike, target: PathLike | None = None) -> Path:
    """Atomically install best.pt as the active plate detector."""
    best_pt = Path(best_pt)
    if not best_pt.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint cần triển khai: {best_pt}")
    if target is None:
        target = Path(__file__).resolve().parent / "models" / "plate_detector_yolov8n.pt"
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged_target = target.with_name(f".{target.name}.{uuid.uuid4().hex}.new")
    try:
        shutil.copy2(best_pt, staged_target)
        staged_target.replace(target)
    finally:
        staged_target.unlink(missing_ok=True)
    return target


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLO plate detector (CPU)")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--model", default="yolov8n.pt")
    args = parser.parse_args()

    print("Starting CPU training...")
    best = train_plate_detector(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_name=args.model,
    )
    print(f"Training finished: {best}")
    print(f"Installed model: {deploy_best_model(best)}")
