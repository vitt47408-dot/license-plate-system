import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from train_plate_detector import (
    deploy_best_model,
    prepare_dataset_from_zip,
    train_plate_detector,
)


def build_dataset_zip(entries):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return output.getvalue()


class PlateTrainingUtilityTests(unittest.TestCase):
    def test_prepares_yolo_dataset_and_resolves_yaml_path(self):
        dataset_zip = build_dataset_zip({
            "package/data.yaml": "path: /old/location\ntrain: images/train\nval: images/val\nnames:\n  0: plate\n",
            "package/images/train/sample.jpg": b"image",
            "package/images/val/sample.jpg": b"image",
            "package/labels/train/sample.txt": b"0 0.5 0.5 0.4 0.2\n",
            "package/labels/val/sample.txt": b"0 0.5 0.5 0.4 0.2\n",
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = prepare_dataset_from_zip(dataset_zip, Path(temp_dir) / "dataset")

            self.assertTrue(yaml_path.is_file())
            self.assertIn(str(yaml_path.parent), yaml_path.read_text(encoding="utf-8"))
            self.assertTrue((yaml_path.parent / "images" / "train").is_dir())

    def test_rejects_zip_traversal(self):
        malicious_zip = build_dataset_zip({"../outside.txt": b"not allowed"})
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "dataset"
            with self.assertRaisesRegex(ValueError, "không an toàn"):
                prepare_dataset_from_zip(malicious_zip, destination)
            self.assertFalse((Path(temp_dir) / "outside.txt").exists())

    def test_requires_existing_dataset_yaml_before_loading_model(self):
        with self.assertRaises(FileNotFoundError):
            train_plate_detector("missing-data.yaml")

    def test_deploy_copies_checkpoint_to_active_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "best.pt"
            target = Path(temp_dir) / "models" / "plate.pt"
            source.write_bytes(b"trained weights")

            result = deploy_best_model(source, target)

            self.assertEqual(result, target)
            self.assertEqual(target.read_bytes(), b"trained weights")


if __name__ == "__main__":
    unittest.main()
