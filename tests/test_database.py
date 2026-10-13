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


if __name__ == "__main__":
    unittest.main()
