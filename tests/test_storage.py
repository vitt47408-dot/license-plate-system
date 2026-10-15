import io
import unittest
from unittest.mock import patch

import storage


class ObjectStorageConfigurationTests(unittest.TestCase):
    def tearDown(self):
        storage.configure_object_storage({})

    def test_storage_is_disabled_without_secrets(self):
        storage.configure_object_storage({})

        self.assertFalse(storage.object_storage_enabled())

    def test_rejects_partial_credentials(self):
        with self.assertRaisesRegex(ValueError, "S3_BUCKET"):
            storage.configure_object_storage({"S3_BUCKET": "bucket"})

    def test_upload_and_read_use_s3_compatible_client(self):
        class FakeS3Client:
            def __init__(self):
                self.uploads = []

            def put_object(self, **kwargs):
                self.uploads.append(kwargs)

            def get_object(self, **kwargs):
                return {"Body": io.BytesIO(b"evidence")}

        client = FakeS3Client()
        secrets = {
            "S3_BUCKET": "plate-data",
            "S3_ACCESS_KEY_ID": "test-key",
            "S3_SECRET_ACCESS_KEY": "test-secret",
            "S3_ENDPOINT_URL": "https://objects.example.test",
            "S3_PREFIX": "app-data",
        }
        with patch("boto3.client", return_value=client):
            storage.configure_object_storage(secrets)

        uri = storage.put_bytes("evidence/capture.jpg", b"evidence", "image/jpeg")

        self.assertEqual(uri, "s3://plate-data/app-data/evidence/capture.jpg")
        self.assertEqual(storage.read_bytes(uri), b"evidence")
        self.assertEqual(client.uploads[0]["ContentType"], "image/jpeg")


if __name__ == "__main__":
    unittest.main()
