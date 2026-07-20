"""
DeepGuard — tests/load/locustfile.py

Locust load testing script to simulate concurrent user operations on the DeepGuard API.
Saves staging overhead by creating an in-memory dummy image file to upload.
"""

import io
import random
from locust import HttpUser, task, between
from PIL import Image


class DeepGuardUser(HttpUser):
    """Simulates active developer client calls against backend endpoints."""
    wait_time = between(1, 3)

    def on_start(self) -> None:
        """Create in-memory dummy JPEG image bytes to use in uploads."""
        img = Image.new("RGB", (224, 224), color=(73, 109, 137))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        self.image_data = img_bytes.getvalue()

    @task(3)
    def check_health(self) -> None:
        """Query basic system observability health status."""
        self.client.get("/api/v1/health")

    @task(1)
    def upload_image_detection(self) -> None:
        """Submit dummy image upload predictions checks."""
        files = {
            "file": ("load_test_crop.jpg", self.image_data, "image/jpeg")
        }
        self.client.post("/api/v1/detect", files=files)

    @task(2)
    def view_history_log(self) -> None:
        """Request prediction log paginations."""
        self.client.get("/api/v1/history?page=1&page_size=10")

    @task(2)
    def query_stats(self) -> None:
        """Request model telemetry performance aggregates."""
        self.client.get("/api/v1/history/stats")
