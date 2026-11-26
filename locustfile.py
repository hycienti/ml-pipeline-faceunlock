"""Flood simulation for FaceUnlock, with Locust.

Each simulated visitor logs in once, then keeps sending the model work:
  * predict a registered face     (weight 4)  POST /api/predict
  * predict a stranger's face     (weight 2)  POST /api/predict
  * log in again with the right face (weight 1)  POST /api/login
  * load the status API           (weight 1)  GET  /api/uptime

Run the whole 1 / 2 / 3 container experiment with:   python scripts/flood_experiment.py
Or by hand against any running app:
    locust -f locustfile.py --headless -u 60 -r 10 -t 60s --host http://localhost:8080
"""
from pathlib import Path

import requests
from locust import HttpUser, between, events, task

PHOTOS = Path(__file__).parent / "locust" / "photos"
USERNAME = "loadtest_liza"
ENROLL = [(PHOTOS / f"liza_{i}.jpg").read_bytes() for i in (1, 2, 3)]
RIGHT_FACE = (PHOTOS / "liza_probe.jpg").read_bytes()  # a different photo of the registered person
STRANGER = (PHOTOS / "stranger.jpg").read_bytes()


@events.test_start.add_listener
def make_sure_the_test_user_exists(environment, **_kwargs):
    files = [("photos", (f"enroll{i}.jpg", data, "image/jpeg")) for i, data in enumerate(ENROLL)]
    response = requests.post(
        f"{environment.host}/api/register", data={"username": USERNAME}, files=files, timeout=120
    )
    print(f"load-test user '{USERNAME}': HTTP {response.status_code} (201 created, 409 already there)")


class Visitor(HttpUser):
    wait_time = between(0.2, 0.6)  # short pauses: enough users keep one container saturated

    def post_photo(self, url, photo, name, **form):
        files = {"photo": ("photo.jpg", photo, "image/jpeg")}
        with self.client.post(url, data=form, files=files, name=name, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:80]}")

    def on_start(self):
        self.post_photo("/api/login", RIGHT_FACE, "login (first, right face)", username=USERNAME)

    @task(4)
    def predict_registered_face(self):
        self.post_photo("/api/predict", RIGHT_FACE, "predict (registered face)")

    @task(2)
    def predict_stranger(self):
        self.post_photo("/api/predict", STRANGER, "predict (stranger)")

    @task(1)
    def login_again(self):
        self.post_photo("/api/login", RIGHT_FACE, "login (right face)", username=USERNAME)

    @task(1)
    def status(self):
        self.client.get("/api/uptime", name="status page API")
