from locust import HttpUser, task
from random import randint

MAX_ID = 100_000


class HelloWorldUser(HttpUser):
    @task
    def hello_world(self):
        self.client.get(f"/similar/{randint(0, MAX_ID)}")
