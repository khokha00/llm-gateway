import random

from locust import HttpUser, between, task

PROMPTS = [
    "Explain the CAP theorem in two sentences.",
    "Write a haiku about distributed systems.",
    "What's the time complexity of quicksort on average?",
    "Summarize the plot of Frankenstein in one paragraph.",
    "Give me three good names for a coffee shop.",
    "What causes tides?",
    "Translate 'good morning' into French, Spanish, and Japanese.",
    "Explain the difference between TCP and UDP.",
]


class GatewayUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(9)
    def chat_completion_varied(self):
        """Realistic traffic: distinct prompts so cache stays cold."""
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                "messages": [{"role": "user", "content": random.choice(PROMPTS)}],
                "temperature": 0.7,
                "max_tokens": 200,
            },
            headers={"Authorization": "Bearer load-test-key"},
        )

    @task(1)
    def chat_completion_cache_probe(self):
        """Identical, deterministic request -- isolates cache-hit latency."""
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "temperature": 0,
            },
            headers={"Authorization": "Bearer load-test-key"},
        )

# Run:
#   locust -f locustfile.py --host http://<gateway-host>:8000 \
#       --users 100 --spawn-rate 5 --run-time 5m --csv results/run1 --headless
