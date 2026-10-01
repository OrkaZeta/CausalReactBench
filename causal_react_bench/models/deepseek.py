import json
import os
import time

from openai import OpenAI


class DeepSeekReasoner:
    def __init__(self, model: str = "deepseek-v4-pro", api_key=os.environ["DS_TOKEN_CP"], max_retries: int = 3):
        self.model = model
        self.max_retries = max_retries

        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def run(self, system_prompt: str, user_prompt: str) -> dict:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    reasoning_effort="high",
                    extra_body={"thinking": {"type": "enabled", }},
                    response_format={"type": "json_object", },
                    max_tokens=2048,
                )

                content = response.choices[0].message.content

                if not content: raise ValueError("Empty DeepSeek response")

                return json.loads(content)

            except Exception as e:
                last_error = e
                time.sleep(2 ** attempt)

        raise RuntimeError(f"DeepSeek failed after {self.max_retries} attempts") from last_error
