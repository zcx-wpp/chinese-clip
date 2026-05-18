import json
import logging
import time

from openai import OpenAI


class LLM:
    def __init__(self, api_key, base_url, model_id, max_retries=5, retry_delay=10):
        self.api_key = api_key
        self.base_url = base_url
        self.model_id = model_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def chat(self, txt_content, system_prompt=None, temperature=0.3):
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        response_content = None
        system_messages = [
            {
                "role": "system",
                "content": (
                    "You are Kimi, an AI assistant provided by Moonshot AI. "
                    "Provide helpful and accurate answers."
                ),
            }
        ]
        if system_prompt:
            system_messages.append({"role": "system", "content": system_prompt})

        for attempt in range(self.max_retries):
            try:
                completion = client.chat.completions.create(
                    model=self.model_id,
                    messages=system_messages + [{"role": "user", "content": txt_content}],
                    temperature=temperature,
                    max_tokens=3000,
                    response_format={"type": "json_object"},
                )
                response_content = json.loads(completion.choices[0].message.content)
                return response_content
            except Exception as exc:
                logging.warning(
                    "LLM request failed (attempt %s/%s): %s",
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logging.error("LLM request failed after maximum retries: %s", exc)
                    return None

        return response_content
