import json
import re

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor


class QwenReviewer:
    def __init__(self, model_id: str = "Qwen/Qwen3.8-27B-FP8"):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto", dtype="auto")
        self.model.eval()

    @torch.inference_mode()
    def run(self, system_prompt: str, user_prompt: str) -> dict:
        messages = [{
            "role": "system",
            "content": system_prompt,
        }, {
            "role": "user",
            "content": user_prompt,
        }]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=True,
        ).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
        )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]

        text = self.processor.decode(
            generated,
            skip_special_tokens=True,
        )

        return self._parse_json(text)

    @staticmethod
    def _parse_json(text: str) -> dict:
        # Qwen thinking mode may emit <think>...</think>
        if "</think>" in text:
            text = text.split("</think>", 1)[-1]

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # fallback: extract final JSON object
        match = re.search(r"\{.*\}", text, flags=re.S)

        if not match:
            raise ValueError(f"No JSON found in Qwen output:\n{text}")

        return json.loads(match.group(0))
