import math

from causal_react_bench.prompts.configs import SYS_SHORT, SYS_TRIM, SYS_SPLIT


def select_system_prompt(seconds: float) -> tuple[str, float]:
    if seconds <= 5.0:
        return SYS_SHORT, seconds
    if seconds <= 7.0:
        return SYS_TRIM, 5.0
    return SYS_SPLIT, 5.0


def reason_user_prompt(prompt: str, seconds: float) -> str:
    return f"""SOURCE_SECONDS: {seconds:.3f}

SOURCE_PROMPT:
{prompt}
"""


class PromptBuilder:
    def __init__(self):
        from causal_react_bench.models.deepseek import DeepSeekReasoner

        self.reasoner = DeepSeekReasoner()

    def process(self, row: dict) -> list[dict]:
        source_prompt = row["prompt"].strip()
        source_seconds = float(row["seconds"])

        if not source_prompt or not math.isfinite(source_seconds) or source_seconds <= 0:
            raise ValueError("Source prompt must be nonempty and duration finite and positive")

        system_prompt, target_seconds = select_system_prompt(source_seconds)

        result = self.reasoner.run(
            system_prompt,
            reason_user_prompt(source_prompt, source_seconds),
        )

        if not isinstance(result, dict) or type(result.get("valid")) is not bool:
            raise ValueError("Reasoning response must contain a boolean valid field")
        if not result["valid"]:
            return []

        candidates = result.get("prompts")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("Valid reasoning response must contain candidate prompts")
        if source_seconds <= 7.0 and len(candidates) != 1:
            raise ValueError("Sources at most 7 seconds must produce exactly one prompt")

        outputs = []

        for item in candidates:
            if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
                raise ValueError("Candidate prompt must be a string")
            prompt = item["prompt"].strip()

            if not prompt:
                continue

            outputs.append({
                "prompt": prompt,
                "duration": target_seconds,
                "src_set": row["src_set"],
                "src_vid": row["src_vid"],

                # Only for independent review; removed from final output.
                "source_prompt": source_prompt,
                "source_duration": source_seconds,
            })

        return outputs
