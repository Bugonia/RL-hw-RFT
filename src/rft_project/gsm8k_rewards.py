from __future__ import annotations

import re


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[-1].strip()


def correctness_reward(responses: list[str], answers: list[str]) -> list[float]:
    extracted = [extract_xml_answer(response) for response in responses]
    return [2.0 if response == answer else 0.0 for response, answer in zip(extracted, answers)]


def integer_reward(responses: list[str]) -> list[float]:
    extracted = [extract_xml_answer(response) for response in responses]
    return [0.5 if response.isdigit() else 0.0 for response in extracted]


def strict_format_reward(responses: list[str]) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    return [0.5 if re.match(pattern, response, flags=re.DOTALL) else 0.0 for response in responses]


def soft_format_reward(responses: list[str]) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    return [0.5 if re.search(pattern, response, flags=re.DOTALL) else 0.0 for response in responses]


def xml_count_score(text: str) -> float:
    score = 0.0
    if text.count("<reasoning>\n") == 1:
        score += 0.125
    if text.count("\n</reasoning>\n") == 1:
        score += 0.125
    if text.count("\n<answer>\n") == 1:
        score += 0.125
        score -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        score += 0.125
        score -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return score


def xml_count_reward(responses: list[str]) -> list[float]:
    return [xml_count_score(response) for response in responses]


def length_penalty_reward(
    responses: list[str],
    target_chars: int = 900,
    penalty_scale: float = 0.0005,
    max_penalty: float = 0.3,
) -> list[float]:
    penalties = []
    for response in responses:
        overflow = max(0, len(response) - target_chars)
        penalties.append(-min(max_penalty, overflow * penalty_scale))
    return penalties


def total_baseline_reward(responses: list[str], answers: list[str]) -> list[float]:
    components = [
        xml_count_reward(responses),
        soft_format_reward(responses),
        strict_format_reward(responses),
        integer_reward(responses),
        correctness_reward(responses, answers),
    ]
    return [sum(values) for values in zip(*components)]


def total_length_aware_reward(responses: list[str], answers: list[str]) -> list[float]:
    components = [
        xml_count_reward(responses),
        soft_format_reward(responses),
        strict_format_reward(responses),
        integer_reward(responses),
        correctness_reward(responses, answers),
        length_penalty_reward(responses),
    ]
    return [sum(values) for values in zip(*components)]

