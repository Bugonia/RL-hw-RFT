from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from rft_project.gsm8k_rewards import (  # noqa: E402
    extract_hash_answer,
    extract_xml_answer,
    total_baseline_reward,
    total_length_aware_reward,
)


def main() -> None:
    good = """<reasoning>
2 + 2 = 4
</reasoning>
<answer>
4
</answer>
"""
    bad = """<reasoning>
2 + 2 = 5
</reasoning>
<answer>
5
</answer>
"""
    verbose = good + ("extra text " * 200)

    assert extract_xml_answer(good) == "4"
    assert extract_hash_answer("reasoning #### 4") == "4"

    baseline = total_baseline_reward([good, bad], ["4", "4"])
    length_aware = total_length_aware_reward([verbose], ["4"])

    assert baseline[0] > baseline[1]
    assert length_aware[0] < total_baseline_reward([verbose], ["4"])[0]

    print("Reward smoke test passed.")
    print({"good_reward": baseline[0], "bad_reward": baseline[1], "verbose_reward": length_aware[0]})


if __name__ == "__main__":
    main()

