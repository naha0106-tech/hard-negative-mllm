"""여러 스크립트가 같이 쓰는 함수 모음.
프롬프트 양식은 여기 한 곳에만 있어야 난이도 측정·학습·평가가 똑같은 양식을 씁니다.
"""
import json
import os
import re

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LOGS = os.path.join(ROOT, "logs")
TRAINSETS = os.path.join(ROOT, "trainsets")
LETTERS = ["A", "B", "C", "D", "E"]
for _d in (DATA, LOGS, TRAINSETS):
    os.makedirs(_d, exist_ok=True)


def load_config():
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Logger:
    """화면에 출력하면서 동시에 logs/ 폴더의 파일에도 적습니다."""

    def __init__(self, name):
        os.makedirs(LOGS, exist_ok=True)
        self.path = os.path.join(LOGS, name)
        self.f = open(self.path, "w", encoding="utf-8")

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        print(msg)
        self.f.write(msg + "\n")
        self.f.flush()


def load_scienceqa(cfg, split):
    from datasets import load_dataset

    d = cfg["datasets"]["scienceqa"]
    if d.get("config"):
        return load_dataset(d["repo"], d["config"], split=d["splits"][split])
    return load_dataset(d["repo"], split=d["splits"][split])


def grade_num(grade):
    m = re.search(r"\d+", str(grade))
    return int(m.group()) if m else None


def build_prompt(question, hint, options):
    """계획서 3.1의 표준 프롬프트. hint가 비어 있으면 Context 줄을 통째로 뺍니다."""
    lines = [question.strip()]
    if hint and str(hint).strip():
        lines.append(f"Context: {str(hint).strip()}")
    lines.append("Options:")
    for letter, opt in zip(LETTERS, options):
        lines.append(f"{letter}. {opt}")
    lines.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(lines)


def largest_remainder(counts, total):
    """층별 개수 비율대로 total개를 나눌 때 각 층에 몇 개씩 줄지 (최대잔여법)."""
    n = sum(counts.values())
    raw = {k: total * v / n for k, v in counts.items()}
    alloc = {k: int(r) for k, r in raw.items()}
    rest = total - sum(alloc.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - alloc[k]), k))[:rest]:
        alloc[k] += 1
    return alloc
