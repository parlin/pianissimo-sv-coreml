"""Compare transcription runs. Usage: python wer.py <run.jsonl> [<run2.jsonl> ...]

Each line: {"file", "ref", "hyp"}. Normalization follows Klang's card: lowercase, strip punctuation.
Also reports agreement between the first run and each later run (hyp vs hyp).
"""
import json
import re
import sys


def norm(s: str) -> list[str]:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return s.split()


def edits(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def load(p):
    return {d["file"]: d for d in map(json.loads, open(p, encoding="utf-8"))}


runs = [(p, load(p)) for p in sys.argv[1:]]
common = set.intersection(*(set(r) for _, r in runs))
print(f"{len(common)} files in common")
base_name, base = runs[0]
for name, run in runs:
    e = sum(edits(norm(run[f]["ref"]), norm(run[f]["hyp"])) for f in common)
    n = sum(len(norm(run[f]["ref"])) for f in common)
    line = f"{name}: WER vs reference text {100 * e / n:.2f}%"
    if run is not base:
        e2 = sum(edits(norm(base[f]["hyp"]), norm(run[f]["hyp"])) for f in common)
        n2 = sum(len(norm(base[f]["hyp"])) for f in common)
        same = sum(norm(base[f]["hyp"]) == norm(run[f]["hyp"]) for f in common)
        line += f" | diff vs {base_name}: {100 * e2 / n2:.2f}% ({same}/{len(common)} identical)"
    print(line)
