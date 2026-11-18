"""Reference transcription with the original Pianissimo NeMo model (PyTorch, CPU).

Usage: python nemo_ref.py <nemo_path> <subset.tsv> <fleurs_audio_dir> <out.jsonl>
subset.tsv: FLEURS rows (id, file, raw text, normalized text, ...).
"""
import json
import sys
import time
from pathlib import Path

import torch
import nemo.collections.asr as nemo_asr

nemo_path, subset, audio_dir, out = sys.argv[1:5]
rows = [l.rstrip("\n").split("\t") for l in open(subset, encoding="utf-8") if l.strip()]
files = [str(Path(audio_dir) / r[1]) for r in rows]

model = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location="cpu")
model.eval()

t0 = time.time()
with torch.no_grad():
    hyps = model.transcribe(files, batch_size=4, verbose=False)
elapsed = time.time() - t0

with open(out, "w", encoding="utf-8") as f:
    for r, h in zip(rows, hyps):
        text = h.text if hasattr(h, "text") else str(h)
        f.write(json.dumps({"file": r[1], "ref": r[2], "hyp": text}, ensure_ascii=False) + "\n")
print(f"done {len(files)} files in {elapsed:.1f}s")
