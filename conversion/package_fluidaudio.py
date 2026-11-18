"""Turn the mobius fp32 export into a FluidAudio v3 local model folder.

Usage: python package_fluidaudio.py <nemo_path> <export_dir> <out_dir> [--skip-palettize]
Produces: Preprocessor.mlmodelc, Encoder.mlmodelc (8-bit kmeans palettized), Decoder.mlmodelc,
JointDecisionv3.mlmodelc, parakeet_vocab.json
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import coremltools as ct
from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig, palettize_weights

nemo_path, export_dir, out_dir = map(Path, sys.argv[1:4])
skip_pal = "--skip-palettize" in sys.argv
out_dir.mkdir(parents=True, exist_ok=True)
pkg_dir = out_dir.parent / (out_dir.name + "_mlpackage")
pkg_dir.mkdir(parents=True, exist_ok=True)

# 1. Encoder: 8-bit kmeans palettization (same recipe as FluidAudio's v3 Encoder and Klang's Danish build)
enc_out = pkg_dir / "Encoder.mlpackage"
if not enc_out.exists():
    if skip_pal:
        shutil.copytree(export_dir / "parakeet_encoder.mlpackage", enc_out)
    else:
        t0 = time.time()
        model = ct.models.MLModel(str(export_dir / "parakeet_encoder.mlpackage"), skip_model_load=True)
        cfg = OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=8))
        pal = palettize_weights(model, cfg)
        pal.save(str(enc_out))
        print(f"palettized encoder in {time.time() - t0:.0f}s")

# 2. Other components copied as-is (fp32 export)
for src, dst in [
    ("parakeet_preprocessor.mlpackage", "Preprocessor.mlpackage"),
    ("parakeet_decoder.mlpackage", "Decoder.mlpackage"),
    ("parakeet_joint_decision_single_step.mlpackage", "JointDecisionv3.mlpackage"),
]:
    if not (pkg_dir / dst).exists():
        shutil.copytree(export_dir / src, pkg_dir / dst)

# 3. Compile to .mlmodelc
for pkg in sorted(pkg_dir.glob("*.mlpackage")):
    target = out_dir / (pkg.stem + ".mlmodelc")
    if target.exists():
        shutil.rmtree(target)
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), str(out_dir)], check=True,
                   stdout=subprocess.DEVNULL)
    print("compiled", target.name)

# 4. Vocabulary from the model's own tokenizer
vocab_path = out_dir / "parakeet_vocab.json"
if not vocab_path.exists():
    from nemo.collections.asr.models import ASRModel
    m = ASRModel.restore_from(str(nemo_path), map_location="cpu")
    tok = m.tokenizer
    vocab = {str(i): tok.ids_to_tokens([i])[0] for i in range(tok.vocab_size)}
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    print(f"vocab: {len(vocab)} tokens")

for p in sorted(out_dir.iterdir()):
    size = subprocess.run(["du", "-sh", str(p)], capture_output=True, text=True).stdout.split()[0]
    print(size, p.name)
