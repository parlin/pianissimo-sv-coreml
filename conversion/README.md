# Reproducing the conversion

About 30 minutes on an M1 Mac.

| File | What |
|---|---|
| `mobius-convert-parakeet.patch` | Patch for `FluidInference/mobius` @ `588b8e7`: full attention for export, `no_grad` instead of `inference_mode` |
| `package_fluidaudio.py` | mobius fp32 export -> FluidAudio v3 folder (8-bit palettized encoder, `.mlmodelc`, `parakeet_vocab.json`) |
| `nemo_ref.py` | Reference transcripts with the original NeMo model (PyTorch, CPU) |
| `wer.py` | WER and run-to-run agreement for one or more `.jsonl` runs |

## You need

- `pianissimo-sv.nemo`. Klang's [Pianissimo CLI](https://www.npmjs.com/package/@klangai/pianissimo-cli)
  downloads it with `pianissimo setup` to `~/Library/Application Support/pianissimo/models/`.
- A Python venv with torch and NeMo. The Pianissimo CLI's own venv works (Python 3.12, torch 2.11, NeMo 3.0).
- coremltools 9.0, installed in a side folder so the venv is left untouched.
- Xcode (for `xcrun coremlcompiler`).

## Steps

```bash
W=work; mkdir -p $W
PY="$HOME/Library/Application Support/pianissimo/venv/bin/python"
NEMO=$(find "$HOME/Library/Application Support/pianissimo/models" -name pianissimo-sv.nemo | head -1)

# 1. coremltools 9.0 next to the venv, plus a numpy 2.x fix in _cast
"$PY" -m pip install --target $W/pylib coremltools==9.0 typer cattrs
sed -i '' 's/dtype(x\.val)/dtype(np.asarray(x.val).reshape(-1)[0])/' \
  $W/pylib/coremltools/converters/mil/frontend/torch/ops.py   # check that only the _cast line changed

# 2. Export with mobius
git clone https://github.com/FluidInference/mobius.git $W/mobius
git -C $W/mobius checkout 588b8e7
git -C $W/mobius apply "$PWD/mobius-convert-parakeet.patch"
(cd $W/mobius/models/stt/parakeet-tdt-v3-0.6b/coreml && \
  PYTHONPATH=$OLDPWD/$W/pylib "$PY" convert-parakeet.py --nemo-path "$NEMO" --output-dir $OLDPWD/$W/sv_coreml_fp32)

# 3. Package for FluidAudio
PYTHONPATH=$W/pylib "$PY" package_fluidaudio.py "$NEMO" $W/sv_coreml_fp32 $W/pianissimo-sv-coreml
```

Do not run Python with `-I`, because that ignores `PYTHONPATH`.

Why the patches:
- **Full attention:** Pianissimo is configured with local attention (`rel_pos_local_attn`), which NeMo
  implements with `as_strided`, which Core ML cannot convert. Inside a 15-second window full attention
  gives the same result.
- **`torch.no_grad()`:** NeMo 3.0 caches an attention mask created in inference mode, and tracing then fails.
- **coremltools `_cast`:** numpy 2.x no longer turns a 1-element array into a scalar.

## Evaluate

```bash
# FLEURS sv_se test: test.tsv and the audio from google/fleurs. The subset is every 7th row (100 clips).
PYTHONPATH=$W/pylib "$PY" nemo_ref.py "$NEMO" subset100.tsv fleurs/audio/test run_nemo.jsonl
# Transcribe the same clips with the Core ML build (via FluidAudio) into run_coreml.jsonl, one
# {"file", "ref", "hyp"} object per line, then:
python3 wer.py run_nemo.jsonl run_coreml.jsonl
```

Expected: about 6.0 % WER for NeMo and about 6.2 % for Core ML, with 85 of 100 transcripts identical.
