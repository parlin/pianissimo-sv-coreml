# Test results

The raw files behind the numbers in the [main README](../README.md#results), measured 2026-11-16.
Audio is not included; it comes from [FLEURS](https://huggingface.co/datasets/google/fleurs)
(`sv_se`, test split, CC BY 4.0).

| File | What |
|---|---|
| `subset100.tsv` | The 100 test clips: every 7th row of FLEURS `sv_se/test.tsv` (18.9 minutes) |
| `long.tsv` | The 20 clips joined into the 4-minute file `long.wav`, in order |
| `wer.txt` | Output of `../conversion/wer.py` for all runs below |
| `mac/run_nemo.jsonl` | Original NeMo model, PyTorch on CPU, 100 clips (the reference) |
| `mac/run_coreml.jsonl` | This Core ML build on a MacBook M1, CPU + Neural Engine, 100 clips |
| `mac/run_long_nemo.jsonl` | Original NeMo model, 4-minute file |
| `mac/run_long_coreml.jsonl` | Core ML build on the Mac, 4-minute file |
| `mac/svbench_time.txt` | `/usr/bin/time -l` for the whole Mac benchmark run (peak memory 639 MB) |
| `iphone/run_coreml.jsonl` | Core ML build on an iPhone 15 Pro (iOS 26.6.2), 100 clips |
| `iphone/run_long_coreml.jsonl` | Core ML build on the iPhone, 4-minute file |
| `iphone/summary.json` | iPhone load times, speed, peak memory and thermal state |
| `iphone/log.txt` | iPhone benchmark log, with memory and thermal state as it ran |

Each `.jsonl` line is one clip: `{"file", "ref", "hyp"}`, where `ref` is the FLEURS transcript and
`hyp` the model's output.

Reproduce `wer.txt` from this folder:

```bash
python3 ../conversion/wer.py mac/run_nemo.jsonl mac/run_coreml.jsonl iphone/run_coreml.jsonl
python3 ../conversion/wer.py mac/run_long_nemo.jsonl mac/run_long_coreml.jsonl iphone/run_long_coreml.jsonl
```

The Mac speed figures in the main README (62x and 137x realtime) were read from the benchmark's
console output, which was not saved; only the timing file above was.
