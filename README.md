# Pianissimo Swedish (svenska) for Core ML

Swedish speech-to-text that runs **on the iPhone and the Mac**, on Apple's Neural Engine. This is
[Klang's Pianissimo-sv](https://huggingface.co/KlangAI/pianissimo-sv) converted to Core ML, in the
folder layout that the [FluidAudio](https://github.com/FluidInference/FluidAudio) Swift package loads.

**This is an unofficial conversion. It is not made by, or affiliated with, Klang.** The model and its
training are Klang's work; this repository only converts and packages it. If Klang publishes an
official Core ML build, prefer that one.

The model files are not in git. They are attached to the [releases](../../releases).

## Results

Measured on 2026-11-16 against the original NeMo model, on 100 clips from the Swedish
[FLEURS](https://huggingface.co/datasets/google/fleurs) test set (every 7th clip, 18.9 minutes of read
speech) and on a 4-minute file made of 20 clips joined together.

### Accuracy (word error rate, lower is better)

| | Original NeMo model (PyTorch, CPU) | This Core ML build |
|---|---|---|
| 100 FLEURS clips | 6.02 % | 6.22 % |
| 4-minute file | 4.91 % | 6.47 % |
| Transcripts identical to the original (normalized) | - | 85 of 100 |

Klang reports 6.51 % on the full FLEURS test set, so the reference run is in line with that.

- **Short clips:** the conversion is close to lossless. It adds 0.2 points of WER, from the 8-bit
  encoder and Core ML numerics.
- **Long recordings:** about 1.5 points are lost at the seams between FluidAudio's 15-second windows.
  You see repeated words at a seam ("först först") and small inflection slips where the context from the
  previous window is missing. FluidAudio's seam options and splitting at pauses are the next things to try.

### Speed, memory and size

| | iPhone 15 Pro (iOS 26) | MacBook M1 (2020) |
|---|---|---|
| 100 short clips | 105x realtime (18.9 min of audio in 10.8 s) | 62x |
| 4-minute file | 207x realtime (1.2 s) | 137x |
| First load (one-time Core ML compile) | 16.7 s | 36 s |
| Later loads | 0.2 s | 0.6 s |
| Peak memory | 132 MB (app footprint) | 639 MB (whole process) |
| Thermal state | nominal throughout | - |

A one-hour meeting takes about 20-35 seconds to transcribe on an iPhone 15 Pro. The iPhone produced the
same transcripts as the Mac (99 of 100 identical, same WER).

| Part | Size |
|---|---|
| `Encoder.mlmodelc` (8-bit palettized) | 567 MB |
| `Decoder.mlmodelc` | 23 MB |
| `JointDecisionv3.mlmodelc` | 12 MB |
| `Preprocessor.mlmodelc` | 0.6 MB |
| `parakeet_vocab.json` (8192 tokens) | 140 KB |
| **Total** | **about 602 MB** (the original `.nemo` is 2.51 GB) |

### Not tested yet

- Real meeting audio: far-field microphones, people talking over each other, dialects. FLEURS is read
  speech.
- Older iPhones (A15/A16) and single recordings of 30-60 minutes.

## Download

Each release has two files with the same contents and a `SHA256SUMS` file:

- `pianissimo-sv-coreml-v1.zip`: for most uses.
- `pianissimo-sv-coreml-v1.aar`: an [Apple Archive](https://developer.apple.com/documentation/applearchive)
  (LZFSE), which an iOS app can unpack with the built-in AppleArchive framework, with no zip library.

Check the download: `shasum -a 256 -c SHA256SUMS`.

## Use with FluidAudio

Tested with FluidAudio at revision `16780f0`. The model is a Parakeet TDT v3 layout, so load it as `.v3`:

```swift
import CoreML
import FluidAudio

let config = MLModelConfiguration()
config.computeUnits = .cpuAndNeuralEngine

let models = try AsrModels.loadLocal(from: modelFolder, version: .v3, configuration: config)
let asr = AsrManager(config: ASRConfig(tdtConfig: TdtConfig(blankId: AsrModelVersion.v3.blankId)))
try await asr.loadModels(models)

var state = TdtDecoderState.make(decoderLayers: await asr.decoderLayerCount)
let result = try await asr.transcribe(audioFileURL, decoderState: &state, language: nil)
print(result.text)              // the transcript
print(result.tokenTimings ?? [])  // start, end and confidence per token
```

Audio is resampled to 16 kHz mono by FluidAudio. The first load on a device compiles the model for the
Neural Engine, which takes a while. Do it once, before the first transcription, and show a
"Preparing model" state.

## What was changed from the original

CC BY 4.0 asks for changes to be stated. Compared with `KlangAI/pianissimo-sv`:

1. **Converted from NeMo to Core ML** with FluidInference's
   [mobius](https://github.com/FluidInference/mobius) export scripts (revision `588b8e7`).
2. **Full attention for export.** Pianissimo uses local attention (`rel_pos_local_attn`, context
   256/256), which Core ML cannot convert. A 15-second window is about 188 encoder frames, under the
   256-frame context, so full attention (`rel_pos`) gives the same result inside a window.
3. **Encoder weights quantized to 8 bits** (k-means palettization), the same recipe as FluidAudio's own
   Parakeet v3 encoder and Klang's Danish Core ML build.
4. **Packaged for FluidAudio:** the parts are compiled to `.mlmodelc` and named `Preprocessor`,
   `Encoder`, `Decoder` and `JointDecisionv3`. `parakeet_vocab.json` is written from the model's own tokenizer.

The scripts are in [`conversion/`](conversion/), with the steps to reproduce the build.

## Limitations

These come from the model, not the conversion:
- **No speaker separation.** The output is one stream of text, without "who said what".
- **No language detection.** It expects Swedish.
- **Names, company names and special terms** can come out wrong ("plusidrot" for "plusgirot"). The
  original model makes the same mistakes, so they are not caused by the conversion.
- Punctuation and numbers may need manual correction (Klang's own caveat).

## Licence and credit

The model is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), like the original.
When you use it, credit Klang, for example:

"Swedish transcription uses Pianissimo-sv by Klang (https://huggingface.co/KlangAI/pianissimo-sv),
licensed under CC BY 4.0, converted to Core ML with 8-bit encoder weights. Pianissimo is based on
NVIDIA Parakeet TDT 0.6B v3 (CC BY 4.0)."

- **Model:** Klang, [KlangAI/pianissimo-sv](https://huggingface.co/KlangAI/pianissimo-sv), CC BY 4.0.
- **Base model:** NVIDIA, [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), CC BY 4.0.
- **Runtime:** [FluidAudio](https://github.com/FluidInference/FluidAudio), Apache-2.0.
- **Test data:** [FLEURS](https://huggingface.co/datasets/google/fleurs), CC BY 4.0.
- **The conversion scripts** in `conversion/` are MIT licensed (see [LICENSE](LICENSE)).
