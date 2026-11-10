# NPU build (Qualcomm Hexagon, QNN)

An opt-in build in which Whisper runs entirely on the phone's NPU: both the
encoder and the decoder, through ONNX Runtime's QNN execution provider. It is
built with `npu/build-npu-apk.sh` and is not part of the regular or F-Droid
builds: the QNN libraries are Qualcomm's and closed source.

On the phone it was developed on (Nothing Phone, Snapdragon 8s Gen 4, Hexagon
v73), a 15 s run takes about 1.1 s, against about 4 s in the GPU build and
13 s on the CPU. On FLEURS (100 clips per language; English: 650 British
clips from OpenSLR 83) the error rates are the CPU build's within noise:

| | NPU | CPU, Q8_0 (with the repetition guard) |
|---|---|---|
| English (en-GB) | 2.82% | 2.75% |
| German | 5.05% | 5.01% |
| Slovak | 6.90% | 7.29% |
| Hebrew | 22.30% | 21.38% |

## Which phones it runs on

Only Snapdragon phones with a Hexagon v73 NPU have been considered, and only
the 8s Gen 4 tested. The models are precompiled for one NPU generation
(here: compiled for the Snapdragon 7 Gen 4, which shares v73 with the 8s
Gen 4), and the APK ships only the v73 NPU library. Elsewhere an NPU model
fails to load with an error naming the model; the GGUF models keep working
on the CPU.

## Models

The NPU models sit next to the GGUF models in

    Android/media/dev.notune.transcribe.dev/models/

| Directory | Used for | Contents |
|---|---|---|
| `qnn-en/` | English, German | stock Whisper large-v3-turbo, from Qualcomm AI Hub |
| `qnn-sk/` | Slovak | NaiveNeuron/whisper-large-v3-turbo-sk |
| `qnn-he/` | Hebrew | ivrit-ai/whisper-large-v3-turbo |
| `qnn-shared/` | all | `tokens.bin` (vocabulary), `mel_filters.bin` |

Each language directory holds `encoder-ep.onnx`, `encoder.bin`,
`decoder-ep.onnx`, `decoder.bin` (about 1 GB together) and `qnn.cfg` (the
encoder input's quantization). A language whose directory is present uses
the NPU; otherwise its GGUF file is used as before. Language detection
(automatic language) still runs on the CPU with the VoxLingua107 model.

How they were made:

- **English/German:** Qualcomm's
  [Whisper-Large-V3-Turbo-Quantized](https://huggingface.co/qualcomm/Whisper-Large-V3-Turbo-Quantized),
  `qnn_context_binary`, w8a16, for the Snapdragon 7 Gen 4 (Apache-2.0). The
  `.onnx` files wrap Qualcomm's `.bin` context binaries in an EPContext node.
- **Slovak, Hebrew:** the fine-tunes, calibrated with AIMET (int8 weights,
  int16 activations, 8-bit KV caches, 40 FLEURS clips) and compiled and
  linked on Qualcomm AI Hub (QAIRT 2.50, Snapdragon 7 Gen 4 QRD).
- **`tokens.bin`, `mel_filters.bin`:** from the large-v3-turbo tokenizer and
  feature extractor; the fine-tunes share both.

## Limits

- Greedy decoding, without timestamps, at most 196 tokens per run (the
  app's runs are at most 15 s, well within that).
- The translate setting has no effect (Whisper Turbo doesn't translate).
- Each run's audio is padded to a 30 s window, as in the CPU build.
