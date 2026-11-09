# GPU build (OpenCL, Adreno)

An opt-in build in which the Whisper encoder runs on the phone's GPU and the
decoder on the CPU. It is built with `gpu/build-gpu-apk.sh` (see the comment
at its top for what it patches) and is not part of the regular or F-Droid
builds, which stay CPU-only.

On the phone it was developed on (Nothing Phone, Snapdragon 8s Gen 4,
Adreno 825), a 15 s run takes about 4 s instead of 13 s with the same 8-bit
models, and 37 of 40 FLEURS test transcripts (en, de, sk, he) matched the CPU
build's, at the same error rate. That is the only device it has been run on.

## Which phones it runs on

| Phone | What happens |
|---|---|
| Qualcomm Snapdragon with an Adreno GPU, arm64, Android 8 or newer | Runs on the GPU. Tested only on the Adreno 825. |
| Other Android phone that lets apps load `libOpenCL.so` (some Exynos and MediaTek phones with Mali GPUs) | Expected to run on the CPU, like the regular build: the OpenCL backend in this build only accepts Adreno GPUs. Not tested. |
| Phone that doesn't let apps load `libOpenCL.so` (e.g. Google Pixel) | **Nothing works, not even recording**: the native library links `libOpenCL.so` directly, and Android refuses to load the app's library when it can't provide that one. Use the regular build. |
| 32-bit or x86 phone | Not supported (arm64-v8a only, like the regular build). |

On Adreno GPUs, the backend needs OpenCL 2.0 or newer with FP16 and subgroup
support (the Adreno 825 reports OpenCL 3.0 with both). It recognises the 6xx, 7xx
(730/740/750), 8xx (810/825/830/840/850) and X1/X2 generations; other Adreno
GPUs (e.g. 720, 732, 735) run on its generic kernels. Generic kernels are
slower, and on an untested GPU the results could also be wrong (the Hexagon
NPU on this chip first produced garbage until it was patched). Check the first
transcriptions on a new device.

What to expect on a supported phone:

- **First start:** compiling the GPU kernels takes tens of seconds. They are
  cached in the app's cache directory, so this happens again only after
  clearing the app's cache or reinstalling.
- **Model loading** takes 3–7 s instead of about 1 s. With automatic
  language, the two most recently used language models stay loaded, so only
  switching to a third language reloads.
- **Memory:** each language model is 874 MB. With two loaded, plus the
  language-ID model and the GPU's working buffers, expect roughly 2 GB (not
  measured).
- **Heat:** during each transcription the GPU sensors reached 73–91 °C and the
  CPU sensors 80–90 °C. Over five minutes of normal dictation the phone's
  surface stayed at 36–40 °C.

## The APK

`build-gpu-apk.sh` builds the debug flavor: package
`dev.notune.transcribe.dev`, app name "Voice Input (dev)", from the F-Droid
flavor, so without a built-in model. It installs alongside the regular app,
not over it. It is signed with the debug key of whoever built it, so an
update from someone else's build means uninstalling first, which deletes the
app's settings and may delete its model folder; keep a copy of the models.

## Models

The app doesn't download models. Speech models are placed in

    Android/media/dev.notune.transcribe.dev/models/

on the phone's shared storage (create `models/` if it isn't there). It can be
reached over USB and with most file managers. Use exactly these file names:

| File | Used for | Size | SHA-256 |
|---|---|---|---|
| `whisper-large-v3-turbo-q8_0.bin` | English, German | 874 MB | `317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1` |
| `whisper-large-v3-turbo-sk-q8_0.bin` | Slovak | 874 MB | `76e808163d58c1dbd6b1418fd78b066ce0f49cdad9b8eb5734ff4dd4bb4ccb62` |
| `whisper-large-v3-turbo-he-q8_0.bin` | Hebrew | 874 MB | `123a936e686b06d45b52dc1790251a1418841352ca94079ae643d57893ffc9a6` |
| `lang-id-voxlingua107-ecapa-Q8_0.gguf` | Automatic language choice | 27 MB | `063e6cb0af3d53085fc7d68356dfde5ecee2ffa7bc79855f7e4ff930081bfe3a` |

Where they come from:

- **English/German:** stock Whisper large-v3-turbo, 8-bit, downloadable as
  [`ggml-large-v3-turbo-q8_0.bin`](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q8_0.bin)
  (MIT). Rename it.
- **Language ID:** downloadable as
  [`lang-id-voxlingua107-ecapa-Q8_0.gguf`](https://huggingface.co/handy-computer/lang-id-voxlingua107-ecapa-gguf/resolve/main/lang-id-voxlingua107-ecapa-Q8_0.gguf)
  (Apache-2.0).
- **Slovak:** no download exists. It was converted from
  [NaiveNeuron/whisper-large-v3-turbo-sk](https://huggingface.co/NaiveNeuron/whisper-large-v3-turbo-sk)
  (MIT) with whisper.cpp's `models/convert-h5-to-ggml.py`, then
  `quantize … q8_0`.
- **Hebrew:** no download exists. It was quantized from
  [ivrit-ai/whisper-large-v3-turbo-ggml](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ggml)
  (`ggml-model.bin`, Apache-2.0) with whisper.cpp's `quantize … q8_0`.

A converted file need not match the checksum above byte for byte, since it
depends on the converter's version. All four licenses allow redistributing
the converted files.

Which files are needed depends on the language setting:

- **Automatic:** the language-ID model plus `whisper-large-v3-turbo-q8_0.bin`
  (the starting language is English). Each run then goes to the model of
  whichever of en/de/sk/he it is detected as; a missing file for a detected
  language is an error.
- **A fixed language among en/de/sk/he:** only that language's file. If it is
  missing, the app shows an error rather than using another model.
- **Without these files:** the app uses a model imported under "Manage speech
  models", as in the regular build. An imported model also loads on the GPU;
  for models other than Whisper (e.g. Parakeet) that is untested.
