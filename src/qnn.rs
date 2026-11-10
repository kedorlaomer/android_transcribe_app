//! Whisper on the Qualcomm NPU (npu-qnn build only, `--features qnn`).
//!
//! A per-language model directory `qnn-<name>/` next to the GGUF models holds
//! a Whisper large-v3-turbo encoder/decoder pair precompiled for the Hexagon
//! NPU (see npu/README.md). The NPU runs both halves; this module computes
//! Whisper's log-mel spectrogram, drives the C++ wrapper in
//! qnn/qnn_whisper.cpp (ONNX Runtime with its QNN execution provider) and
//! turns the output tokens into text.

use std::ffi::{c_char, c_int, CString};
use std::path::Path;

const SAMPLE_RATE: usize = 16_000;
const N_FFT: usize = 400;
const HOP: usize = 160;
const MEL_BINS: usize = 128;
const FRAMES: usize = 3000;
/// One encoder window: 30 s.
const WINDOW_SAMPLES: usize = SAMPLE_RATE * 30;
/// Language tokens: `<|en|>` is 50259; the order is Whisper's.
const FIRST_LANG_TOKEN: i32 = 50259;
const LANGUAGES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi",
    "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la",
    "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy",
    "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
    "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
    "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue",
];
/// Most text tokens one window can produce (the decoder's self cache holds
/// 199 positions, 4 of them for the prompt).
const MAX_TOKENS: usize = 196;

#[repr(C)]
struct QwModel {
    _private: [u8; 0],
}

extern "C" {
    fn qw_open(dir: *const c_char, err: *mut c_char, err_len: usize) -> *mut QwModel;
    fn qw_close(model: *mut QwModel);
    fn qw_run(
        model: *mut QwModel,
        mel: *const f32,
        lang: c_int,
        translate: c_int,
        out: *mut c_int,
        max_out: c_int,
        detected: *mut c_int,
        enc_ms: *mut f64,
        dec_ms: *mut f64,
        err: *mut c_char,
        err_len: usize,
    ) -> c_int;
}

fn err_string(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

/// Tells the NPU driver where the NPU-side library (libQnnHtpV73Skel.so,
/// shipped with the APK's native libraries) is. Call before loading a model.
pub fn configure_env(native_lib_dir: &Path) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let path = format!("{};/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp", native_lib_dir.display());
        std::env::set_var("ADSP_LIBRARY_PATH", &path);
        log::info!("ADSP_LIBRARY_PATH={}", path);
    });
}

/// Whisper's byte-level BPE vocabulary, as raw bytes per token id.
struct Tokens(Vec<Vec<u8>>);

impl Tokens {
    /// Reads `tokens.bin`: u32 count, then per id a u16 length and the bytes.
    fn load(path: &Path) -> Result<Tokens, String> {
        let data = std::fs::read(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        let bad = || format!("{}: truncated", path.display());
        let count = u32::from_le_bytes(data.get(0..4).ok_or_else(bad)?.try_into().unwrap()) as usize;
        let mut tokens = Vec::with_capacity(count);
        let mut at = 4;
        for _ in 0..count {
            let len = u16::from_le_bytes(data.get(at..at + 2).ok_or_else(bad)?.try_into().unwrap()) as usize;
            at += 2;
            tokens.push(data.get(at..at + len).ok_or_else(bad)?.to_vec());
            at += len;
        }
        Ok(Tokens(tokens))
    }

    fn text(&self, ids: &[i32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .filter_map(|&id| self.0.get(id as usize))
            .flatten()
            .copied()
            .collect();
        String::from_utf8_lossy(&bytes).trim().to_string()
    }
}

/// Whisper's log-mel front end (as in OpenAI's and Hugging Face's code):
/// reflect-padded 400-point Hann-windowed STFT every 160 samples, power
/// spectrum through the 128 slaney mel filters, log10, clamped to 8 below
/// the maximum and scaled to (x + 4) / 4.
struct MelFrontEnd {
    /// 128 x 201, row per mel bin (`mel_filters.bin`).
    filters: Vec<f32>,
    window: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl MelFrontEnd {
    fn load(path: &Path) -> Result<MelFrontEnd, String> {
        let data = std::fs::read(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        let bins = N_FFT / 2 + 1;
        if data.len() != MEL_BINS * bins * 4 {
            return Err(format!("{}: unexpected size {}", path.display(), data.len()));
        }
        let filters = data.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        // Periodic Hann window, as torch.hann_window / window_function("hann").
        let window = (0..N_FFT)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos())
            .map(|w| w as f32)
            .collect();
        let (mut cos, mut sin) = (Vec::with_capacity(N_FFT * bins), Vec::with_capacity(N_FFT * bins));
        for k in 0..bins {
            for n in 0..N_FFT {
                let a = 2.0 * std::f64::consts::PI * (k * n % N_FFT) as f64 / N_FFT as f64;
                cos.push(a.cos() as f32);
                sin.push(a.sin() as f32);
            }
        }
        Ok(MelFrontEnd { filters, window, cos, sin })
    }

    /// Features for up to 30 s of audio, zero-padded to 30 s: 128 x 3000.
    fn features(&self, samples: &[f32]) -> Vec<f32> {
        let mut audio = vec![0.0f32; WINDOW_SAMPLES];
        let n = samples.len().min(WINDOW_SAMPLES);
        audio[..n].copy_from_slice(&samples[..n]);
        // Reflect padding by N_FFT / 2 on both sides (center=True).
        let pad = N_FFT / 2;
        let mut padded = Vec::with_capacity(WINDOW_SAMPLES + 2 * pad);
        padded.extend((1..=pad).rev().map(|i| audio[i]));
        padded.extend_from_slice(&audio);
        padded.extend((1..=pad).map(|i| audio[WINDOW_SAMPLES - 1 - i]));

        let bins = N_FFT / 2 + 1;
        let mut power = vec![0.0f32; bins];
        let mut frame = vec![0.0f32; N_FFT];
        let mut mel = vec![0.0f32; MEL_BINS * FRAMES];
        // The last STFT frame is dropped, as in Whisper.
        for t in 0..FRAMES {
            let start = t * HOP;
            for i in 0..N_FFT {
                frame[i] = padded[start + i] * self.window[i];
            }
            // Silence: skip the transform (its power is zero anyway).
            if frame.iter().all(|&x| x == 0.0) {
                power.iter_mut().for_each(|p| *p = 0.0);
            } else {
                for k in 0..bins {
                    let (c, s) = (&self.cos[k * N_FFT..(k + 1) * N_FFT], &self.sin[k * N_FFT..(k + 1) * N_FFT]);
                    let (mut re, mut im) = (0.0f32, 0.0f32);
                    for i in 0..N_FFT {
                        re += frame[i] * c[i];
                        im += frame[i] * s[i];
                    }
                    power[k] = re * re + im * im;
                }
            }
            for m in 0..MEL_BINS {
                let f = &self.filters[m * bins..(m + 1) * bins];
                let e: f32 = f.iter().zip(&power).map(|(a, b)| a * b).sum();
                mel[m * FRAMES + t] = e.max(1e-10).log10();
            }
        }
        let max = mel.iter().copied().fold(f32::MIN, f32::max);
        mel.iter_mut().for_each(|x| *x = (x.max(max - 8.0) + 4.0) / 4.0);
        mel
    }
}

/// A Whisper model loaded on the NPU.
pub struct QnnWhisper {
    model: *mut QwModel,
    tokens: Tokens,
    mel: MelFrontEnd,
}

// The model is used by one thread at a time (the engine's mutex).
unsafe impl Send for QnnWhisper {}

impl Drop for QnnWhisper {
    fn drop(&mut self) {
        unsafe { qw_close(self.model) }
    }
}

impl QnnWhisper {
    /// True if `dir` looks like an NPU model directory.
    pub fn is_model_dir(dir: &Path) -> bool {
        dir.join("encoder-ep.onnx").is_file() && dir.join("decoder-ep.onnx").is_file()
    }

    /// Loads the model in `dir`; `shared` holds tokens.bin and mel_filters.bin.
    pub fn load(dir: &Path, shared: &Path) -> Result<QnnWhisper, String> {
        let tokens = Tokens::load(&shared.join("tokens.bin"))?;
        let mel = MelFrontEnd::load(&shared.join("mel_filters.bin"))?;
        let c_dir = CString::new(dir.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        let mut err = [0u8; 512];
        let model = unsafe { qw_open(c_dir.as_ptr(), err.as_mut_ptr() as *mut c_char, err.len()) };
        if model.is_null() {
            return Err(format!("NPU model {}: {}", dir.display(), err_string(&err)));
        }
        Ok(QnnWhisper { model, tokens, mel })
    }

    /// Transcribes up to 30 s of 16 kHz audio (longer input is cut; the
    /// engine's runs are at most 15 s). `language` is a Whisper code like
    /// "de"; `None` detects it.
    pub fn transcribe(&mut self, samples: &[f32], language: Option<&str>, translate: bool) -> Result<String, String> {
        let lang_token = match language {
            Some(code) => {
                let code = code.split('-').next().unwrap_or(code).to_ascii_lowercase();
                let at = LANGUAGES.iter().position(|&l| l == code).ok_or_else(|| format!("unsupported language {}", code))?;
                FIRST_LANG_TOKEN + at as i32
            }
            None => -1,
        };
        let started = std::time::Instant::now();
        let mel = self.mel.features(samples);
        let mel_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut out = vec![0 as c_int; MAX_TOKENS];
        let (mut detected, mut enc_ms, mut dec_ms) = (0 as c_int, 0.0f64, 0.0f64);
        let mut err = [0u8; 512];
        let n = unsafe {
            qw_run(
                self.model,
                mel.as_ptr(),
                lang_token,
                translate as c_int,
                out.as_mut_ptr(),
                out.len() as c_int,
                &mut detected,
                &mut enc_ms,
                &mut dec_ms,
                err.as_mut_ptr() as *mut c_char,
                err.len(),
            )
        };
        if n < 0 {
            return Err(format!("NPU run failed: {}", err_string(&err)));
        }
        log::info!(
            "npu: mel {:.0} ms, encoder {:.0} ms, decoder {:.0} ms ({} tokens, language {})",
            mel_ms,
            enc_ms,
            dec_ms,
            n,
            LANGUAGES.get((detected - FIRST_LANG_TOKEN) as usize).unwrap_or(&"?")
        );
        Ok(self.tokens.text(&out[..n as usize]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Matches Hugging Face's WhisperFeatureExtractor on a reference signal
    /// (written by npu/make-assets.py into $QNN_TEST_DIR).
    #[test]
    fn mel_matches_reference() {
        let Ok(dir) = std::env::var("QNN_TEST_DIR") else { return };
        let dir = Path::new(&dir);
        let read = |name: &str| -> Vec<f32> {
            std::fs::read(dir.join(name)).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
        };
        let mel = MelFrontEnd::load(&dir.join("mel_filters.bin")).unwrap().features(&read("mel_test_in.f32"));
        let want = read("mel_test_out.f32");
        let worst = mel.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 2e-3, "largest difference {}", worst);
    }
}
