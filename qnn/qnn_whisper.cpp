// Whisper on the Qualcomm NPU (Hexagon HTP) through ONNX Runtime's QNN
// execution provider. A model directory holds a w8a16 encoder/decoder pair
// precompiled for the HTP by Qualcomm AI Hub (see npu/README.md):
//
//   encoder-ep.onnx, encoder.bin   log-mel [1,128,3000] -> cross-attention K/V
//   decoder-ep.onnx, decoder.bin   one token per call, 199-token self cache
//   qnn.cfg                        the encoder input's quantization
//
// The .onnx files only wrap the context binaries (.bin) in an EPContext node.
// Inputs and outputs are quantized: the mel is uint16, the K/V caches uint8
// (zero point 128), the attention mask uint16 (65535 = attend, 0 = masked).
// Decoding is greedy. src/qnn.rs computes the mel and turns tokens into text.
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kLayers = 4, kHeads = 20, kHeadDim = 64, kCross = 1500, kSelf = 199, kMask = 200;
constexpr int kVocab = 51866, kMelBins = 128, kMelFrames = 3000;
constexpr int kSot = 50258, kEot = 50257, kTranslate = 50359, kTranscribe = 50360, kNoTimestamps = 50364;
constexpr int kFirstLang = 50259, kLastLang = 50358;
constexpr uint16_t kAttend = 65535, kMasked = 0;
constexpr uint8_t kCacheZero = 128;

constexpr size_t kCrossN = size_t(kHeads) * kHeadDim * kCross;
constexpr size_t kSelfN = size_t(kHeads) * kHeadDim * kSelf;
constexpr int64_t kMelShape[] = {1, kMelBins, kMelFrames};
constexpr int64_t kKCrossShape[] = {kHeads, 1, kHeadDim, kCross}, kVCrossShape[] = {kHeads, 1, kCross, kHeadDim};
constexpr int64_t kKSelfShape[] = {kHeads, 1, kHeadDim, kSelf}, kVSelfShape[] = {kHeads, 1, kSelf, kHeadDim};
constexpr int64_t kIdsShape[] = {1, 1}, kPosShape[] = {1}, kMaskShape[] = {1, 1, 1, kMask};
constexpr int64_t kLogitsShape[] = {1, kVocab, 1, 1};

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

Ort::Env & env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "qnn_whisper");
    return env;
}

Ort::SessionOptions session_options() {
    Ort::SessionOptions so;
    // Everything runs on the NPU; ONNX Runtime's own CPU threads would only
    // spin while waiting for it.
    so.SetIntraOpNumThreads(1);
    so.SetInterOpNumThreads(1);
    so.AddConfigEntry("session.intra_op.allow_spinning", "0");
    so.AddConfigEntry("session.inter_op.allow_spinning", "0");
    // Fail instead of silently running the graph on the CPU.
    so.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    std::unordered_map<std::string, std::string> qnn = {
        {"backend_path", "libQnnHtp.so"},
        {"htp_performance_mode", "burst"},
    };
    so.AppendExecutionProvider("QNN", qnn);
    return so;
}

std::string with_names(const char * prefix, int layer, const char * suffix) {
    return std::string(prefix) + std::to_string(layer) + suffix;
}

}  // namespace

struct qw_model {
    Ort::Session encoder{nullptr}, decoder{nullptr};
    float input_scale = 0;
    int input_zero_point = 0;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<uint16_t> mel_q = std::vector<uint16_t>(size_t(kMelBins) * kMelFrames);
    std::vector<std::vector<uint8_t>> cross = std::vector<std::vector<uint8_t>>(2 * kLayers, std::vector<uint8_t>(kCrossN));
    std::vector<std::vector<uint8_t>> self_a = std::vector<std::vector<uint8_t>>(2 * kLayers, std::vector<uint8_t>(kSelfN));
    std::vector<std::vector<uint8_t>> self_b = std::vector<std::vector<uint8_t>>(2 * kLayers, std::vector<uint8_t>(kSelfN));
    std::vector<uint16_t> logits = std::vector<uint16_t>(kVocab);
    std::vector<uint16_t> mask = std::vector<uint16_t>(kMask);

    std::vector<std::string> enc_out_names, dec_in_names, dec_out_names;

    // One decoder step: feeds `token` at `position`, reads the logits and
    // swaps the self caches. `in` is the cache holding the previous steps.
    void step(int32_t token, int32_t position, std::vector<std::vector<uint8_t>> *& in,
              std::vector<std::vector<uint8_t>> *& out) {
        mask[kMask - 1 - position] = kAttend;
        std::vector<Ort::Value> ins, outs;
        ins.push_back(Ort::Value::CreateTensor<int32_t>(mem, &token, 1, kIdsShape, 2));
        ins.push_back(Ort::Value::CreateTensor<int32_t>(mem, &position, 1, kPosShape, 1));
        ins.push_back(Ort::Value::CreateTensor<uint16_t>(mem, mask.data(), kMask, kMaskShape, 4));
        for (int l = 0; l < kLayers; ++l) {
            ins.push_back(Ort::Value::CreateTensor<uint8_t>(mem, cross[2 * l].data(), kCrossN, kKCrossShape, 4));
            ins.push_back(Ort::Value::CreateTensor<uint8_t>(mem, cross[2 * l + 1].data(), kCrossN, kVCrossShape, 4));
            ins.push_back(Ort::Value::CreateTensor<uint8_t>(mem, (*in)[2 * l].data(), kSelfN, kKSelfShape, 4));
            ins.push_back(Ort::Value::CreateTensor<uint8_t>(mem, (*in)[2 * l + 1].data(), kSelfN, kVSelfShape, 4));
            outs.push_back(Ort::Value::CreateTensor<uint8_t>(mem, (*out)[2 * l].data(), kSelfN, kKSelfShape, 4));
            outs.push_back(Ort::Value::CreateTensor<uint8_t>(mem, (*out)[2 * l + 1].data(), kSelfN, kVSelfShape, 4));
        }
        outs.push_back(Ort::Value::CreateTensor<uint16_t>(mem, logits.data(), kVocab, kLogitsShape, 4));
        std::vector<const char *> in_names, out_names;
        for (auto & s : dec_in_names) in_names.push_back(s.c_str());
        for (auto & s : dec_out_names) out_names.push_back(s.c_str());
        decoder.Run(Ort::RunOptions{nullptr}, in_names.data(), ins.data(), ins.size(), out_names.data(), outs.data(),
                    outs.size());
        std::swap(in, out);
    }

    int argmax(int first, int last) const {
        int best = first;
        for (int i = first + 1; i <= last; ++i)
            if (logits[i] > logits[best]) best = i;
        return best;
    }
};

extern "C" {

// Opens the model in `dir`. Returns null and writes `err` on failure (e.g.
// no QNN-capable NPU, or files missing).
qw_model * qw_open(const char * dir, char * err, size_t err_len) {
    try {
        auto m = std::make_unique<qw_model>();
        std::ifstream cfg(std::string(dir) + "/qnn.cfg");
        std::string line;
        while (std::getline(cfg, line)) {
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            auto key = line.substr(0, eq), value = line.substr(eq + 1);
            if (key == "input_scale") m->input_scale = std::stof(value);
            if (key == "input_zero_point") m->input_zero_point = std::stoi(value);
        }
        if (m->input_scale <= 0) throw std::runtime_error(std::string(dir) + "/qnn.cfg: no input_scale");

        m->encoder = Ort::Session(env(), (std::string(dir) + "/encoder-ep.onnx").c_str(), session_options());
        m->decoder = Ort::Session(env(), (std::string(dir) + "/decoder-ep.onnx").c_str(), session_options());

        for (int l = 0; l < kLayers; ++l) {
            m->enc_out_names.push_back(with_names("k_cache_cross_", l, ""));
            m->enc_out_names.push_back(with_names("v_cache_cross_", l, ""));
        }
        m->dec_in_names = {"input_ids", "position_ids", "attention_mask"};
        for (int l = 0; l < kLayers; ++l) {
            m->dec_in_names.push_back(with_names("k_cache_cross_", l, ""));
            m->dec_in_names.push_back(with_names("v_cache_cross_", l, ""));
            m->dec_in_names.push_back(with_names("k_cache_self_", l, "_in"));
            m->dec_in_names.push_back(with_names("v_cache_self_", l, "_in"));
            m->dec_out_names.push_back(with_names("k_cache_self_", l, "_out"));
            m->dec_out_names.push_back(with_names("v_cache_self_", l, "_out"));
        }
        m->dec_out_names.push_back("logits");
        return m.release();
    } catch (const std::exception & e) {
        snprintf(err, err_len, "%s", e.what());
        return nullptr;
    }
}

void qw_close(qw_model * m) { delete m; }

// Transcribes one 30 s window. `mel` is Whisper's normalised log-mel,
// 128 x 3000 floats, row-major by mel bin. `lang` is the language token
// (50259..50358), or < 0 to detect it, which writes the detected token to
// `*detected`. Writes up to `max_out` text tokens to `out` and returns their
// count, or -1 with `err` set.
int qw_run(qw_model * m, const float * mel, int lang, int translate, int * out, int max_out, int * detected,
           double * enc_ms, double * dec_ms, char * err, size_t err_len) {
    try {
        for (size_t i = 0; i < m->mel_q.size(); ++i) {
            const float q = std::nearbyint(mel[i] / m->input_scale) + m->input_zero_point;
            m->mel_q[i] = uint16_t(std::clamp(q, 0.0f, 65535.0f));
        }
        auto t0 = std::chrono::steady_clock::now();
        {
            const char * in_names[] = {"input_features"};
            Ort::Value mel_value = Ort::Value::CreateTensor<uint16_t>(m->mem, m->mel_q.data(), m->mel_q.size(), kMelShape, 3);
            std::vector<const char *> out_names;
            std::vector<Ort::Value> outs;
            for (int i = 0; i < 2 * kLayers; ++i) {
                out_names.push_back(m->enc_out_names[i].c_str());
                outs.push_back(Ort::Value::CreateTensor<uint8_t>(m->mem, m->cross[i].data(), kCrossN,
                                                                 i % 2 ? kVCrossShape : kKCrossShape, 4));
            }
            m->encoder.Run(Ort::RunOptions{nullptr}, in_names, &mel_value, 1, out_names.data(), outs.data(), outs.size());
        }
        *enc_ms = ms_since(t0);

        t0 = std::chrono::steady_clock::now();
        for (auto & b : m->self_a) std::fill(b.begin(), b.end(), kCacheZero);
        std::fill(m->mask.begin(), m->mask.end(), kMasked);
        auto * in = &m->self_a, * next = &m->self_b;
        int position = 0;
        m->step(kSot, position++, in, next);
        if (lang < 0) lang = m->argmax(kFirstLang, kLastLang);
        *detected = lang;
        const int prompt[] = {lang, translate ? kTranslate : kTranscribe, kNoTimestamps};
        for (int t : prompt) m->step(t, position++, in, next);

        int n = 0;
        while (n < max_out) {
            const int token = m->argmax(0, kVocab - 1);
            if (token == kEot || token >= kEot) break;  // end, or a stray special token
            out[n++] = token;
            if (position >= kSelf) break;  // the self cache is full
            m->step(token, position++, in, next);
        }
        *dec_ms = ms_since(t0);
        return n;
    } catch (const std::exception & e) {
        snprintf(err, err_len, "%s", e.what());
        return -1;
    }
}

}  // extern "C"
