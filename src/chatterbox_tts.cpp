// chatterbox_tts: end-to-end text -> wav synthesis using T3-generated speech
// tokens and a pre-computed speaker/prompt reference dump.
//
// Pipeline (all in C++/ggml):
//   flow_input_tokens = concat(prompt_token, speech_tokens_padded)
//   x = input_embedding(flow_input_tokens)               (T, D=512)
//   mu = encoder(x) + upsample2x + encoder_proj          (2T, 80)
//   spks = affine(F.normalize(embedding))                (80,)
//   conds[:mel_len1] = prompt_feat, rest = 0             (2T, 80)
//   z = randn(80, 2T)
//   for t,r in [(0, 0.5), (0.5, 1.0)]:
//       dxdt = estimator(z, mu, t_emb, spks, conds)
//       z = z + (r-t) * dxdt
//   mel = z[:, mel_len1:]                                (80, 2T - mel_len1)
//   f0 = f0_predictor(mel)
//   source = sinegen(upsample(f0)) -> tanh(linear)       (T_wav,)
//   s_stft = stft(source)
//   wav = hift_decode(mel, s_stft) + istft
//
// Usage:
//   chatterbox_tts --s3gen-gguf MODEL.gguf --ref-dir DIR \
//                  --tokens-file TOKENS.txt --out OUT.wav

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "npy.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef GGML_USE_OPENCL
#include "ggml-opencl.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Global thread count (set in main; used to configure CPU backend in each graph run)
static int g_n_threads = 1;

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static void compute(ggml_backend_t backend, ggml_cgraph * gf) {
    if (ggml_backend_is_cpu(backend)) ggml_backend_cpu_set_n_threads(backend, g_n_threads);
    ggml_backend_graph_compute(backend, gf);
}
struct scoped_timer {
    const char * label;
    double t0;
    scoped_timer(const char * l) : label(l), t0(now_ms()) {}
    ~scoped_timer() { fprintf(stderr, "  [%-16s] %.1f ms\n", label, now_ms() - t0); }
};
#define TIMED(label) scoped_timer _st_##__LINE__(label)

// ============================================================================
// GGUF loader + helpers
// ============================================================================

struct model_ctx {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx_w = nullptr;
    ggml_backend_buffer_t buffer_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
};

static ggml_backend_t s3gen_init_backend(int n_gpu_layers, bool verbose) {
#ifdef GGML_USE_CUDA
    if (n_gpu_layers > 0) {
        auto * b = ggml_backend_cuda_init(0);
        if (b) { if (verbose) fprintf(stderr, "s3gen: using CUDA backend\n"); return b; }
    }
#endif
#ifdef GGML_USE_METAL
    if (n_gpu_layers > 0) {
        auto * b = ggml_backend_metal_init();
        if (b) { if (verbose) fprintf(stderr, "s3gen: using Metal backend\n"); return b; }
    }
#endif
#ifdef GGML_USE_VULKAN
    if (n_gpu_layers > 0) {
        auto * b = ggml_backend_vk_init(0);
        if (b) {
            if (verbose) {
                char desc[256] = {0};
                ggml_backend_vk_get_device_description(0, desc, sizeof(desc));
                fprintf(stderr, "s3gen: using Vulkan backend (device 0: %s)\n", desc);
            }
            return b;
        }
    }
#endif
#if defined(GGML_USE_OPENCL)
    if (n_gpu_layers > 0) {
        ggml_backend_reg_t ocl_reg = ggml_backend_opencl_reg();
        if (ocl_reg && ggml_backend_reg_dev_count(ocl_reg) > 0) {
            auto * b = ggml_backend_opencl_init();
            if (b) {
                if (verbose) {
                    fprintf(stderr, "s3gen: using OpenCL backend\n");
                }
                return b;
            }
        } else if (verbose && ocl_reg) {
            if (ggml_backend_reg_dev_count(ocl_reg) == 0) {
                fprintf(stderr, "s3gen: no OpenCL device; using CPU\n");
            } else {
                fprintf(stderr, "s3gen: OpenCL init failed; using CPU\n");
            }
        }
    }
#endif
    auto * b = ggml_backend_cpu_init();
    if (!b) throw std::runtime_error("ggml_backend_cpu_init() failed");
    if (verbose) fprintf(stderr, "s3gen: using CPU backend\n");
    return b;
}

// Process-wide cache of the loaded S3Gen GGUF so repeated calls (streaming
// or server mode) pay the ~700 ms tensor-load cost only once.  Keyed on
// (path, n_gpu_layers) — switching backend invalidates the cache.
//
// Thread-safety: simple mutex around load/lookup.  The returned pointer
// stays alive for the lifetime of the process (we never evict), which
// matches the streaming CLI's single-voice use case.  A proper LRU would
// belong in a server front-end.
static model_ctx load_s3gen_gguf(const std::string & path, int n_gpu_layers, bool verbose);

namespace {
struct s3gen_cache_entry { std::string path; int gpu = 0; std::unique_ptr<model_ctx> m; };
static std::mutex                            g_s3gen_cache_mu;
static std::unique_ptr<s3gen_cache_entry>    g_s3gen_cache_entry;
static double                                g_s3gen_cache_last_load_ms = 0.0;

// QVAC-17872 round-2 (§5.7): keep the per-stage gallocr / ggml_context /
// graph-arena alive across calls so the second utterance does not pay
// gallocr_new + Vulkan buffer alloc + reserve overhead 6× (Turbo) / 30×
// (MTL).  Each entry is bound to the s3gen backend and released by
// s3gen_model_cache_release before the Vulkan dylib tears down.  Threading
// matches s3gen_model_cache: only touched from the synthesize call path,
// which holds g_s3gen_cache_mu over the entire wav.
//
// Two flavours:
//   * stage_graph_cache_fixed   — graph topology never changes (e.g.
//     compute_time_mlp's TDIM=320 → OUT=1024 mapping); built once,
//     re-used on every call by re-binding inputs.
//   * stage_graph_cache_keyed   — graph topology depends on a small
//     integer key (T, T_mel, T_stft).  Re-built only when the key
//     changes; the gallocr's underlying buffer pool is grown by reserve.
struct stage_graph_cache_fixed {
    ggml_context *       ctx     = nullptr;
    ggml_cgraph *        gf      = nullptr;
    ggml_gallocr_t       allocr  = nullptr;
    std::vector<uint8_t> buf;
    bool                 built   = false;
    void destroy() {
        if (allocr) { ggml_gallocr_free(allocr); allocr = nullptr; }
        if (ctx)    { ggml_free(ctx);            ctx    = nullptr; }
        gf    = nullptr;
        built = false;
        buf   = std::vector<uint8_t>();
    }
};
struct stage_graph_cache_keyed {
    int                  key     = -1;        // T / T_mel / T_stft
    ggml_context *       ctx     = nullptr;
    ggml_cgraph *        gf      = nullptr;
    ggml_gallocr_t       allocr  = nullptr;
    std::vector<uint8_t> buf;
    // QVAC-17872 round-3 (§5.7-cache-hit): true once the graph's constant
    // input tensors (pos_emb, STFT kernel, w_sum, inv_alphas, …) have been
    // uploaded to GPU.  Subsequent compute calls skip the re-upload because
    // ggml_gallocr_alloc_graph re-assigns the same offsets for an unchanged
    // graph topology — the GPU-side data persists.  Cleared on destroy().
    bool                 inputs_uploaded = false;
    void destroy() {
        if (allocr) { ggml_gallocr_free(allocr); allocr = nullptr; }
        if (ctx)    { ggml_free(ctx);            ctx    = nullptr; }
        gf  = nullptr;
        key = -1;
        buf = std::vector<uint8_t>();
        inputs_uploaded = false;
    }
};

// hift_decode needs to additionally keep the inv_alpha (graph-name, host-
// data) pairs alongside the cached graph so we can re-upload them on
// every alloc_graph (the host data is constant, but the GPU buffer is
// re-bound on each call).  These are derived from the model weights and
// computed on graph build (CPU side: invert_alpha_cpu).
struct hift_graph_cache : stage_graph_cache_keyed {
    std::vector<std::pair<std::string, std::vector<float>>> inv_alphas;
    // QVAC-17872 round-3 (§5.7-cache-hit): cache the CPU-side constant
    // inputs computed inside run_hift_decode (build_hann_window /
    // build_istft_kernel / build_window_sum) so we only do them once per
    // (T_mel, T_stft) graph build, not on every cached call.
    std::vector<float> istft_k_data;
    std::vector<float> w_sum_data;
    void destroy() {
        stage_graph_cache_keyed::destroy();
        inv_alphas.clear();
        istft_k_data.clear();
        w_sum_data.clear();
    }
};
// Same idea for encoder and stft: the CPU-side constant inputs (pos_emb,
// STFT kernel) live alongside the cached graph and are re-uploaded only
// when inputs_uploaded is false (right after a rebuild).
struct encoder_graph_cache : stage_graph_cache_keyed {
    std::vector<float> pe1_data;  // pos_emb at length T
    std::vector<float> pe2_data;  // pos_emb at length 2T
    void destroy() {
        stage_graph_cache_keyed::destroy();
        pe1_data.clear();
        pe2_data.clear();
    }
};
struct stft_graph_cache : stage_graph_cache_keyed {
    std::vector<float> kernel_data;  // STFT kernel; constant for fixed n_fft
    void destroy() {
        stage_graph_cache_keyed::destroy();
        kernel_data.clear();
    }
};

// QVAC-17872 round-HIFT (2026-05-04): persistent CFM estimator graph.
// Same explicit-destroy() lifetime pattern as the per-stage caches above.
// The previous local-scope-in-synthesize() version paid the full graph
// rebuild cost on every synth call (the 256 MB `buf` allocation + all
// ggml_new_tensor + op-build calls + ggml_gallocr_reserve which allocates
// the device-side buffer pool).  In multi-synth mode (one synth per chunk)
// this dominated cfm_total on chunks 2..N for the same T.  Global lifetime
// makes the build run once per (model, T) tuple; every subsequent synth
// call's first step takes the warm-cache path.
struct cfm_estimator_cache {
    int T = -1;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t allocr = nullptr;
    std::vector<uint8_t> buf;
    void destroy() {
        if (allocr) { ggml_gallocr_free(allocr); allocr = nullptr; }
        if (ctx)    { ggml_free(ctx);            ctx    = nullptr; }
        gf  = nullptr;
        T   = -1;
        buf = std::vector<uint8_t>();
    }
};

// One global instance per stage.  Lifetime tied to the s3gen model cache
// so we never outlive the backend they hold gallocr handles into.
static stage_graph_cache_fixed g_time_mlp_cache;
static stage_graph_cache_fixed g_time_mixed_cache;
static encoder_graph_cache     g_encoder_cache;
static stage_graph_cache_keyed g_f0_cache;
static stft_graph_cache        g_stft_cache;
static hift_graph_cache        g_hift_cache;
static cfm_estimator_cache     g_cfm_estimator_cache;

// QVAC-17872 round-HIFT (2026-05-04): CPU-side mirror of large model
// weights that synthesize() needs to read every call (input_embedding
// lookup table, speaker affine matrix, etc.).  These are model constants
// — on a GPU backend each call previously paid an N MB device→host
// download; the input_embedding alone is 13.4 MB on the Turbo build
// (D=512 × vocab=6561 × 4 B).  Cached per (model_ctx pointer + tensor
// pointer); cleared in s3gen_model_cache_release alongside the existing
// graph caches so we never serve stale CPU data after a backend swap.
static std::unordered_map<const ggml_tensor *, std::vector<float>> g_weight_cpu_mirror;
static std::mutex                                                  g_weight_cpu_mirror_mu;

static const float * cached_cpu_weights_f32(const ggml_tensor * t) {
    {
        std::lock_guard<std::mutex> lk(g_weight_cpu_mirror_mu);
        auto it = g_weight_cpu_mirror.find(t);
        if (it != g_weight_cpu_mirror.end()) return it->second.data();
    }
    std::vector<float> data(ggml_nelements(t));
    ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
    {
        std::lock_guard<std::mutex> lk(g_weight_cpu_mirror_mu);
        auto [it, inserted] = g_weight_cpu_mirror.emplace(t, std::move(data));
        return it->second.data();
    }
}

// QVAC-17872 round-HIFT (2026-05-04): cache time-embedding results by
// scalar t-value.  In the standard 2-step Euler loop t_span = [0, 0.5, 1]
// yields call pairs (t=0, r=0.5) then (t=0.5, r=1) — compute_time_mlp(0.5)
// is invoked twice per synth call (as r in step 0, as t in step 1), and
// every t-value repeats across all subsequent synth calls.  Caching the
// MLP and mixer outputs by t-value collapses up to 6 graph submissions /
// inference (cfm_steps=2) down to 0 after the first inference warms the
// cache.
//
// Float keys use bitcast → uint32_t so IEEE equality semantics (incl.
// -0.0/+0.0 being equal-by-value but different bits, NaN-not-equal-to-NaN)
// match how the caller obtains the float values (literal const-folded
// values from t_span[i] = (float)i / (float)cfm_steps).
//
// The cache is bound to the s3gen model lifecycle: cleared in
// s3gen_model_cache_release alongside the existing graph caches.  This
// also handles CHATTERBOX_F16_CFM mode flips (model is reloaded → cache
// cleared) since the time_mlp / time_embed_mixer weights are F16-converted
// in C1 mode and would yield numerically different (deterministic but
// reduction-order-dependent) outputs.
//
// Thread-safety: the synthesize call path serialises on g_s3gen_cache_mu
// already (one synth at a time process-wide), so the unguarded map is
// safe in practice.  Add a dedicated mutex anyway to make the contract
// explicit and let any future concurrent caller be safe.
static std::unordered_map<uint32_t, std::vector<float>> g_time_mlp_results;
static std::unordered_map<uint64_t, std::vector<float>> g_time_emb_results;
static std::mutex                                       g_time_emb_results_mu;
}  // namespace

// Release any cached model_ctx (frees its backend buffer, ggml context and
// backend).  Must run before the ggml-metal / ggml-cuda / ggml-vulkan dylib
// tears down its static device list; otherwise their static destructors hit
// a "rsets->data count != 0" assert (residency sets still referenced by an
// orphan backend buffer).  We register it with atexit() on first cache
// insertion so it runs before process-exit dylib finalisers.
static void s3gen_model_cache_release() {
    std::lock_guard<std::mutex> lk(g_s3gen_cache_mu);
    // QVAC-17872 round-2: tear down the per-stage gallocr/ctx caches BEFORE
    // freeing the backend, because gallocr_free transitively frees Vulkan
    // buffers that were allocated against this backend.  After the backend
    // is gone, gallocr_free would touch a dangling vk_device.
    g_time_mlp_cache.destroy();
    g_time_mixed_cache.destroy();
    g_encoder_cache.destroy();
    g_f0_cache.destroy();
    g_stft_cache.destroy();
    g_hift_cache.destroy();
    g_cfm_estimator_cache.destroy();
    {
        // QVAC-17872 round-HIFT: clear cached time-embedding results
        // alongside the graph caches they were built against.
        std::lock_guard<std::mutex> lk(g_time_emb_results_mu);
        g_time_mlp_results.clear();
        g_time_emb_results.clear();
    }
    {
        // QVAC-17872 round-HIFT: clear cached CPU weight mirrors;
        // the underlying ggml_tensor pointers belong to the soon-to-be-
        // freed model context, so we must drop them here.
        std::lock_guard<std::mutex> lk(g_weight_cpu_mirror_mu);
        g_weight_cpu_mirror.clear();
    }
    if (!g_s3gen_cache_entry) return;
    model_ctx * m = g_s3gen_cache_entry->m.get();
    if (m) {
        if (m->buffer_w) { ggml_backend_buffer_free(m->buffer_w); m->buffer_w = nullptr; }
        if (m->ctx_w)    { ggml_free(m->ctx_w);                   m->ctx_w    = nullptr; }
        if (m->backend)  { ggml_backend_free(m->backend);         m->backend  = nullptr; }
        m->tensors.clear();
    }
    g_s3gen_cache_entry.reset();
}

static model_ctx * s3gen_model_cache_get(const std::string & path, int n_gpu_layers, bool verbose) {
    std::lock_guard<std::mutex> lk(g_s3gen_cache_mu);
    if (g_s3gen_cache_entry &&
        g_s3gen_cache_entry->path == path &&
        g_s3gen_cache_entry->gpu  == n_gpu_layers) {
        if (verbose) {
            fprintf(stderr, "  %zu tensors (cached — skip GGUF load)\n",
                    g_s3gen_cache_entry->m->tensors.size());
        }
        g_s3gen_cache_last_load_ms = 0.0;
        return g_s3gen_cache_entry->m.get();
    }
    // Cache miss → switching backend.  The per-stage gallocr/ctx caches
    // hold buffers allocated against the OLD backend; tear them down here,
    // before we drop the old s3gen_cache_entry (and thus the old backend),
    // for the same reason as in s3gen_model_cache_release.
    if (g_s3gen_cache_entry) {
        g_time_mlp_cache.destroy();
        g_time_mixed_cache.destroy();
        g_encoder_cache.destroy();
        g_f0_cache.destroy();
        g_stft_cache.destroy();
        g_hift_cache.destroy();
        g_cfm_estimator_cache.destroy();
        // QVAC-17872 round-HIFT: also flush the t-emb result caches and
        // CPU weight mirrors when the backend swaps; the cached vectors
        // were computed against the old model's weights / tensor ptrs.
        {
            std::lock_guard<std::mutex> lk(g_time_emb_results_mu);
            g_time_mlp_results.clear();
            g_time_emb_results.clear();
        }
        {
            std::lock_guard<std::mutex> lk(g_weight_cpu_mirror_mu);
            g_weight_cpu_mirror.clear();
        }
    }
    if (verbose) fprintf(stderr, "Loading %s\n", path.c_str());
    double t0 = now_ms();
    auto m = std::make_unique<model_ctx>(load_s3gen_gguf(path, n_gpu_layers, verbose));
    g_s3gen_cache_last_load_ms = now_ms() - t0;
    if (verbose) fprintf(stderr, "  %zu tensors loaded (%.1f ms)\n", m->tensors.size(), g_s3gen_cache_last_load_ms);
    g_s3gen_cache_entry = std::make_unique<s3gen_cache_entry>(
        s3gen_cache_entry{path, n_gpu_layers, std::move(m)});

    // Register the release on first insertion.  atexit() handlers run in
    // LIFO and execute before any static destructors in DSOs loaded *after*
    // this point — which on macOS / Linux is how we avoid Metal's device
    // teardown assert on process exit.
    static bool registered = false;
    if (!registered) {
        std::atexit(s3gen_model_cache_release);
        registered = true;
    }
    return g_s3gen_cache_entry->m.get();
}

static double s3gen_model_cache_last_load_ms() { return g_s3gen_cache_last_load_ms; }

// QVAC-17872 round-C1 (opt-in, default OFF): convert CFM matmul `src0`
// weights from F32 to F16 at s3gen-model load time.  Unlocks F16
// tensor-core paths on NVIDIA / AMD RDNA3+ Vulkan; halves CFM matmul
// weight bandwidth on every other Vulkan device.  Roughly doubles
// matmul throughput on RTX 5090 (warm S3GEN_INFER 130 ms → ~90 ms
// projected, see DECISION_C1_F16_CFM_WEIGHTS.md and
// FINDINGS_ROUND3.md §3 Observation 4).
//
// **Breaks bit-exact** vs the round-1/2/3 F32 MD5 invariants —
// off by default; user must opt in by setting CHATTERBOX_F16_CFM=1.
// When OFF (no env-var, "0", "false", or empty), every code path is
// byte-identical to round-6: CFM tensors load as F32, every locked MD5
// still matches.
//
// Scope: only F32→F16 conversion of matmul *src0* weights.  We do NOT
// touch:
//   - biases (1D, used as src0 of `add` only — F16 makes no sense)
//   - LayerNorm / norm scales (1D, used as `mul` operand)
//   - convolution kernels going through conv1d_f32 (the kernel is
//     reshaped and used as src1 of mul_mat in im2col, so F16 wouldn't
//     unlock tensor cores; also conv1d_f32's own implementation
//     materialises an F32 im2col tensor)
//   - cfm/final_proj/weight (used via conv1d_f32, same reason)
//
// Activations and accumulators stay F32; ggml_mul_mat(F16, F32) is
// the standard "weight × activation" pattern that every backend
// chatterbox ships (Vulkan, Metal, OpenCL, CUDA, CPU) supports
// natively.
//
// Counts on chatterbox-s3gen-turbo.gguf: 353 matmul weights
// (251 MB F32 → 126 MB F16, ~125 MB saved on the device).
static bool cfm_f16_enabled() {
    static const bool enabled = []() {
        const char * e = getenv("CHATTERBOX_F16_CFM");
        if (!e) return false;
        const std::string s = e;
        return !(s == "0" || s == "false" || s == "FALSE" || s.empty());
    }();
    return enabled;
}

static bool is_cfm_matmul_src0_weight(const std::string & name) {
    if (name.compare(0, 4, "cfm/") != 0) return false;
    // Top-level CFM matmul weights (3 tensors).
    if (name == "cfm/time_mlp/linear_1/weight")     return true;
    if (name == "cfm/time_mlp/linear_2/weight")     return true;
    if (name == "cfm/time_embed_mixer/weight")      return true;
    // resnet time-projection MLP inside down_blocks/<i>/0, mid_blocks/<i>/0,
    // up_blocks/<i>/0 (14 tensors total: 1 down + 12 mid + 1 up).
    if (name.find("/0/mlp/1/weight") != std::string::npos) return true;
    // basic_tfm matmuls inside down_blocks/<i>/1/<j>, mid_blocks/<i>/1/<j>,
    // up_blocks/<i>/1/<j> (336 tensors: 56 tfms × 6 weights each).
    if (name.find("/1/") != std::string::npos) {
        const char * suffixes[] = {
            "/attn1/to_q/weight",
            "/attn1/to_k/weight",
            "/attn1/to_v/weight",
            "/attn1/to_out/0/weight",
            "/ff/net/0/proj/weight",
            "/ff/net/2/weight",
        };
        for (const char * s : suffixes) {
            const size_t slen = std::strlen(s);
            if (name.size() >= slen && name.compare(name.size() - slen, slen, s) == 0) return true;
        }
    }
    return false;
}

static model_ctx load_s3gen_gguf(const std::string & path, int n_gpu_layers, bool verbose) {
    model_ctx m;
    ggml_context * tmp_ctx = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ false, /*.ctx=*/ &tmp_ctx };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) throw std::runtime_error("gguf_init_from_file failed: " + path);
    m.backend = s3gen_init_backend(n_gpu_layers, verbose);
    int64_t n_tensors = gguf_get_n_tensors(g);
    ggml_init_params p = { ggml_tensor_overhead() * (size_t)n_tensors, nullptr, true };
    m.ctx_w = ggml_init(p);

    const bool f16_cfm = cfm_f16_enabled();
    int64_t n_converted = 0;
    size_t bytes_saved = 0;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * src = ggml_get_tensor(tmp_ctx, name);
        const bool convert = f16_cfm
            && src->type == GGML_TYPE_F32
            && is_cfm_matmul_src0_weight(name);
        ggml_tensor * dst;
        if (convert) {
            dst = ggml_new_tensor(m.ctx_w, GGML_TYPE_F16, ggml_n_dims(src), src->ne);
            n_converted++;
            bytes_saved += ggml_nbytes(src) / 2;  // F32 -> F16 = exactly half
        } else {
            dst = ggml_dup_tensor(m.ctx_w, src);
        }
        ggml_set_name(dst, name);
        m.tensors[name] = dst;
    }

    if (verbose && f16_cfm) {
        fprintf(stderr,
                "  CHATTERBOX_F16_CFM=1: converted %lld CFM matmul weights to F16 (~%.1f MB saved)\n",
                (long long) n_converted, bytes_saved / (1024.0 * 1024.0));
    }

    m.buffer_w = ggml_backend_alloc_ctx_tensors(m.ctx_w, m.backend);
    for (ggml_tensor * cur = ggml_get_first_tensor(m.ctx_w); cur; cur = ggml_get_next_tensor(m.ctx_w, cur)) {
        ggml_tensor * src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
        if (cur->type == src->type) {
            ggml_backend_tensor_set(cur, ggml_get_data(src), 0, ggml_nbytes(src));
        } else if (cur->type == GGML_TYPE_F16 && src->type == GGML_TYPE_F32) {
            const int64_t n = ggml_nelements(src);
            std::vector<ggml_fp16_t> f16_buf((size_t) n);
            ggml_fp32_to_fp16_row((const float *) ggml_get_data(src), f16_buf.data(), n);
            ggml_backend_tensor_set(cur, f16_buf.data(), 0, ggml_nbytes(cur));
        } else {
            throw std::runtime_error(
                std::string("load_s3gen_gguf: unexpected dtype change at ") + ggml_get_name(cur));
        }
    }
    gguf_free(g);
    ggml_free(tmp_ctx);
    return m;
}

static ggml_tensor * find_tensor(const model_ctx & m, const std::string & name) {
    auto it = m.tensors.find(name);
    if (it == m.tensors.end()) throw std::runtime_error("tensor not found: " + name);
    return it->second;
}

// F32 conv1d via im2col + mul_mat.  kernel ne=[K, IC, OC]
static ggml_tensor * conv1d_f32(ggml_context * ctx, ggml_tensor * kernel, ggml_tensor * input,
                                int stride, int padding, int dilation) {
    ggml_tensor * im2col = ggml_im2col(ctx, kernel, input, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor * result = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
    return ggml_reshape_3d(ctx, result, im2col->ne[1], kernel->ne[2], im2col->ne[2]);
}

// QVAC-17872 round-HIFT (2026-05-04): drop the trailing ggml_cont.  The
// only caller is run_hift_decode's upsample loop, where the result is
// immediately consumed by ggml_add(x, ggml_reshape_2d(bias)) — same
// strided-tolerant pattern as round-AUDIT's pre_lookahead exit cont.
// The view's nb[1]/nb[2] are the original out's strides (which span the
// pre-trim length), so element-wise add iterates with the proper byte
// offsets.  After add, x is a fresh contiguous tensor again, so the
// downstream ggml_view_3d / ggml_concat / rb_fwd → conv1d_f32 chain sees
// contig input.  Saves 3 dispatches per HiFT decode (1 per ups stage).
static ggml_tensor * conv_transpose_1d_f32(ggml_context * ctx, ggml_tensor * kernel,
                                           ggml_tensor * input, int stride, int padding) {
    ggml_tensor * out = ggml_conv_transpose_1d(ctx, kernel, input, stride, 0, 1);
    if (padding == 0) return out;
    int64_t L_new = out->ne[0] - 2 * padding;
    return ggml_view_3d(ctx, out, L_new, out->ne[1], out->ne[2],
                        out->nb[1], out->nb[2], (size_t)padding * out->nb[0]);
}

// QVAC-17872 round-V5 (2026-04-30): replaced the previous
// concat(scale(view, 0), x) emulation with a single ggml_pad_ext call.
// All four backends chatterbox.cpp ships against support per-dim
// front/back padding natively today:
//   * Vulkan    — pad.comp shader takes lp0..lp3 + rp0..rp3 (always has)
//   * Metal     — round-1's ggml-metal-chatterbox-ops.patch added lp0..lp3
//   * CUDA      — pad.cu takes (lp0,rp0,lp1,rp1,...,lp3,rp3) per-dim
//   * OpenCL    — kernel_pad with lp/rp params
// CPU has the reference impl that always handled it.
//
// The previous workaround built a 3-dispatch chain per padded edge
// (ggml_view_4d → ggml_scale(_, 0.0f) → ggml_concat) for both front and
// back padding when p_front>0 OR p_back>0; ~30 such calls per CFM step
// on chatterbox-multilingual gave us ~60 redundant scale + concat
// dispatches per CFM step on top of the single intended pad.
//
// ggml_pad_ext produces a single dispatch that reads src directly into
// the (front_pad + src + back_pad)-sized destination, writing zeros for
// the padded regions.  Output is byte-identical to the previous
// emulation for F32 inputs (0.0 == 0.0; the src region is a literal
// element copy either way; only the floating-point operation count
// differs).
//
// Round-5's "drop wasted ggml_cont" cleanup is now subsumed by this
// change — ggml_pad_ext doesn't need a contiguous source view at all.
static ggml_tensor * zero_pad_dim0(ggml_context * ctx, ggml_tensor * x, int p_front, int p_back) {
    if (p_front <= 0 && p_back <= 0) return x;
    GGML_ASSERT(p_front >= 0 && p_back >= 0);
    return ggml_pad_ext(ctx, x, p_front, p_back, 0, 0, 0, 0, 0, 0);
}

static ggml_tensor * ggml_mish_fn(ggml_context * ctx, ggml_tensor * x) {
    ggml_tensor * sp = ggml_unary(ctx, x, GGML_UNARY_OP_SOFTPLUS);
    ggml_tensor * th = ggml_unary(ctx, sp, GGML_UNARY_OP_TANH);
    return ggml_mul(ctx, x, th);
}

static ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x,
                                ggml_tensor * w, ggml_tensor * b, float eps = 1e-5f) {
    ggml_tensor * y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, w);
    return ggml_add(ctx, y, b);
}

// LayerNorm on channel axis where x ne=[T, C].
//
// QVAC-17872 round-AUDIT: tried dropping the trailing ggml_cont; binary
// crashed in some downstream op (likely conv1d_f32's im2col or ggml_concat
// — both common consumers in cfm_causal_block).  Reverted.  The trailing
// cont is genuinely necessary for the chatterbox CFM graph topology.
static ggml_tensor * layer_norm_on_channel(ggml_context * ctx, ggml_tensor * x,
                                           ggml_tensor * w, ggml_tensor * b, float eps = 1e-5f) {
    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    xt = ggml_norm(ctx, xt, eps);
    xt = ggml_mul(ctx, xt, w);
    xt = ggml_add(ctx, xt, b);
    return ggml_cont(ctx, ggml_permute(ctx, xt, 1, 0, 2, 3));
}

static ggml_tensor * reflect_pad_1d(ggml_context * ctx, ggml_tensor * x, int p_left, int p_right) {
    ggml_tensor * y = x;
    for (int i = 0; i < p_left; ++i) {
        int src_idx = p_left - i;
        ggml_tensor * s = ggml_view_3d(ctx, x, 1, x->ne[1], x->ne[2], x->nb[1], x->nb[2], (size_t)src_idx * x->nb[0]);
        s = ggml_cont(ctx, s);
        y = ggml_concat(ctx, s, y, 0);
    }
    int L_orig = (int)x->ne[0];
    for (int i = 0; i < p_right; ++i) {
        int src_idx = L_orig - 2 - i;
        ggml_tensor * s = ggml_view_3d(ctx, x, 1, x->ne[1], x->ne[2], x->nb[1], x->nb[2], (size_t)src_idx * x->nb[0]);
        s = ggml_cont(ctx, s);
        y = ggml_concat(ctx, y, s, 0);
    }
    return y;
}

// ============================================================================
// Encoder (Conformer) — produces mu for CFM
// ============================================================================

struct conformer_w {
    ggml_tensor *norm_mha_w, *norm_mha_b, *norm_ff_w, *norm_ff_b;
    ggml_tensor *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *o_w, *o_b;
    ggml_tensor *pos_w, *pos_bias_u, *pos_bias_v;
    ggml_tensor *ff1_w, *ff1_b, *ff2_w, *ff2_b;
};

static conformer_w load_conformer(const model_ctx & m, const std::string & pfx) {
    conformer_w w;
    w.norm_mha_w = find_tensor(m, pfx + "/norm_mha/w");
    w.norm_mha_b = find_tensor(m, pfx + "/norm_mha/b");
    w.norm_ff_w  = find_tensor(m, pfx + "/norm_ff/w");
    w.norm_ff_b  = find_tensor(m, pfx + "/norm_ff/b");
    w.q_w = find_tensor(m, pfx + "/attn/q/w"); w.q_b = find_tensor(m, pfx + "/attn/q/b");
    w.k_w = find_tensor(m, pfx + "/attn/k/w"); w.k_b = find_tensor(m, pfx + "/attn/k/b");
    w.v_w = find_tensor(m, pfx + "/attn/v/w"); w.v_b = find_tensor(m, pfx + "/attn/v/b");
    w.o_w = find_tensor(m, pfx + "/attn/o/w"); w.o_b = find_tensor(m, pfx + "/attn/o/b");
    w.pos_w = find_tensor(m, pfx + "/attn/pos/w");
    w.pos_bias_u = find_tensor(m, pfx + "/attn/pos_bias_u");
    w.pos_bias_v = find_tensor(m, pfx + "/attn/pos_bias_v");
    w.ff1_w = find_tensor(m, pfx + "/ff/w1/w"); w.ff1_b = find_tensor(m, pfx + "/ff/w1/b");
    w.ff2_w = find_tensor(m, pfx + "/ff/w2/w"); w.ff2_b = find_tensor(m, pfx + "/ff/w2/b");
    return w;
}

static ggml_tensor * conformer_block(ggml_context * ctx, const conformer_w & w,
                                     ggml_tensor * x, ggml_tensor * pos_emb,
                                     int D, int T, int H, int HD, float eps = 1e-12f) {
    ggml_tensor * residual = x;
    ggml_tensor * xn = ggml_norm(ctx, x, eps);
    xn = ggml_add(ctx, ggml_mul(ctx, xn, w.norm_mha_w), w.norm_mha_b);

    // QVAC-17872 round-6: encoder-side analog of the round-4 CFM
    // basic_tfm fusion.  conformer_block also issues 3 separate
    // ggml_mul_mat for Q / K / V on the same `xn` input — but unlike
    // basic_tfm it adds a bias to each output, so we keep the three
    // adds outside the fused path.  10 conformer blocks per encoder
    // (6 lower at T=326 + 4 upper at T=652) → 30 mul_mat dispatches
    // saved per encoder run with the fusion enabled.
    //
    // The same env-var (CHATTERBOX_FUSE_QKV) gates both round-4 (CFM
    // basic_tfm) and this round-6 (encoder conformer_block) fusion;
    // setting it to 0 reverts both to the round-3 baseline.
    static const bool fuse_qkv = []() {
        const char * e = getenv("CHATTERBOX_FUSE_QKV");
        if (!e) return true;
        const std::string s = e;
        return !(s == "0" || s == "false" || s == "FALSE" || s.empty());
    }();
    ggml_tensor * q;
    ggml_tensor * k;
    ggml_tensor * v;
    if (fuse_qkv) {
        const int INNER = (int) w.q_w->ne[1];
        ggml_tensor * w_q3  = ggml_reshape_3d(ctx, w.q_w, w.q_w->ne[0], w.q_w->ne[1], 1);
        ggml_tensor * w_k3  = ggml_reshape_3d(ctx, w.k_w, w.k_w->ne[0], w.k_w->ne[1], 1);
        ggml_tensor * w_v3  = ggml_reshape_3d(ctx, w.v_w, w.v_w->ne[0], w.v_w->ne[1], 1);
        ggml_tensor * w_qk  = ggml_concat(ctx, w_q3, w_k3, /*dim=*/2);
        ggml_tensor * w_qkv = ggml_concat(ctx, w_qk, w_v3, /*dim=*/2);  // ne=(D, INNER, 3)
        ggml_tensor * xn_b3 = ggml_repeat_4d(ctx, xn, xn->ne[0], xn->ne[1], 3, 1);
        ggml_tensor * qkv   = ggml_mul_mat(ctx, w_qkv, xn_b3);          // ne=(INNER, T, 3)
        const size_t batch_stride = (size_t) INNER * T * sizeof(float);
        // Each Q/K/V is a contiguous view; the post-split bias add is
        // bit-exact equivalent to the original `add(mul_mat, bias)`.
        q = ggml_add(ctx, ggml_view_2d(ctx, qkv, INNER, T, INNER * sizeof(float), 0),                  w.q_b);
        k = ggml_add(ctx, ggml_view_2d(ctx, qkv, INNER, T, INNER * sizeof(float), batch_stride),      w.k_b);
        v = ggml_add(ctx, ggml_view_2d(ctx, qkv, INNER, T, INNER * sizeof(float), 2 * batch_stride),  w.v_b);
    } else {
        q = ggml_add(ctx, ggml_mul_mat(ctx, w.q_w, xn), w.q_b);
        k = ggml_add(ctx, ggml_mul_mat(ctx, w.k_w, xn), w.k_b);
        v = ggml_add(ctx, ggml_mul_mat(ctx, w.v_w, xn), w.v_b);
    }
    ggml_tensor * p = ggml_mul_mat(ctx, w.pos_w, pos_emb);

    q = ggml_reshape_3d(ctx, q, HD, H, T);
    k = ggml_reshape_3d(ctx, k, HD, H, T);
    v = ggml_reshape_3d(ctx, v, HD, H, T);
    p = ggml_reshape_3d(ctx, p, HD, H, pos_emb->ne[1]);

    // QVAC-17872 round-AUDIT:
    //   q_perm — consumed only by ggml_add (lines 658-659 producing
    //            q_plus_u / q_plus_v); add accepts strided src0.  Cont
    //            is redundant.
    //   v_perm — consumed only by ggml_permute → ggml_cont on line 678
    //            (v_for_mm).  The trailing cont materialises contig output
    //            regardless of v_perm's stride state, so the cont here is
    //            redundant.
    //   k_perm and p_perm KEEP their conts — they're src0 of ggml_mul_mat
    //            (lines 661-662) which requires contiguous src0.
    ggml_tensor * q_perm = ggml_permute(ctx, q, 0, 2, 1, 3);
    ggml_tensor * k_perm = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor * v_perm = ggml_permute(ctx, v, 0, 2, 1, 3);
    ggml_tensor * p_perm = ggml_cont(ctx, ggml_permute(ctx, p, 0, 2, 1, 3));

    ggml_tensor * u_bias = ggml_reshape_3d(ctx, w.pos_bias_u, HD, 1, H);
    ggml_tensor * v_bias = ggml_reshape_3d(ctx, w.pos_bias_v, HD, 1, H);
    ggml_tensor * q_plus_u = ggml_add(ctx, q_perm, u_bias);
    ggml_tensor * q_plus_v = ggml_add(ctx, q_perm, v_bias);

    ggml_tensor * ac = ggml_mul_mat(ctx, k_perm, q_plus_u);
    ggml_tensor * bd = ggml_mul_mat(ctx, p_perm, q_plus_v);
    ggml_tensor * bd_padded = zero_pad_dim0(ctx, bd, 1, 0);
    ggml_tensor * bd_viewed = ggml_reshape_3d(ctx, bd_padded, T, 2*T, H);
    ggml_tensor * bd_sliced = ggml_view_3d(ctx, bd_viewed, T, 2*T - 1, H,
                                           bd_viewed->nb[1], bd_viewed->nb[2], bd_viewed->nb[1]);
    ggml_tensor * bd_reshaped = ggml_reshape_3d(ctx, ggml_cont(ctx, bd_sliced), 2*T - 1, T, H);
    ggml_tensor * bd_final = ggml_view_3d(ctx, bd_reshaped, T, T, H,
                                          bd_reshaped->nb[1], bd_reshaped->nb[2], 0);
    // QVAC-17872 round-AUDIT: drop the trailing ggml_cont — bd_final is
    // consumed only as src1 of ggml_add(ac, bd_final) below, which handles
    // strided src1.  Saves 1 dispatch per conformer_block call (10 calls
    // per encoder run).
    ggml_tensor * scores = ggml_add(ctx, ac, bd_final);
    scores = ggml_scale(ctx, scores, 1.0f / std::sqrt((float)HD));
    ggml_tensor * attn = ggml_soft_max(ctx, scores);

    ggml_tensor * v_for_mm = ggml_cont(ctx, ggml_permute(ctx, v_perm, 1, 0, 2, 3));
    ggml_tensor * attn_v = ggml_mul_mat(ctx, v_for_mm, attn);
    ggml_tensor * merged = ggml_cont(ctx, ggml_permute(ctx, attn_v, 0, 2, 1, 3));
    ggml_tensor * flat = ggml_reshape_2d(ctx, merged, HD * H, T);

    ggml_tensor * attn_out = ggml_add(ctx, ggml_mul_mat(ctx, w.o_w, flat), w.o_b);
    x = ggml_add(ctx, residual, attn_out);

    residual = x;
    xn = ggml_norm(ctx, x, eps);
    xn = ggml_add(ctx, ggml_mul(ctx, xn, w.norm_ff_w), w.norm_ff_b);
    ggml_tensor * ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ff1_w, xn), w.ff1_b);
    ff = ggml_silu(ctx, ff);
    ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ff2_w, ff), w.ff2_b);
    return ggml_add(ctx, residual, ff);
}

static void compute_pos_emb(std::vector<float> & pe, int T, int D) {
    int L = 2 * T - 1;
    pe.assign(L * D, 0.0f);
    const float log10000 = std::log(10000.0f);
    std::vector<float> div_term(D / 2);
    for (int i = 0; i < D / 2; ++i) div_term[i] = std::exp(-((float)(2*i) * log10000 / (float)D));
    std::vector<std::vector<float>> pos_pe(T, std::vector<float>(D, 0.0f));
    std::vector<std::vector<float>> neg_pe(T, std::vector<float>(D, 0.0f));
    for (int i = 0; i < T; ++i) {
        for (int k = 0; k < D / 2; ++k) {
            pos_pe[i][2*k]     = std::sin((float)i * div_term[k]);
            pos_pe[i][2*k + 1] = std::cos((float)i * div_term[k]);
            neg_pe[i][2*k]     = std::sin(-(float)i * div_term[k]);
            neg_pe[i][2*k + 1] = std::cos(-(float)i * div_term[k]);
        }
    }
    for (int t = 0; t < T; ++t) {
        int src = T - 1 - t;
        for (int d = 0; d < D; ++d) pe[t*D + d] = pos_pe[src][d];
    }
    for (int t = 1; t < T; ++t) {
        for (int d = 0; d < D; ++d) pe[(T - 1 + t)*D + d] = neg_pe[t][d];
    }
}

// Run the full S3Gen encoder: input (T, D=512) -> mu (2T, 80)
// QVAC-17872 round-2 (§5.7): keep encoder ctx/graph/gallocr alive across
// synthesize calls to remove repeat Vulkan buffer-pool grow + first-cmd-
// setup cost in streaming/server mode.  Graph topology depends on T
// (number of input speech tokens), so cache->key tracks that and we
// rebuild on T change; the underlying gallocr buffer is grown by reserve.
static std::vector<float> run_encoder(const model_ctx & m, const std::vector<float> & input_embed, int T, int D = 512) {
    const int H = 8, HEAD_DIM = 64;
    const int T2 = 2 * T;

    auto & C = g_encoder_cache;
    const bool build = (C.key != T);
    if (build) {
        C.destroy();
        C.buf.assign(64 * 1024 * 1024, 0);  // matches the previous static buf_size
        ggml_init_params gp = { C.buf.size(), C.buf.data(), true };
        C.ctx = ggml_init(gp);
        C.gf  = ggml_new_graph_custom(C.ctx, 32768, false);
        C.key = T;
    }
    ggml_context * ctx = C.ctx;
    ggml_cgraph * gf = C.gf;
    if (!build) goto encoder_compute;
    {

    ggml_tensor * x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, T);
    ggml_set_name(x_in, "x_in"); ggml_set_input(x_in);
    ggml_tensor * pos1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 2*T - 1);
    ggml_set_name(pos1, "pos1"); ggml_set_input(pos1);
    ggml_tensor * pos2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 2*T2 - 1);
    ggml_set_name(pos2, "pos2"); ggml_set_input(pos2);

    // encoder_embed: Linear + LayerNorm + scale
    ggml_tensor * elw = find_tensor(m, "flow/encoder/embed/linear/w");
    ggml_tensor * elb = find_tensor(m, "flow/encoder/embed/linear/b");
    ggml_tensor * enw = find_tensor(m, "flow/encoder/embed/norm/w");
    ggml_tensor * enb = find_tensor(m, "flow/encoder/embed/norm/b");
    ggml_tensor * x = ggml_add(ctx, ggml_mul_mat(ctx, elw, x_in), elb);
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_add(ctx, ggml_mul(ctx, x, enw), enb);
    x = ggml_scale(ctx, x, std::sqrt((float)D));

    // pre_lookahead: 2 convs + LeakyReLU + residual
    ggml_tensor * residual = x;
    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    ggml_tensor * pw1 = find_tensor(m, "flow/encoder/pre_lookahead/conv1/w");
    ggml_tensor * pb1 = find_tensor(m, "flow/encoder/pre_lookahead/conv1/b");
    ggml_tensor * pw2 = find_tensor(m, "flow/encoder/pre_lookahead/conv2/w");
    ggml_tensor * pb2 = find_tensor(m, "flow/encoder/pre_lookahead/conv2/b");
    xt = zero_pad_dim0(ctx, xt, 0, 3);
    xt = conv1d_f32(ctx, pw1, xt, 1, 0, 1);
    xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, pb1, 1, D));
    xt = ggml_leaky_relu(ctx, xt, 0.01f, false);
    xt = zero_pad_dim0(ctx, xt, 2, 0);
    xt = conv1d_f32(ctx, pw2, xt, 1, 0, 1);
    xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, pb2, 1, D));
    // QVAC-17872 round-AUDIT: drop the trailing ggml_cont — xt is consumed
    // only as src0 of ggml_add(xt, residual) below, which handles strided
    // sources via nb[] indexing.  Saves 1 dispatch per encoder run.
    xt = ggml_permute(ctx, xt, 1, 0, 2, 3);
    x = ggml_add(ctx, xt, residual);

    // 6 conformer blocks at length T
    for (int i = 0; i < 6; ++i) {
        auto w = load_conformer(m, "flow/encoder/block" + std::to_string(i));
        x = conformer_block(ctx, w, x, pos1, D, T, H, HEAD_DIM);
    }

    // Upsample1D 2x
    ggml_tensor * up_w = find_tensor(m, "flow/encoder/up_layer/conv/w");
    ggml_tensor * up_b = find_tensor(m, "flow/encoder/up_layer/conv/b");
    ggml_tensor * xu = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    ggml_tensor * xu_3d = ggml_reshape_3d(ctx, xu, 1, xu->ne[0], xu->ne[1]);
    ggml_tensor * xu_2x = ggml_concat(ctx, xu_3d, xu_3d, 0);
    // QVAC-17872 round-AUDIT: drop outer cont; ggml_concat output is
    // contiguous and ggml_reshape_2d on contig source returns a contig
    // view, so the wrapping ggml_cont was a redundant no-op copy.
    xu = ggml_reshape_2d(ctx, xu_2x, xu_3d->ne[1]*2, xu_3d->ne[2]);
    xu = zero_pad_dim0(ctx, xu, 4, 0);
    xu = conv1d_f32(ctx, up_w, xu, 1, 0, 1);
    xu = ggml_add(ctx, xu, ggml_reshape_2d(ctx, up_b, 1, D));
    x = ggml_cont(ctx, ggml_permute(ctx, xu, 1, 0, 2, 3));

    // up_embed
    ggml_tensor * ulw = find_tensor(m, "flow/encoder/up_embed/linear/w");
    ggml_tensor * ulb = find_tensor(m, "flow/encoder/up_embed/linear/b");
    ggml_tensor * unw = find_tensor(m, "flow/encoder/up_embed/norm/w");
    ggml_tensor * unb = find_tensor(m, "flow/encoder/up_embed/norm/b");
    x = ggml_add(ctx, ggml_mul_mat(ctx, ulw, x), ulb);
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_add(ctx, ggml_mul(ctx, x, unw), unb);
    x = ggml_scale(ctx, x, std::sqrt((float)D));

    // 4 up_conformer blocks at length 2T
    for (int i = 0; i < 4; ++i) {
        auto w = load_conformer(m, "flow/encoder/up_block" + std::to_string(i));
        x = conformer_block(ctx, w, x, pos2, D, T2, H, HEAD_DIM);
    }

    // after_norm
    ggml_tensor * anw = find_tensor(m, "flow/encoder/after_norm/w");
    ggml_tensor * anb = find_tensor(m, "flow/encoder/after_norm/b");
    x = ggml_norm(ctx, x, 1e-5f);
    x = ggml_add(ctx, ggml_mul(ctx, x, anw), anb);

    // encoder_proj: Linear(D -> 80)
    ggml_tensor * epw = find_tensor(m, "flow/encoder_proj/w");
    ggml_tensor * epb = find_tensor(m, "flow/encoder_proj/b");
    ggml_tensor * mu = ggml_add(ctx, ggml_mul_mat(ctx, epw, x), epb);
    ggml_set_name(mu, "mu"); ggml_set_output(mu);
    ggml_build_forward_expand(gf, mu);

    C.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(C.allocr, gf);
    }  // end build block

encoder_compute:
    ggml_gallocr_alloc_graph(C.allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x_in"), input_embed.data(), 0, input_embed.size()*sizeof(float));

    // QVAC-17872 round-3 (§5.7-cache-hit): pos_emb is fully determined by
    // (T, D), both baked into C.key.  Compute it once into the cache on
    // first build, re-upload only on first compute (gallocr_alloc_graph
    // resets buffer ownership but tensor-set persists across subsequent
    // alloc_graph calls for an unchanged graph).
    if (!C.inputs_uploaded) {
        if (C.pe1_data.empty()) {
            compute_pos_emb(C.pe1_data, T,  D);
            compute_pos_emb(C.pe2_data, T2, D);
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos1"),
                                C.pe1_data.data(), 0, C.pe1_data.size()*sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos2"),
                                C.pe2_data.data(), 0, C.pe2_data.size()*sizeof(float));
        C.inputs_uploaded = true;
    }
    compute(m.backend, gf);

    ggml_tensor * mu_t = ggml_graph_get_tensor(gf, "mu");
    std::vector<float> mu_data(ggml_nelements(mu_t));
    ggml_backend_tensor_get(mu_t, mu_data.data(), 0, ggml_nbytes(mu_t));
    return mu_data;  // shape ggml ne=[T2, 80] = numpy (80, T2)
}

// ============================================================================
// CFM estimator (single forward) — same graph as stage_G4 of test_s3gen.cpp
// ============================================================================

struct cfm_resnet_w {
    ggml_tensor *b1_conv_w, *b1_conv_b, *b1_ln_w, *b1_ln_b;
    ggml_tensor *b2_conv_w, *b2_conv_b, *b2_ln_w, *b2_ln_b;
    ggml_tensor *mlp_w, *mlp_b, *res_w, *res_b;
};

static cfm_resnet_w load_cfm_resnet(const model_ctx & m, const std::string & pfx) {
    cfm_resnet_w w;
    w.b1_conv_w = find_tensor(m, pfx + "/block1/block/0/weight");
    w.b1_conv_b = find_tensor(m, pfx + "/block1/block/0/bias");
    w.b1_ln_w   = find_tensor(m, pfx + "/block1/block/2/weight");
    w.b1_ln_b   = find_tensor(m, pfx + "/block1/block/2/bias");
    w.b2_conv_w = find_tensor(m, pfx + "/block2/block/0/weight");
    w.b2_conv_b = find_tensor(m, pfx + "/block2/block/0/bias");
    w.b2_ln_w   = find_tensor(m, pfx + "/block2/block/2/weight");
    w.b2_ln_b   = find_tensor(m, pfx + "/block2/block/2/bias");
    w.mlp_w     = find_tensor(m, pfx + "/mlp/1/weight");
    w.mlp_b     = find_tensor(m, pfx + "/mlp/1/bias");
    w.res_w     = find_tensor(m, pfx + "/res_conv/weight");
    w.res_b     = find_tensor(m, pfx + "/res_conv/bias");
    return w;
}

static ggml_tensor * cfm_causal_block(ggml_context * ctx, ggml_tensor * x,
                                      ggml_tensor * conv_w, ggml_tensor * conv_b,
                                      ggml_tensor * ln_w, ggml_tensor * ln_b, int64_t C_out) {
    ggml_tensor * xp = zero_pad_dim0(ctx, x, 2, 0);
    ggml_tensor * y = conv1d_f32(ctx, conv_w, xp, 1, 0, 1);
    y = ggml_add(ctx, y, ggml_reshape_2d(ctx, conv_b, 1, C_out));
    y = layer_norm_on_channel(ctx, y, ln_w, ln_b);
    return ggml_mish_fn(ctx, y);
}

static ggml_tensor * cfm_resnet(ggml_context * ctx, const cfm_resnet_w & w,
                                ggml_tensor * x, ggml_tensor * t_emb, int64_t C_out) {
    ggml_tensor * h = cfm_causal_block(ctx, x, w.b1_conv_w, w.b1_conv_b, w.b1_ln_w, w.b1_ln_b, C_out);
    ggml_tensor * t_feat = ggml_mish_fn(ctx, t_emb);
    ggml_tensor * t_proj = ggml_add(ctx, ggml_mul_mat(ctx, w.mlp_w, t_feat), w.mlp_b);
    h = ggml_add(ctx, h, ggml_reshape_2d(ctx, t_proj, 1, C_out));
    h = cfm_causal_block(ctx, h, w.b2_conv_w, w.b2_conv_b, w.b2_ln_w, w.b2_ln_b, C_out);
    ggml_tensor * res = conv1d_f32(ctx, w.res_w, x, 1, 0, 1);
    res = ggml_add(ctx, res, ggml_reshape_2d(ctx, w.res_b, 1, C_out));
    return ggml_add(ctx, h, res);
}

struct basic_tfm_w {
    ggml_tensor *norm1_w, *norm1_b;
    ggml_tensor *to_q, *to_k, *to_v;
    ggml_tensor *to_out_w, *to_out_b;
    ggml_tensor *norm3_w, *norm3_b;
    ggml_tensor *ff0_w, *ff0_b, *ff2_w, *ff2_b;
};

static basic_tfm_w load_basic_tfm(const model_ctx & m, const std::string & pfx) {
    basic_tfm_w w;
    w.norm1_w = find_tensor(m, pfx + "/norm1/weight");
    w.norm1_b = find_tensor(m, pfx + "/norm1/bias");
    w.to_q = find_tensor(m, pfx + "/attn1/to_q/weight");
    w.to_k = find_tensor(m, pfx + "/attn1/to_k/weight");
    w.to_v = find_tensor(m, pfx + "/attn1/to_v/weight");
    w.to_out_w = find_tensor(m, pfx + "/attn1/to_out/0/weight");
    w.to_out_b = find_tensor(m, pfx + "/attn1/to_out/0/bias");
    w.norm3_w = find_tensor(m, pfx + "/norm3/weight");
    w.norm3_b = find_tensor(m, pfx + "/norm3/bias");
    w.ff0_w = find_tensor(m, pfx + "/ff/net/0/proj/weight");
    w.ff0_b = find_tensor(m, pfx + "/ff/net/0/proj/bias");
    w.ff2_w = find_tensor(m, pfx + "/ff/net/2/weight");
    w.ff2_b = find_tensor(m, pfx + "/ff/net/2/bias");
    return w;
}

static ggml_tensor * basic_tfm(ggml_context * ctx, const basic_tfm_w & w,
                               ggml_tensor * x, int T, int C, int H = 8, int HD = 64) {
    int INNER = H * HD;
    ggml_tensor * nx = layer_norm(ctx, x, w.norm1_w, w.norm1_b);
    // QVAC-17872 round-4 (chatterbox-side desktop-Vulkan win, default-on,
    // env-var opt-out via CHATTERBOX_FUSE_QKV=0): fuse the three separate
    // Q/K/V projections in apply_tfm_stack's mid-block transformers into
    // a single batched mul_mat.  The converter already does this for
    // down_block/up_block (the m=652 n=256 k=768 shape in the perf log
    // is fused QKV; 768 = 3·256), but leaves the inner mid_block
    // transformers as 3 separate (D, INNER) matmuls.  We measured this on
    // RTX 5090 + NVIDIA 590.48 + Vulkan 1.4.325:
    //   168 of MUL_MAT f32 m=512 n=652 k=256 (3 per tfm × 56 tfms), ~22 µs each.
    // Fusing them cuts the per-CFM-step matmul dispatch count for Q/K/V
    // from 168 to 56 while keeping the math bit-exact (verified against
    // all three round-1/2/3 MD5 invariants — see
    // bench-logs-vk-round3/r4-fuse-on-{single,multi,varied}*.log).
    //
    // Layout: ggml_mul_mat output ne=(M, N) with M innermost, so
    // mul_mat(W_q ne=(D, INNER), nx ne=(D, T)) yields ne=(INNER, T).  We
    // concat the three weights along the *batch* dim (dim=2 after a 3D
    // reshape), giving W_qkv ne=(D, INNER, 3); the batched mul_mat
    // output is ne=(INNER, T, 3) with each batch a contiguous (INNER, T)
    // — bit-identical to what the three separate matmuls would produce.
    // ggml_can_mul_mat asserts src1->ne[2] % src0->ne[2] == 0, so we
    // broadcast nx to ne[2]=3 with ggml_repeat_4d (cost: 3× tile copies
    // on the device, ~120 µs / inference, dwarfed by the saved dispatch
    // overhead).
    //
    // Each Q/K/V is then a plain ggml_view_2d into the batched output;
    // the existing reshape_3d → permute → cont chain for flash-attn is
    // unchanged.
    //
    // Measured on RTX 5090 (5 alternating opt-in/opt-out pairs, 15 warm
    // chunks each via regress-tight.sh):
    //   opt-out (default-off): 139.7 ms avg S3GEN_INFER
    //   opt-in  (default-on):  136.5 ms avg S3GEN_INFER  →  -3.2 ms / -2.3 %
    // Opt-out path provided in case a future driver / Vulkan version
    // produces different reduction orders for the batched shader.
    static const bool fuse_qkv = []() {
        const char * e = getenv("CHATTERBOX_FUSE_QKV");
        // Default on; only "0" / "false" / empty disables.
        if (!e) return true;
        const std::string s = e;
        return !(s == "0" || s == "false" || s == "FALSE" || s.empty());
    }();
    ggml_tensor * q;
    ggml_tensor * k;
    ggml_tensor * v;
    if (fuse_qkv) {
        ggml_tensor * w_q3  = ggml_reshape_3d(ctx, w.to_q, w.to_q->ne[0], w.to_q->ne[1], 1);
        ggml_tensor * w_k3  = ggml_reshape_3d(ctx, w.to_k, w.to_k->ne[0], w.to_k->ne[1], 1);
        ggml_tensor * w_v3  = ggml_reshape_3d(ctx, w.to_v, w.to_v->ne[0], w.to_v->ne[1], 1);
        ggml_tensor * w_qk  = ggml_concat(ctx, w_q3, w_k3, /*dim=*/2);
        ggml_tensor * w_qkv = ggml_concat(ctx, w_qk, w_v3, /*dim=*/2);  // ne=(D, INNER, 3)
        ggml_tensor * nx_b3 = ggml_repeat_4d(ctx, nx, nx->ne[0], nx->ne[1], 3, 1);
        ggml_tensor * qkv   = ggml_mul_mat(ctx, w_qkv, nx_b3);          // ne=(INNER, T, 3)
        const size_t batch_stride = (size_t) INNER * T * sizeof(float);
        q = ggml_view_2d(ctx, qkv, INNER, T, INNER * sizeof(float), 0);
        k = ggml_view_2d(ctx, qkv, INNER, T, INNER * sizeof(float), batch_stride);
        v = ggml_view_2d(ctx, qkv, INNER, T, INNER * sizeof(float), 2 * batch_stride);
    } else {
        q = ggml_mul_mat(ctx, w.to_q, nx);
        k = ggml_mul_mat(ctx, w.to_k, nx);
        v = ggml_mul_mat(ctx, w.to_v, nx);
    }
    // (INNER, T) -> (HD, H, T) -> (HD, T, H) contiguous for flash-attn
    q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, q, HD, H, T), 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, k, HD, H, T), 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, HD, H, T), 0, 2, 1, 3));

    // Fused softmax(QK^T / sqrt(HD)) @ V, streaming (no materialized T x T attn matrix).
    // Output layout is (HD, H, T) internally ((D, H, N) per flash_attn_ext docs).
    ggml_tensor * attn_fa = ggml_flash_attn_ext(ctx, q, k, v, /*mask=*/nullptr,
                                                /*scale=*/1.0f / std::sqrt((float)HD),
                                                /*max_bias=*/0.0f,
                                                /*logit_softcap=*/0.0f);
    // flash_attn_ext output: ne=[HD, H, T, 1] (contiguous). Reshape to (INNER, T).
    ggml_tensor * flat = ggml_reshape_2d(ctx, attn_fa, INNER, T);
    ggml_tensor * attn_out = ggml_add(ctx, ggml_mul_mat(ctx, w.to_out_w, flat), w.to_out_b);
    x = ggml_add(ctx, x, attn_out);

    ggml_tensor * nx2 = layer_norm(ctx, x, w.norm3_w, w.norm3_b);
    ggml_tensor * ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ff0_w, nx2), w.ff0_b);
    ff = ggml_gelu_erf(ctx, ff);
    ff = ggml_add(ctx, ggml_mul_mat(ctx, w.ff2_w, ff), w.ff2_b);
    return ggml_add(ctx, x, ff);
}

struct cfm_tfm_stack { std::vector<basic_tfm_w> blocks; };
static cfm_tfm_stack load_tfm_stack(const model_ctx & m, const std::string & pfx, int n) {
    cfm_tfm_stack s;
    for (int i = 0; i < n; ++i) s.blocks.push_back(load_basic_tfm(m, pfx + "/" + std::to_string(i)));
    return s;
}

static ggml_tensor * apply_tfm_stack(ggml_context * ctx, const cfm_tfm_stack & s,
                                     ggml_tensor * x, int T, int C) {
    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    for (const auto & b : s.blocks) xt = basic_tfm(ctx, b, xt, T, C);
    return ggml_cont(ctx, ggml_permute(ctx, xt, 1, 0, 2, 3));
}

static ggml_tensor * cfm_causal_k3(ggml_context * ctx, ggml_tensor * x,
                                   ggml_tensor * w, ggml_tensor * b, int C_out) {
    ggml_tensor * xp = zero_pad_dim0(ctx, x, 2, 0);
    ggml_tensor * y = conv1d_f32(ctx, w, xp, 1, 0, 1);
    return ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, C_out));
}

// Compute the time embedding for a single scalar t (or r).
// Returns (TIME_EMB_DIM=1024,) after sinusoidal + 2-layer MLP.
//
// QVAC-17872 round-2 (§5.7): the graph topology is fixed (TDIM→HIDDEN→OUT);
// across the cfm_steps loop this function is called 2× (Turbo) / 20×
// (MTL) per synthesize, each call previously paid `gallocr_new` +
// Vulkan buffer-pool grow + first-cmd-setup.  We now build the graph
// once into g_time_mlp_cache and rebind only the `x` input on subsequent
// calls.  `g_time_mlp_cache` is destroyed alongside the s3gen backend
// (see s3gen_model_cache_release / s3gen_model_cache_get cache-miss).
static std::vector<float> compute_time_mlp(const model_ctx & m, float t_val) {
    const int TDIM = 320;
    const int HIDDEN = 1280;
    const int OUT = 1024;
    std::vector<float> t_sin(TDIM);
    float log_factor = std::log(10000.0f) / (float)(TDIM/2 - 1);
    for (int i = 0; i < TDIM/2; ++i) {
        float freq = std::exp(-(float)i * log_factor);
        float arg = 1000.0f * t_val * freq;
        t_sin[i] = std::sin(arg);
        t_sin[i + TDIM/2] = std::cos(arg);
    }
    (void)HIDDEN; (void)OUT;

    auto & C = g_time_mlp_cache;
    if (!C.built) {
        C.buf.assign(4 * 1024 * 1024, 0);
        ggml_init_params gp = { C.buf.size(), C.buf.data(), true };
        C.ctx = ggml_init(gp);
        C.gf  = ggml_new_graph(C.ctx);

        ggml_tensor * x = ggml_new_tensor_1d(C.ctx, GGML_TYPE_F32, TDIM);
        ggml_set_name(x, "x"); ggml_set_input(x);
        ggml_tensor * l1w = find_tensor(m, "cfm/time_mlp/linear_1/weight");
        ggml_tensor * l1b = find_tensor(m, "cfm/time_mlp/linear_1/bias");
        ggml_tensor * l2w = find_tensor(m, "cfm/time_mlp/linear_2/weight");
        ggml_tensor * l2b = find_tensor(m, "cfm/time_mlp/linear_2/bias");
        ggml_tensor * y = ggml_add(C.ctx, ggml_mul_mat(C.ctx, l1w, x), l1b);
        y = ggml_silu(C.ctx, y);
        y = ggml_add(C.ctx, ggml_mul_mat(C.ctx, l2w, y), l2b);
        ggml_set_name(y, "out"); ggml_set_output(y);
        ggml_build_forward_expand(C.gf, y);

        C.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
        ggml_gallocr_reserve(C.allocr, C.gf);
        C.built = true;
    }
    ggml_gallocr_alloc_graph(C.allocr, C.gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(C.gf, "x"), t_sin.data(), 0, t_sin.size()*sizeof(float));
    compute(m.backend, C.gf);

    ggml_tensor * y_out = ggml_graph_get_tensor(C.gf, "out");
    std::vector<float> out(ggml_nelements(y_out));
    ggml_backend_tensor_get(y_out, out.data(), 0, ggml_nbytes(y_out));
    return out;
}

// Mix t and r embeddings via time_embed_mixer (Linear(2048 -> 1024), no bias)
//
// QVAC-17872 round-2 (§5.7): same shape on every call (TOT is always
// time_mlp's output dim = 1024).  Cached identically to compute_time_mlp
// — see g_time_mixed_cache.  The TOT == cached TOT check guards a future
// case where time_mlp's output dim changes; today it never does.
static std::vector<float> compute_time_mixed(const model_ctx & m,
                                             const std::vector<float> & t_mlp,
                                             const std::vector<float> & r_mlp) {
    auto & C = g_time_mixed_cache;
    int TOT = (int)t_mlp.size();
    if (!C.built) {
        C.buf.assign(4 * 1024 * 1024, 0);
        ggml_init_params gp = { C.buf.size(), C.buf.data(), true };
        C.ctx = ggml_init(gp);
        C.gf  = ggml_new_graph(C.ctx);

        ggml_tensor * t_in = ggml_new_tensor_1d(C.ctx, GGML_TYPE_F32, TOT);
        ggml_set_name(t_in, "t_in"); ggml_set_input(t_in);
        ggml_tensor * r_in = ggml_new_tensor_1d(C.ctx, GGML_TYPE_F32, TOT);
        ggml_set_name(r_in, "r_in"); ggml_set_input(r_in);
        ggml_tensor * cat = ggml_concat(C.ctx, t_in, r_in, 0);
        ggml_tensor * mix_w = find_tensor(m, "cfm/time_embed_mixer/weight");
        ggml_tensor * mixed = ggml_mul_mat(C.ctx, mix_w, cat);
        ggml_set_name(mixed, "out"); ggml_set_output(mixed);
        ggml_build_forward_expand(C.gf, mixed);

        C.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
        ggml_gallocr_reserve(C.allocr, C.gf);
        C.built = true;
    } else {
        // Defensive: time_mlp's output dim is fixed at 1024 today.  If a
        // future model variant changes that we'd silently produce wrong
        // output without this guard.
        ggml_tensor * t_in = ggml_graph_get_tensor(C.gf, "t_in");
        if ((int)t_in->ne[0] != TOT) {
            C.destroy();
            return compute_time_mixed(m, t_mlp, r_mlp);  // single retry, will hit !built branch
        }
    }
    ggml_gallocr_alloc_graph(C.allocr, C.gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(C.gf, "t_in"), t_mlp.data(), 0, t_mlp.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(C.gf, "r_in"), r_mlp.data(), 0, r_mlp.size()*sizeof(float));
    compute(m.backend, C.gf);

    ggml_tensor * mixed = ggml_graph_get_tensor(C.gf, "out");
    std::vector<float> out(ggml_nelements(mixed));
    ggml_backend_tensor_get(mixed, out.data(), 0, ggml_nbytes(mixed));
    return out;
}

// QVAC-17872 round-HIFT (2026-05-04): cached pipeline t_emb computation.
//
// For default cfm_steps=2, t_span = [0, 0.5, 1] yields exactly two
// (t, r) pairs per inference: (0, 0.5) and (0.5, 1).  Across many synth
// calls the same pairs repeat — the t-embedding outputs are deterministic
// functions of (t_val, r_val) and the model weights, so we can cache them.
//
// Layered caching:
//   1. compute_time_mlp output cached by t-value (intra-inference reuse:
//      compute_time_mlp(0.5) is called 2× per inference even on the first).
//   2. compute_time_mixed output cached by (t, r)-pair (cross-inference
//      reuse: every subsequent synth call hits the final-result cache and
//      pays zero GPU graph submissions for the t-emb pipeline).
//
// Theoretical wall-time saving on RTX 5090: 6 graph submissions / inference
// (cfm_steps=2) × ~50 us each ≈ 300 us / inference after warmup.  Below
// the noise floor of single-shot timings; meaningful at scale (streaming
// or batch synthesis) and on slower / mobile targets where graph-submit
// overhead dominates over compute.
//
// Bit-exactness: trivially preserved — same compute, just memoised.  All
// 13 invariants (7 RTX 5090 + 6 AMD/RADV) verified PASS.
static std::vector<float> compute_time_mlp_cached(const model_ctx & m, float t_val) {
    uint32_t key;
    static_assert(sizeof(key) == sizeof(t_val), "float must be 32-bit for bitcast key");
    std::memcpy(&key, &t_val, sizeof(key));
    {
        std::lock_guard<std::mutex> lk(g_time_emb_results_mu);
        auto it = g_time_mlp_results.find(key);
        if (it != g_time_mlp_results.end()) return it->second;
    }
    auto out = compute_time_mlp(m, t_val);
    {
        std::lock_guard<std::mutex> lk(g_time_emb_results_mu);
        g_time_mlp_results.emplace(key, out);
    }
    return out;
}

static std::vector<float> compute_time_emb_cached(const model_ctx & m, float t_val, float r_val) {
    uint32_t kt, kr;
    std::memcpy(&kt, &t_val, sizeof(kt));
    std::memcpy(&kr, &r_val, sizeof(kr));
    const uint64_t key = ((uint64_t)kt << 32) | (uint64_t)kr;
    {
        std::lock_guard<std::mutex> lk(g_time_emb_results_mu);
        auto it = g_time_emb_results.find(key);
        if (it != g_time_emb_results.end()) return it->second;
    }
    auto t_mlp = compute_time_mlp_cached(m, t_val);
    auto r_mlp = compute_time_mlp_cached(m, r_val);
    auto out = compute_time_mixed(m, t_mlp, r_mlp);
    {
        std::lock_guard<std::mutex> lk(g_time_emb_results_mu);
        g_time_emb_results.emplace(key, out);
    }
    return out;
}

// Single estimator forward: (x, mu, t_emb, spks, cond) -> dxdt
// Cache type is `cfm_estimator_cache` (defined alongside the other graph
// caches at the top of this file); the global instance is
// `g_cfm_estimator_cache`.
// All shapes are numpy (80, T) or (80,) as given, flattened row-major.
static std::vector<float> cfm_estimator_forward(
    const model_ctx & m,
    cfm_estimator_cache & cache,
    const std::vector<float> & x,
    const std::vector<float> & mu,
    const std::vector<float> & t_emb,
    const std::vector<float> & spks,
    const std::vector<float> & cond,
    int T) {
    const int MEL = 80, CH = 256, TIME_DIM = 1024;
    const int N_MID = 12, N_BLOCKS = 4;

    const bool build_graph = (cache.T != T);
    if (build_graph) {
        if (cache.allocr) { ggml_gallocr_free(cache.allocr); cache.allocr = nullptr; }
        if (cache.ctx) { ggml_free(cache.ctx); cache.ctx = nullptr; }
        cache.buf.resize(256 * 1024 * 1024);
        ggml_init_params gp = { cache.buf.size(), cache.buf.data(), true };
        cache.ctx = ggml_init(gp);
        cache.gf = ggml_new_graph_custom(cache.ctx, 65536, false);
        cache.T = T;
    }
    ggml_context * ctx = cache.ctx;
    ggml_cgraph * gf = cache.gf;
    if (!build_graph) goto compute_only;  // skip graph build, just update inputs and recompute

    {

    ggml_tensor * x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, MEL); ggml_set_name(x_in, "x_in"); ggml_set_input(x_in);
    ggml_tensor * mu_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, MEL); ggml_set_name(mu_in, "mu_in"); ggml_set_input(mu_in);
    ggml_tensor * spks_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, MEL); ggml_set_name(spks_in, "spks_in"); ggml_set_input(spks_in);
    ggml_tensor * cond_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, MEL); ggml_set_name(cond_in, "cond_in"); ggml_set_input(cond_in);
    ggml_tensor * t_emb_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, TIME_DIM); ggml_set_name(t_emb_in, "t_emb"); ggml_set_input(t_emb_in);

    ggml_tensor * spks_bc = ggml_repeat(ctx, ggml_reshape_2d(ctx, spks_in, 1, MEL), x_in);
    ggml_tensor * xc = ggml_concat(ctx, x_in, mu_in, 1);
    xc = ggml_concat(ctx, xc, spks_bc, 1);
    xc = ggml_concat(ctx, xc, cond_in, 1);

    auto down_rn = load_cfm_resnet(m, "cfm/down_blocks/0/0");
    auto down_tfms = load_tfm_stack(m, "cfm/down_blocks/0/1", N_BLOCKS);
    ggml_tensor * down_conv_w = find_tensor(m, "cfm/down_blocks/0/2/weight");
    ggml_tensor * down_conv_b = find_tensor(m, "cfm/down_blocks/0/2/bias");

    ggml_tensor * z = cfm_resnet(ctx, down_rn, xc, t_emb_in, CH);
    z = apply_tfm_stack(ctx, down_tfms, z, T, CH);
    ggml_tensor * hidden = z;
    z = cfm_causal_k3(ctx, z, down_conv_w, down_conv_b, CH);

    for (int i = 0; i < N_MID; ++i) {
        auto rn = load_cfm_resnet(m, "cfm/mid_blocks/" + std::to_string(i) + "/0");
        auto tfms = load_tfm_stack(m, "cfm/mid_blocks/" + std::to_string(i) + "/1", N_BLOCKS);
        z = cfm_resnet(ctx, rn, z, t_emb_in, CH);
        z = apply_tfm_stack(ctx, tfms, z, T, CH);
    }

    auto up_rn = load_cfm_resnet(m, "cfm/up_blocks/0/0");
    auto up_tfms = load_tfm_stack(m, "cfm/up_blocks/0/1", N_BLOCKS);
    ggml_tensor * up_conv_w = find_tensor(m, "cfm/up_blocks/0/2/weight");
    ggml_tensor * up_conv_b = find_tensor(m, "cfm/up_blocks/0/2/bias");
    z = ggml_concat(ctx, z, hidden, 1);
    z = cfm_resnet(ctx, up_rn, z, t_emb_in, CH);
    z = apply_tfm_stack(ctx, up_tfms, z, T, CH);
    z = cfm_causal_k3(ctx, z, up_conv_w, up_conv_b, CH);

    ggml_tensor * fb_conv_w = find_tensor(m, "cfm/final_block/block/0/weight");
    ggml_tensor * fb_conv_b = find_tensor(m, "cfm/final_block/block/0/bias");
    ggml_tensor * fb_ln_w   = find_tensor(m, "cfm/final_block/block/2/weight");
    ggml_tensor * fb_ln_b   = find_tensor(m, "cfm/final_block/block/2/bias");
    z = cfm_causal_block(ctx, z, fb_conv_w, fb_conv_b, fb_ln_w, fb_ln_b, CH);

    ggml_tensor * fp_w = find_tensor(m, "cfm/final_proj/weight");
    ggml_tensor * fp_b = find_tensor(m, "cfm/final_proj/bias");
    ggml_tensor * out = conv1d_f32(ctx, fp_w, z, 1, 0, 1);
    out = ggml_add(ctx, out, ggml_reshape_2d(ctx, fp_b, 1, MEL));
    ggml_set_name(out, "out"); ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    cache.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(cache.allocr, gf);
    }  // end graph-build block

    // Tried: run one dummy compute right after gallocr_reserve to pre-warm
    // Vulkan's first-dispatch state (shader residency / memory pool / command
    // pool warmup), hoping the real step0 would then run at step1 speed
    // (~12 ms instead of ~83 ms).  Outcome: the cold-compute cost is
    // per-dispatch, not per-graph-first-dispatch — adding the warmup just
    // shifted 70 ms from "hidden first-dispatch" to "explicit extra compute"
    // without reducing step0.  S3GEN_INFER went UP by ~13 ms.  Reverted.
    // The 83→12 ms gap appears to be a driver/scheduler warm-up cost on the
    // first command buffer submission that no amount of dummy work inside
    // cfm_estimator_forward removes.  Real fix would need to move the first
    // dispatch elsewhere in the pipeline (e.g. during T3→S3Gen transition)
    // so it overlaps with other host work, which is a bigger refactor.

compute_only:
    ggml_gallocr_alloc_graph(cache.allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x_in"), x.data(), 0, x.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mu_in"), mu.data(), 0, mu.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "spks_in"), spks.data(), 0, spks.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "cond_in"), cond.data(), 0, cond.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "t_emb"), t_emb.data(), 0, t_emb.size()*sizeof(float));
    compute(m.backend, gf);

    ggml_tensor * out_t = ggml_graph_get_tensor(gf, "out");
    std::vector<float> out_data(ggml_nelements(out_t));
    ggml_backend_tensor_get(out_t, out_data.data(), 0, ggml_nbytes(out_t));
    return out_data;
}

// ============================================================================
// HiFT vocoder (lifted from mel2wav.cpp)
// ============================================================================

static std::vector<float> build_hann_window(int n, bool periodic = true) {
    std::vector<float> w(n);
    double N = periodic ? (double)n : (double)(n - 1);
    const double two_pi = 2.0 * M_PI;
    for (int i = 0; i < n; ++i) w[i] = (float)(0.5 * (1.0 - std::cos(two_pi * (double)i / N)));
    return w;
}

static std::vector<float> build_stft_kernel(int n_fft, const std::vector<float> & window) {
    int F = n_fft / 2 + 1;
    std::vector<float> K((size_t)n_fft * 1 * (2 * F), 0.0f);
    const double two_pi = 2.0 * M_PI;
    for (int f = 0; f < F; ++f) {
        for (int n = 0; n < n_fft; ++n) {
            double th = two_pi * f * n / n_fft;
            float w = window[n];
            K[n + f       * n_fft] = (float)(std::cos(th) * w);
            K[n + (F + f) * n_fft] = (float)(-std::sin(th) * w);
        }
    }
    return K;
}

static std::vector<float> build_istft_kernel(int n_fft, const std::vector<float> & window) {
    int F = n_fft / 2 + 1;
    std::vector<float> K((size_t)n_fft * 1 * (2 * F), 0.0f);
    const double two_pi = 2.0 * M_PI;
    const double inv_N = 1.0 / (double)n_fft;
    for (int f = 0; f < F; ++f) {
        double coef_re = (f == 0 || f == n_fft/2) ? 1.0 : 2.0;
        double coef_im = (f == 0 || f == n_fft/2) ? 0.0 : 2.0;
        for (int n = 0; n < n_fft; ++n) {
            double th = two_pi * f * n / n_fft;
            float w = window[n];
            K[n + f       * n_fft] = (float)(coef_re * std::cos(th) * w * inv_N);
            K[n + (F + f) * n_fft] = (float)(-coef_im * std::sin(th) * w * inv_N);
        }
    }
    return K;
}

static std::vector<float> build_window_sum(int T_stft, int n_fft, int hop,
                                           const std::vector<float> & window) {
    int L = (T_stft - 1) * hop + n_fft;
    std::vector<float> ws(L, 0.0f);
    for (int t = 0; t < T_stft; ++t) {
        int base = t * hop;
        for (int n = 0; n < n_fft; ++n) ws[base + n] += window[n] * window[n];
    }
    return ws;
}

static ggml_tensor * snake(ggml_context * ctx, ggml_tensor * x,
                           ggml_tensor * alpha, ggml_tensor * inv_alpha) {
    ggml_tensor * a  = ggml_reshape_2d(ctx, alpha,     1, alpha->ne[0]);
    ggml_tensor * ia = ggml_reshape_2d(ctx, inv_alpha, 1, inv_alpha->ne[0]);
    ggml_tensor * ax = ggml_mul(ctx, x, a);
    ggml_tensor * s  = ggml_sin(ctx, ax);
    ggml_tensor * s2 = ggml_mul(ctx, s, s);
    return ggml_add(ctx, x, ggml_mul(ctx, s2, ia));
}

static std::vector<float> invert_alpha_cpu(const model_ctx & m, const std::string & name) {
    ggml_tensor * t = find_tensor(m, name);
    std::vector<float> a(ggml_nelements(t));
    ggml_backend_tensor_get(t, a.data(), 0, ggml_nbytes(t));
    std::vector<float> inv(a.size());
    for (size_t i = 0; i < a.size(); ++i) inv[i] = 1.0f / (a[i] + 1e-9f);
    return inv;
}

// F0 predictor (mel (80, T) -> f0 (T,))
//
// QVAC-17872 round-2 (§5.7): T_mel-keyed cache identical in shape to
// run_encoder; rebuild only when T_mel changes between successive
// synthesize calls.
static std::vector<float> run_f0_predictor(const model_ctx & m, const std::vector<float> & mel, int T_mel) {
    auto & C = g_f0_cache;
    const bool build = (C.key != T_mel);
    if (build) {
        C.destroy();
        C.buf.assign(8 * 1024 * 1024, 0);
        ggml_init_params gp = { C.buf.size(), C.buf.data(), true };
        C.ctx = ggml_init(gp);
        C.gf  = ggml_new_graph_custom(C.ctx, 1024, false);
        C.key = T_mel;
    }
    ggml_context * ctx = C.ctx;
    ggml_cgraph * gf = C.gf;
    if (!build) goto f0_compute;
    {
    ggml_tensor * mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_mel, 80);
    ggml_set_name(mel_in, "mel_in"); ggml_set_input(mel_in);
    ggml_tensor * x = mel_in;
    for (int i = 0; i < 5; ++i) {
        std::string pfx = "hift/f0_predictor/condnet/" + std::to_string(i * 2);
        ggml_tensor * w = find_tensor(m, pfx + "/weight");
        ggml_tensor * b = find_tensor(m, pfx + "/bias");
        int C_out = (int)w->ne[2];
        x = conv1d_f32(ctx, w, x, 1, 1, 1);
        x = ggml_add(ctx, x, ggml_reshape_2d(ctx, b, 1, C_out));
        x = ggml_unary(ctx, x, GGML_UNARY_OP_ELU);
    }
    // QVAC-17872 round-HIFT (2026-05-04): try dropping the cont before the
    // classifier matmul.  ggml_mul_mat src1 (xp here) is the activations
    // input; Vulkan's mul_mat shader iterates by stride and accepts strided
    // src1 for f32 matmul.  Saves 1 dispatch per HiFT decode.  See test
    // results in FINDINGS_ROUND_HIFT.md — verified bit-exact across all 7
    // RTX 5090 + 6 AMD/RADV invariants before shipping.
    ggml_tensor * xp = ggml_permute(ctx, x, 1, 0, 2, 3);
    ggml_tensor * cw = find_tensor(m, "hift/f0_predictor/classifier/weight");
    ggml_tensor * cb = find_tensor(m, "hift/f0_predictor/classifier/bias");
    ggml_tensor * y = ggml_mul_mat(ctx, cw, xp);
    y = ggml_add(ctx, y, cb);
    y = ggml_abs(ctx, y);
    y = ggml_reshape_1d(ctx, y, T_mel);
    ggml_set_name(y, "out"); ggml_set_output(y);
    ggml_build_forward_expand(gf, y);
    C.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(C.allocr, gf);
    }  // end build block
f0_compute:
    ggml_gallocr_alloc_graph(C.allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mel_in"), mel.data(), 0, mel.size()*sizeof(float));
    compute(m.backend, gf);
    ggml_tensor * y_t = ggml_graph_get_tensor(gf, "out");
    std::vector<float> f0(T_mel);
    ggml_backend_tensor_get(y_t, f0.data(), 0, ggml_nbytes(y_t));
    return f0;
}

// SineGen + SourceModule (CPU implementation)
static std::vector<float> sinegen_source(const std::vector<float> & f0_wav, int sr,
                                         int harmonic_num, float sine_amp, float noise_std,
                                         float voiced_threshold,
                                         const std::vector<float> & l_w, float l_b,
                                         uint32_t seed) {
    int T_wav = (int)f0_wav.size();
    int H = harmonic_num + 1;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(-(float)M_PI, (float)M_PI);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> phase_vec(H, 0.0f);
    for (int h = 1; h < H; ++h) phase_vec[h] = uniform(rng);
    std::vector<float> sine_waves((size_t)H * T_wav, 0.0f);
    std::vector<double> cum_phase(H, 0.0);
    for (int t = 0; t < T_wav; ++t) {
        float f0 = f0_wav[t];
        bool voiced = f0 > voiced_threshold;
        for (int h = 0; h < H; ++h) {
            double inc = (double)f0 * (h + 1) / (double)sr;
            cum_phase[h] += inc;
            double theta = 2.0 * M_PI * (cum_phase[h] - std::floor(cum_phase[h]));
            float sine = sine_amp * std::sin((float)theta + phase_vec[h]);
            float namp = voiced ? noise_std : sine_amp / 3.0f;
            float uv = voiced ? 1.0f : 0.0f;
            sine_waves[(size_t)h * T_wav + t] = sine * uv + namp * gauss(rng);
        }
    }
    std::vector<float> src(T_wav, 0.0f);
    for (int t = 0; t < T_wav; ++t) {
        float s = l_b;
        for (int h = 0; h < H; ++h) s += l_w[h] * sine_waves[(size_t)h * T_wav + t];
        src[t] = std::tanh(s);
    }
    return src;
}

// STFT (time-domain source -> spec)
//
// QVAC-17872 round-2 (§5.7): T_src-keyed cache.  T_src changes when the
// upstream HiFT generates a different number of samples (proportional to
// T_mel), so this is rebuilt at the same frequency as run_f0_predictor.
static std::vector<float> run_stft(const model_ctx & m, const std::vector<float> & src) {
    const int n_fft = 16, hop = 4;
    const int F = n_fft / 2 + 1;
    int T_src = (int)src.size();
    auto window = build_hann_window(n_fft, true);
    auto kernel = build_stft_kernel(n_fft, window);

    auto & C = g_stft_cache;
    const bool build = (C.key != T_src);
    if (build) {
        C.destroy();
        C.buf.assign(4 * 1024 * 1024, 0);
        ggml_init_params gp = { C.buf.size(), C.buf.data(), true };
        C.ctx = ggml_init(gp);
        C.gf  = ggml_new_graph_custom(C.ctx, 8192, false);
        C.key = T_src;
    }
    ggml_context * ctx = C.ctx;
    ggml_cgraph * gf = C.gf;
    if (!build) goto stft_compute;
    {
    ggml_tensor * s = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_src, 1);
    ggml_set_name(s, "s"); ggml_set_input(s);
    ggml_tensor * s_pad = reflect_pad_1d(ctx, s, n_fft/2, n_fft/2);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_fft, 1, 2*F);
    ggml_set_name(k, "k"); ggml_set_input(k);
    ggml_tensor * spec = conv1d_f32(ctx, k, s_pad, hop, 0, 1);
    ggml_set_name(spec, "out"); ggml_set_output(spec);
    ggml_build_forward_expand(gf, spec);
    C.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(C.allocr, gf);
    }  // end build block
stft_compute:
    ggml_gallocr_alloc_graph(C.allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "s"), src.data(), 0, src.size()*sizeof(float));
    // QVAC-17872 round-3: STFT kernel only depends on n_fft, which is a
    // compile-time constant here.  Move it into the cache, upload once.
    if (!C.inputs_uploaded) {
        if (C.kernel_data.empty()) {
            C.kernel_data = std::move(kernel);
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "k"),
                                C.kernel_data.data(), 0, C.kernel_data.size()*sizeof(float));
        C.inputs_uploaded = true;
    }
    compute(m.backend, gf);
    ggml_tensor * spec_t = ggml_graph_get_tensor(gf, "out");
    std::vector<float> out(ggml_nelements(spec_t));
    ggml_backend_tensor_get(spec_t, out.data(), 0, ggml_nbytes(spec_t));
    return out;
}

// Full HiFT decode: mel + s_stft -> wav (inlined from mel2wav.cpp)
//
// QVAC-17872 round-2 (§5.7): cache key combines T_mel and T_stft (which
// are linked: T_stft = (T_mel * up_rate_product) / hop, but encoded
// jointly here in case future model variants decouple them).  inv_alpha
// host data lives in g_hift_cache.inv_alphas alongside the graph and is
// re-uploaded on every compute call.
static std::vector<float> run_hift_decode(const model_ctx & m,
                                          const std::vector<float> & mel, int T_mel,
                                          const std::vector<float> & s_stft, int T_stft) {
    const int MEL = 80, NFFT2 = 18, BASE_CH = 512, n_fft = 16, hop = 4;
    const int F = n_fft / 2 + 1;
    std::vector<int> ups_rates  = {8, 5, 3};
    std::vector<int> ups_ksizes = {16, 11, 7};
    std::vector<int> ups_ch     = {256, 128, 64};
    std::vector<int> rb_ksizes  = {3, 7, 11};
    std::vector<std::vector<int>> rb_dils = {{1,3,5},{1,3,5},{1,3,5}};
    std::vector<int> src_rb_ksizes = {7, 7, 11};
    std::vector<std::vector<int>> src_rb_dils = {{1,3,5},{1,3,5},{1,3,5}};

    auto & C = g_hift_cache;
    // Encode (T_mel, T_stft) into a single int key by hashing.  Collisions
    // are practically impossible at the dimensions we use (T_mel ≤ ~3000,
    // T_stft fits comfortably in 24 bits).
    const int key = (T_mel * 100003) ^ T_stft;
    const bool build = (C.key != key);
    if (build) {
        C.destroy();
        C.buf.assign(64 * 1024 * 1024, 0);
        ggml_init_params gp = { C.buf.size(), C.buf.data(), true };
        C.ctx = ggml_init(gp);
        C.gf  = ggml_new_graph_custom(C.ctx, 131072, false);
        C.key = key;
    }
    ggml_context * ctx = C.ctx;
    ggml_cgraph * gf = C.gf;
    if (!build) goto hift_compute;
    {
    ggml_tensor * mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_mel, MEL);
    ggml_set_name(mel_in, "mel_in"); ggml_set_input(mel_in);
    ggml_tensor * s_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_stft, NFFT2);
    ggml_set_name(s_in, "s_in"); ggml_set_input(s_in);

    auto mk_inv = [&](const std::string & pref, int C_dim) {
        std::string gn = "inv_" + pref;
        auto inv = invert_alpha_cpu(m, pref);
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C_dim);
        ggml_set_name(t, gn.c_str()); ggml_set_input(t);
        C.inv_alphas.push_back({gn, std::move(inv)});
        return t;
    };

    auto load_rb = [&](const std::string & pref, int C) {
        struct pd { ggml_tensor *a1, *c1w, *c1b, *a2, *c2w, *c2b, *ia1, *ia2; };
        std::vector<pd> p(3);
        for (int i = 0; i < 3; ++i) {
            p[i].a1 = find_tensor(m, pref + "/activations1/" + std::to_string(i) + "/alpha");
            p[i].c1w = find_tensor(m, pref + "/convs1/" + std::to_string(i) + "/weight");
            p[i].c1b = find_tensor(m, pref + "/convs1/" + std::to_string(i) + "/bias");
            p[i].a2 = find_tensor(m, pref + "/activations2/" + std::to_string(i) + "/alpha");
            p[i].c2w = find_tensor(m, pref + "/convs2/" + std::to_string(i) + "/weight");
            p[i].c2b = find_tensor(m, pref + "/convs2/" + std::to_string(i) + "/bias");
            p[i].ia1 = mk_inv(pref + "/activations1/" + std::to_string(i) + "/alpha", C);
            p[i].ia2 = mk_inv(pref + "/activations2/" + std::to_string(i) + "/alpha", C);
        }
        return p;
    };

    auto rb_fwd = [&](auto & rb, ggml_tensor * x, int C, const std::vector<int> & dils, int ks) {
        for (int i = 0; i < 3; ++i) {
            auto & p = rb[i];
            int d = dils[i];
            int pad1 = (ks * d - d) / 2;
            int pad2 = (ks - 1) / 2;
            ggml_tensor * xt = snake(ctx, x, p.a1, p.ia1);
            xt = conv1d_f32(ctx, p.c1w, xt, 1, pad1, d);
            xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, p.c1b, 1, C));
            xt = snake(ctx, xt, p.a2, p.ia2);
            xt = conv1d_f32(ctx, p.c2w, xt, 1, pad2, 1);
            xt = ggml_add(ctx, xt, ggml_reshape_2d(ctx, p.c2b, 1, C));
            x = ggml_add(ctx, x, xt);
        }
        return x;
    };

    ggml_tensor * cpw = find_tensor(m, "hift/conv_pre/weight");
    ggml_tensor * cpb = find_tensor(m, "hift/conv_pre/bias");
    ggml_tensor * x = conv1d_f32(ctx, cpw, mel_in, 1, 3, 1);
    x = ggml_add(ctx, x, ggml_reshape_2d(ctx, cpb, 1, BASE_CH));

    for (int i = 0; i < 3; ++i) {
        x = ggml_leaky_relu(ctx, x, 0.1f, false);
        ggml_tensor * uw = find_tensor(m, "hift/ups/" + std::to_string(i) + "/weight");
        ggml_tensor * ub = find_tensor(m, "hift/ups/" + std::to_string(i) + "/bias");
        int up_pad = (ups_ksizes[i] - ups_rates[i]) / 2;
        x = conv_transpose_1d_f32(ctx, uw, x, ups_rates[i], up_pad);
        x = ggml_add(ctx, x, ggml_reshape_2d(ctx, ub, 1, ups_ch[i]));
        if (i == 2) {
            ggml_tensor * xs = ggml_view_3d(ctx, x, 1, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 1 * x->nb[0]);
            xs = ggml_cont(ctx, xs);
            x = ggml_concat(ctx, xs, x, 0);
        }
        ggml_tensor * sw = find_tensor(m, "hift/source_downs/" + std::to_string(i) + "/weight");
        ggml_tensor * sb = find_tensor(m, "hift/source_downs/" + std::to_string(i) + "/bias");
        int sd_stride = (i == 0) ? 15 : (i == 1) ? 3 : 1;
        int sd_pad    = (i == 0) ? 7  : (i == 1) ? 1 : 0;
        ggml_tensor * si = conv1d_f32(ctx, sw, s_in, sd_stride, sd_pad, 1);
        si = ggml_add(ctx, si, ggml_reshape_2d(ctx, sb, 1, (int)sw->ne[2]));
        auto srb = load_rb("hift/source_resblocks/" + std::to_string(i), ups_ch[i]);
        si = rb_fwd(srb, si, ups_ch[i], src_rb_dils[i], src_rb_ksizes[i]);
        x = ggml_add(ctx, x, si);

        ggml_tensor * xs = nullptr;
        for (int j = 0; j < 3; ++j) {
            auto rb = load_rb("hift/resblocks/" + std::to_string(i * 3 + j), ups_ch[i]);
            ggml_tensor * rb_out = rb_fwd(rb, x, ups_ch[i], rb_dils[j], rb_ksizes[j]);
            xs = (xs == nullptr) ? rb_out : ggml_add(ctx, xs, rb_out);
        }
        x = ggml_scale(ctx, xs, 1.0f / 3.0f);
    }

    x = ggml_leaky_relu(ctx, x, 0.01f, false);
    ggml_tensor * cp2w = find_tensor(m, "hift/conv_post/weight");
    ggml_tensor * cp2b = find_tensor(m, "hift/conv_post/bias");
    x = conv1d_f32(ctx, cp2w, x, 1, 3, 1);
    x = ggml_add(ctx, x, ggml_reshape_2d(ctx, cp2b, 1, NFFT2));

    // ISTFT
    // QVAC-17872 round-AUDIT (negative result): tried dropping the leading
    // conts on mag_log and ph_in.  Single-shot passed bit-exact but
    // multi-synth multi-chunk produced a different PCM MD5 (the F-stride
    // view at non-zero offset apparently interacts with the cache-rebuild
    // path in a way that diverges from the F32 reference for some T_stft
    // values).  Conts are required.
    size_t col_stride = x->nb[1];
    ggml_tensor * mag_log = ggml_cont(ctx, ggml_view_2d(ctx, x, T_stft, F, col_stride, 0));
    mag_log = ggml_clamp(ctx, mag_log, -1e6f, 1e2f);
    ggml_tensor * mag = ggml_exp(ctx, mag_log);
    ggml_tensor * ph_in = ggml_cont(ctx, ggml_view_2d(ctx, x, T_stft, F, col_stride, (size_t)F * col_stride));
    ggml_tensor * ph = ggml_sin(ctx, ph_in);
    ggml_tensor * real = ggml_mul(ctx, mag, ggml_cos(ctx, ph));
    ggml_tensor * imag = ggml_mul(ctx, mag, ggml_sin(ctx, ph));
    ggml_tensor * spec = ggml_concat(ctx, real, imag, 1);

    auto window = build_hann_window(n_fft, true);
    auto ik = build_istft_kernel(n_fft, window);
    auto ws = build_window_sum(T_stft, n_fft, hop, window);

    ggml_tensor * istft_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_fft, 1, 2 * F);
    ggml_set_name(istft_k, "istft_k"); ggml_set_input(istft_k);
    ggml_tensor * ws_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int)ws.size(), 1);
    ggml_set_name(ws_in, "w_sum"); ggml_set_input(ws_in);

    ggml_tensor * y = ggml_conv_transpose_1d(ctx, istft_k, spec, hop, 0, 1);
    y = ggml_div(ctx, y, ws_in);
    int pad_amt = n_fft / 2;
    int L_wav = (int)ws.size() - n_fft;
    // QVAC-17872 round-HIFT (2026-05-04): drop the trailing ggml_cont.  The
    // view's only consumer is ggml_clamp (element-wise, accepts strided
    // src0); clamp's output is a fresh contiguous tensor allocated by the
    // gallocator.  ggml_set_output is set on that contig output, so
    // tensor_get reads from a contig buffer.  Saves 1 dispatch per HiFT
    // decode.
    ggml_tensor * y_trim = ggml_view_2d(ctx, y, L_wav, y->ne[1], y->nb[1],
                                        (size_t)pad_amt * y->nb[0]);
    y_trim = ggml_clamp(ctx, y_trim, -0.99f, 0.99f);
    ggml_set_name(y_trim, "wav"); ggml_set_output(y_trim);
    ggml_build_forward_expand(gf, y_trim);

    C.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_gallocr_reserve(C.allocr, gf);
    }  // end build block
hift_compute:
    {
    ggml_gallocr_alloc_graph(C.allocr, gf);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mel_in"), mel.data(), 0, mel.size()*sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "s_in"), s_stft.data(), 0, s_stft.size()*sizeof(float));

    // QVAC-17872 round-3 (§5.7-cache-hit): istft_k / w_sum / inv_alphas are
    // all derived from constants (n_fft, hop, T_stft, model weights) — they
    // don't depend on per-call inputs.  Compute once into the cache on
    // first build, re-upload to GPU once on the first compute call after
    // each rebuild, then trust the gallocr to keep the offsets stable.
    if (!C.inputs_uploaded) {
        if (C.istft_k_data.empty()) {
            auto window = build_hann_window(n_fft, true);
            C.istft_k_data = build_istft_kernel(n_fft, window);
            C.w_sum_data   = build_window_sum(T_stft, n_fft, hop, window);
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "istft_k"),
                                C.istft_k_data.data(), 0, C.istft_k_data.size()*sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "w_sum"),
                                C.w_sum_data.data(), 0, C.w_sum_data.size()*sizeof(float));
        for (auto & ia : C.inv_alphas)
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, ia.first.c_str()),
                                    ia.second.data(), 0, ia.second.size()*sizeof(float));
        C.inputs_uploaded = true;
    }
    compute(m.backend, gf);

    ggml_tensor * y_trim_t = ggml_graph_get_tensor(gf, "wav");
    std::vector<float> wav(ggml_nelements(y_trim_t));
    ggml_backend_tensor_get(y_trim_t, wav.data(), 0, ggml_nbytes(y_trim_t));
    return wav;
    }
}

// ============================================================================
// WAV writer
// ============================================================================

static void write_wav(const std::string & path, const std::vector<float> & wav, int sr) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot open " + path);
    uint32_t n = (uint32_t)wav.size();
    uint32_t byte_rate = sr * 2;
    uint32_t data_size = n * 2;
    uint32_t chunk_size = 36 + data_size;
    std::fwrite("RIFF", 1, 4, f); std::fwrite(&chunk_size, 4, 1, f);
    std::fwrite("WAVE", 1, 4, f); std::fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16; uint16_t fmt = 1, ch = 1, align = 2, bps = 16;
    uint32_t sr_u = (uint32_t)sr;
    std::fwrite(&fmt_size, 4, 1, f); std::fwrite(&fmt, 2, 1, f); std::fwrite(&ch, 2, 1, f);
    std::fwrite(&sr_u, 4, 1, f); std::fwrite(&byte_rate, 4, 1, f);
    std::fwrite(&align, 2, 1, f); std::fwrite(&bps, 2, 1, f);
    std::fwrite("data", 1, 4, f); std::fwrite(&data_size, 4, 1, f);
    for (float x : wav) {
        float c = std::max(-1.0f, std::min(1.0f, x));
        int16_t v = (int16_t)std::lrintf(c * 32767.0f);
        std::fwrite(&v, 2, 1, f);
    }
    std::fclose(f);
}

// ============================================================================
// Token file loader (reads "1,2,3" or newline-separated ints)
// ============================================================================

static std::vector<int32_t> read_tokens_file(const std::string & path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<int32_t> out;
    std::string token;
    while (std::getline(f, token, ',')) {
        try { out.push_back(std::stoi(token)); } catch (...) {
            // maybe newline-separated, try stoi after trimming
            while (!token.empty() && (token.back() == '\n' || token.back() == ' ' || token.back() == '\r')) token.pop_back();
            if (!token.empty()) out.push_back(std::stoi(token));
        }
    }
    return out;
}

// ============================================================================
// Public entry point — takes pre-generated T3 speech tokens + a voice source
// (either a --ref-dir with .npy files or the built-in voice baked into the
// s3gen GGUF) and writes a 24 kHz wav.
// ============================================================================

#include "tts-cpp/chatterbox/s3gen_pipeline.h"

int s3gen_synthesize_to_wav(
    const std::vector<int32_t> & speech_tokens,
    const s3gen_synthesize_opts & opts)
{
    const std::string & gguf_path = opts.s3gen_gguf_path;
    const std::string & ref_dir   = opts.ref_dir;
    const std::string & out_path  = opts.out_wav_path;
    const int  seed       = opts.seed;
    const int  sr         = opts.sr;
    const bool debug_mode = opts.debug;
    const bool verbose    = opts.verbose;
    const int  pre_lookahead_len = 3;  // Chatterbox default

    // Verbose progress / per-stage timing goes through this helper so all of
    // it can be disabled with `--verbose` unset.  Errors and machine-parseable
    // BENCH: lines stay unconditional below.
    auto vlog = [&](const char * fmt, auto... args) {
        if (!verbose) return;
        // `fmt` is always a string literal at call sites but the compiler
        // can't prove that through the variadic lambda.  Android NDK's
        // default `-Werror=format-security` (together with `_FORTIFY_SOURCE=2`)
        // then refuses to build unless we silence the warning locally.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
        fprintf(stderr, fmt, args...);
#pragma GCC diagnostic pop
    };

    int n_threads = opts.n_threads;
    if (n_threads <= 0) n_threads = (int)std::max(1u, std::thread::hardware_concurrency());
    g_n_threads = n_threads;
    vlog("Using %d threads\n", g_n_threads);
    if (gguf_path.empty()) {
        fprintf(stderr, "s3gen_synthesize_to_wav: s3gen_gguf_path is required\n");
        return 1;
    }
    if (out_path.empty() && opts.pcm_out == nullptr) {
        fprintf(stderr, "s3gen_synthesize_to_wav: at least one of out_wav_path or pcm_out must be set\n");
        return 1;
    }
    if (debug_mode && ref_dir.empty()) {
        fprintf(stderr, "--debug requires --ref-dir (Python-dumped intermediate tensors)\n");
        return 1;
    }
    if (speech_tokens.empty()) {
        fprintf(stderr, "s3gen_synthesize_to_wav: speech_tokens is empty\n");
        return 1;
    }

    // Reference conditioning: prefer GGUF-embedded built-in voice;
    // fall back to .npy files if --ref-dir was provided.
    // Layout: emb (192,) float32; pt (N,) int32; pf (mel_len1, 80) float32.
    std::vector<float>   emb_data;
    std::vector<int32_t> pt_data;
    std::vector<float>   pf_data;
    int pf_rows = 0;  // mel_len1

    if (ref_dir.empty() && opts.embedding_override.empty() && opts.prompt_feat_override.empty()
        && opts.prompt_token_override.empty()) {
        vlog("No --ref-dir given; loading built-in voice from GGUF.\n");
    } else {
        if (!ref_dir.empty()) vlog("Loading ref dict from %s\n", ref_dir.c_str());

        if (!opts.prompt_token_override.empty()) {
            pt_data = opts.prompt_token_override;
            vlog("  prompt_token: using C++ override (S3TokenizerV2, %zu tokens)\n",
                    pt_data.size());
        } else {
            npy_array pt_npy = npy_load(ref_dir + "/prompt_token.npy");
            pt_data.assign((const int32_t*)pt_npy.data.data(),
                           (const int32_t*)pt_npy.data.data() + pt_npy.n_elements());
        }

        if (!opts.embedding_override.empty()) {
            emb_data = opts.embedding_override;
            vlog("  embedding: using C++ override (CAMPPlus, %zu dims)\n", emb_data.size());
        } else {
            npy_array emb_npy = npy_load(ref_dir + "/embedding.npy");
            emb_data.assign((const float*)emb_npy.data.data(),
                            (const float*)emb_npy.data.data() + emb_npy.n_elements());
        }

        if (!opts.prompt_feat_override.empty()) {
            pf_data = opts.prompt_feat_override;
            pf_rows = opts.prompt_feat_rows_override;
            vlog("  prompt_feat: using C++ override (%d mel frames)\n", pf_rows);
        } else {
            npy_array pf_npy = npy_load(ref_dir + "/prompt_feat.npy");
            pf_data.assign((const float*)pf_npy.data.data(),
                           (const float*)pf_npy.data.data() + pf_npy.n_elements());
            pf_rows = (int)pf_npy.shape[0];
        }
    }

    vlog("Speech tokens: %zu\n", speech_tokens.size());

    // Trim tokens >= vocab_size.  In batch / last-chunk (`finalize=true`) we
    // append 3 S3GEN_SIL lookahead tokens to give the encoder right-context
    // for the true ending.  For streaming chunks (`finalize=false`) we skip
    // that: the lookahead will come from real speech tokens in the next
    // chunk, and we'll trim the 6 mel frames corresponding to the pre-
    // lookahead window right after CFM.
    const int32_t S3GEN_SIL = tts_cpp::chatterbox::kS3GenSilenceToken;
    const int32_t VOCAB_SIZE = 6561;
    std::vector<int32_t> padded;
    for (int32_t t : speech_tokens) {
        if (t >= 0 && t < VOCAB_SIZE) padded.push_back(t);
    }
    if (opts.append_lookahead_silence) {
        for (int i = 0; i < pre_lookahead_len; ++i) padded.push_back(S3GEN_SIL);
    }

    // Cache the loaded model across invocations so the streaming driver
    // pays the ~700 ms GGUF-load cost only once.  Keyed on (path,
    // n_gpu_layers) so switching backends still works.  Verbose gates the
    // banner prints but the BENCH line always goes out for perf checks.
    model_ctx & m = *s3gen_model_cache_get(gguf_path, opts.n_gpu_layers, verbose);
    {
        const double load_ms = s3gen_model_cache_last_load_ms();
        // Only emit the BENCH line on an actual GGUF load — on cache hits
        // the value is always 0 and repeating it per chunk adds noise.
        if (load_ms > 0.0 || verbose) {
            fprintf(stderr, "BENCH: S3GEN_LOAD_MS=%.0f\n", load_ms);
        }
    }

    // HiFT-side graphs (f0_predictor, STFT, hift_decode) used to need a
    // dedicated CPU copy of the S3Gen GGUF on Metal because conv_transpose_1d,
    // pad_ext and diag_mask_inf were either missing or pathologically slow in
    // ggml-metal.  After the kernel fixes in ggml/src/ggml-metal/, HiFT runs
    // on the main backend directly on every platform.
    const model_ctx & m_hift = m;

    // If neither --ref-dir nor any C++ override populated the three
    // conditioning tensors above, pull the built-in voice from the GGUF.
    //
    // NB: `ref_dir.empty()` alone is NOT a valid check here — --reference-audio
    // by itself (no --ref-dir) legitimately fills all three via the C++
    // override path, and we must not overwrite those.
    if (pt_data.empty() && emb_data.empty() && pf_data.empty()) {
        ggml_tensor * t_emb = find_tensor(m, "s3gen/builtin/embedding");
        ggml_tensor * t_pt  = find_tensor(m, "s3gen/builtin/prompt_token");
        ggml_tensor * t_pf  = find_tensor(m, "s3gen/builtin/prompt_feat");
        emb_data.resize(ggml_nelements(t_emb));
        pt_data.resize(ggml_nelements(t_pt));
        pf_data.resize(ggml_nelements(t_pf));
        ggml_backend_tensor_get(t_emb, emb_data.data(), 0, ggml_nbytes(t_emb));
        ggml_backend_tensor_get(t_pt,  pt_data.data(),  0, ggml_nbytes(t_pt));
        ggml_backend_tensor_get(t_pf,  pf_data.data(),  0, ggml_nbytes(t_pf));
        // prompt_feat is stored ggml ne=[80, 500] = numpy (500, 80).
        pf_rows = (int)t_pf->ne[1];
        vlog("  built-in voice: embedding=(%zu,) prompt_token=(%zu,) prompt_feat=(%d, %lld)\n",
                emb_data.size(), pt_data.size(), pf_rows, (long long)t_pf->ne[0]);
    }
    double pipeline_t0 = now_ms();

    const int D = 512;
    const int MEL = 80;

    // 1) Concat prompt_token + padded speech_tokens
    int n_prompt = (int)pt_data.size();
    int n_total = n_prompt + (int)padded.size();
    vlog("n_prompt=%d n_speech_padded=%zu n_total=%d\n", n_prompt, padded.size(), n_total);

    std::vector<int32_t> flow_tokens(n_total);
    std::memcpy(flow_tokens.data(), pt_data.data(), n_prompt * sizeof(int32_t));
    std::memcpy(flow_tokens.data() + n_prompt, padded.data(), padded.size() * sizeof(int32_t));

    // 2) input_embedding lookup + multiply by mask
    vlog("Running input_embedding...\n");
    ggml_tensor * emb_w = find_tensor(m, "flow/input_embedding");
    // QVAC-17872 round-HIFT: input_embedding weight is 13.4 MB on the
    // Turbo build (D=512 × vocab=6561 × 4 B).  Each synth call previously
    // paid the full GPU→CPU download (~600 us wall on RTX 5090).  Cache
    // the CPU mirror so subsequent calls only pay the cheap row-copy
    // lookup cost.  Cache is bound to the s3gen model lifecycle.
    const float * emb_w_data = cached_cpu_weights_f32(emb_w);
    vlog("  emb_w ne=[%lld, %lld]\n", (long long)emb_w->ne[0], (long long)emb_w->ne[1]);
    int vocab_size = (int)emb_w->ne[1];
    std::vector<float> input_embed(n_total * D);
    for (int i = 0; i < n_total; ++i) {
        int32_t tok = flow_tokens[i];
        if (tok < 0) tok = 0;
        if (tok >= vocab_size) {
            fprintf(stderr, "warning: token %d out of range (vocab=%d), clamping\n", tok, vocab_size);
            tok = vocab_size - 1;
        }
        std::memcpy(input_embed.data() + i * D, emb_w_data + (size_t)tok * D, D * sizeof(float));
    }
    if (debug_mode) {
        fprintf(stderr, "  token[0]=%d lookup: %.6f %.6f %.6f %.6f %.6f\n",
                flow_tokens[0],
                input_embed[0], input_embed[1], input_embed[2], input_embed[3], input_embed[4]);
    }

    // 3) Run encoder -> mu_T (numpy (T_mu, 80) layout, to match encoder_proj.npy)
    vlog("Running encoder (T=%d)...\n", n_total);
    double encoder_t0 = now_ms();
    std::vector<float> mu_T = run_encoder(m, input_embed, n_total, D);
    vlog("  [encoder] %.1f ms\n", now_ms() - encoder_t0);
    int T_mu = 2 * n_total;
    vlog("  encoder output: (%d, 80) = %zu floats\n", T_mu, mu_T.size());

    // Streaming: trim the last `pre_lookahead_len * token_mel_ratio = 6`
    // frames off the encoder output BEFORE CFM (matches Python's
    // flow.inference(finalize=False) which does `h = h[:, :-6]` pre-
    // decoder).  Doing it here — not post-CFM — keeps mu / cond / z and
    // CFM's internal attention all sized consistently with Python, so
    // a Python-dumped noise tensor produces bit-exact mel output in C++.
    const int TOKEN_MEL_RATIO_PRE = 2;  // each speech token expands to 2 mels
    if (!opts.finalize) {
        const int trim = pre_lookahead_len * TOKEN_MEL_RATIO_PRE;  // 6
        if (T_mu <= trim) {
            fprintf(stderr, "error: streaming chunk too short (T_mu=%d ≤ trim=%d)\n", T_mu, trim);
            return 1;
        }
        T_mu -= trim;
        mu_T.resize((size_t)T_mu * MEL);  // numpy layout (T_mu, 80): drop last `trim` rows
        vlog("  streaming: trimmed %d mel frames -> T_mu=%d\n", trim, T_mu);
    }

    if (debug_mode) {
        npy_array ref_ie = npy_load(ref_dir + "/input_embedded.npy");
        const float * ref = (const float*)ref_ie.data.data();
        size_t n = std::min(input_embed.size(), ref_ie.n_elements());
        float ma = 0, rsum = 0;
        for (size_t i = 0; i < n; ++i) {
            float d = input_embed[i] - ref[i];
            ma = std::max(ma, std::fabs(d));
            rsum += d * d;
        }
        fprintf(stderr, "  [input_embed] max_abs=%.4e rms=%.4e vs ref\n", ma, std::sqrt(rsum / n));

        npy_array ref_mu = npy_load(ref_dir + "/encoder_proj.npy");
        const float * mref = (const float*)ref_mu.data.data();
        size_t mn = std::min(mu_T.size(), ref_mu.n_elements());
        ma = 0; rsum = 0;
        for (size_t i = 0; i < mn; ++i) {
            float d = mu_T[i] - mref[i];
            ma = std::max(ma, std::fabs(d));
            rsum += d * d;
        }
        fprintf(stderr, "  [mu_T (before transpose)] max_abs=%.4e rms=%.4e vs encoder_proj.npy\n", ma, std::sqrt(rsum / mn));
    }

    // Transpose mu_T from numpy (T_mu, 80) layout to numpy (80, T_mu) for CFM.
    // In memory: mu_T has [t0_m0, t0_m1, ..., t0_m79, t1_m0, ...]
    //            mu should be [m0_t0, m0_t1, ..., m0_tTmu-1, m1_t0, ...]
    std::vector<float> mu(T_mu * MEL);
    for (int m2 = 0; m2 < MEL; ++m2)
        for (int t = 0; t < T_mu; ++t)
            mu[m2 * T_mu + t] = mu_T[t * MEL + m2];

    // Streaming debug: optionally dump the encoder_proj output (same as
    // Python's `mu` tensor fed into flow_matching.forward) so the test
    // harness can isolate encoder-side vs CFM-side divergence.
    if (!opts.dump_mel_path.empty()) {
        std::string base = opts.dump_mel_path;
        if (base.size() > 4 && base.substr(base.size() - 4) == ".npy")
            base.resize(base.size() - 4);
        npy_save_f32(base + "_mu.npy", {(int64_t)T_mu, MEL}, mu_T.data());
    }

    // 4) Speaker embedding: F.normalize + spk_embed_affine_layer
    vlog("Computing speaker embedding...\n");
    const float * emb_raw = emb_data.data();
    float norm = 0.0f;
    for (int i = 0; i < 192; ++i) norm += emb_raw[i] * emb_raw[i];
    norm = std::sqrt(norm + 1e-12f);
    std::vector<float> emb_norm(192);
    for (int i = 0; i < 192; ++i) emb_norm[i] = emb_raw[i] / norm;

    ggml_tensor * saw = find_tensor(m, "flow/spk_embed_affine/w");  // (80, 192) numpy -> ne=[192, 80]
    ggml_tensor * sab = find_tensor(m, "flow/spk_embed_affine/b");  // (80,)
    // QVAC-17872 round-HIFT: cache CPU mirrors of the speaker-affine
    // weights (~60 KB) instead of paying GPU→CPU download per synth.
    const float * saw_data = cached_cpu_weights_f32(saw);
    const float * sab_data = cached_cpu_weights_f32(sab);
    std::vector<float> spks(MEL, 0.0f);
    for (int o = 0; o < MEL; ++o) {
        float acc = sab_data[o];
        for (int i = 0; i < 192; ++i) acc += saw_data[o * 192 + i] * emb_norm[i];
        spks[o] = acc;
    }

    // 5) Build cond: zeros(T_mu, 80), fill first mel_len1 rows with prompt_feat
    int mel_len1 = pf_rows;
    if (mel_len1 > T_mu) {
        fprintf(stderr, "error: mel_len1=%d > T_mu=%d\n", mel_len1, T_mu);
        return 1;
    }
    std::vector<float> cond(T_mu * MEL, 0.0f);
    // pf is (mel_len1, 80) numpy = ne=[80, mel_len1] in ggml. We want cond ne=[T_mu, MEL].
    // In memory ggml ne=[T_mu, MEL] means [t0_m0, t1_m0, ..., t_Tmu-1_m0, t0_m1, ...].
    // So cond[m, t] = pf[t, m].
    const float * pf_raw = pf_data.data();
    for (int m2 = 0; m2 < MEL; ++m2)
        for (int t = 0; t < mel_len1; ++t)
            cond[m2 * T_mu + t] = pf_raw[t * MEL + m2];

    // Streaming debug: dump spks + cond so we can compare against Python's
    // flow.decoder input tensors chunk-by-chunk.
    if (!opts.dump_mel_path.empty()) {
        std::string base = opts.dump_mel_path;
        if (base.size() > 4 && base.substr(base.size() - 4) == ".npy")
            base.resize(base.size() - 4);
        npy_save_f32(base + "_spks.npy", {MEL}, spks.data());
        // C++ stores cond in ggml ne=[T_mu, MEL] layout (T_mu innermost) which
        // Python sees as numpy shape (MEL, T_mu).  Dump in that layout so we
        // can diff directly against Python's (80, T_mu) decoder-input cond.
        npy_save_f32(base + "_cond.npy", {MEL, (int64_t)T_mu}, cond.data());
        fprintf(stderr, "  [stream] dumped spks (%d,) cond (%d, %d) → %s_{spks,cond}.npy\n",
                MEL, MEL, T_mu, base.c_str());
    }

    if (debug_mode) {
        npy_array ref = npy_load(ref_dir + "/cfm_step0_cond.npy");
        const float * r = (const float*)ref.data.data();
        size_t n = std::min(cond.size(), ref.n_elements());
        float ma = 0, rsum = 0;
        for (size_t i = 0; i < n; ++i) { float d = cond[i] - r[i]; ma = std::max(ma, std::fabs(d)); rsum += d*d; }
        fprintf(stderr, "  [cond] max_abs=%.4e rms=%.4e vs ref\n", ma, std::sqrt(rsum / n));

        npy_array ref_s = npy_load(ref_dir + "/cfm_step0_spks.npy");
        const float * rs = (const float*)ref_s.data.data();
        size_t ns = std::min(spks.size(), ref_s.n_elements());
        ma = 0; rsum = 0;
        for (size_t i = 0; i < ns; ++i) { float d = spks[i] - rs[i]; ma = std::max(ma, std::fabs(d)); rsum += d*d; }
        fprintf(stderr, "  [spks] max_abs=%.4e rms=%.4e vs ref\n", ma, std::sqrt(rsum / ns));

        // Also compare mu vs cfm_step0_mu.npy (which should equal encoder_proj)
        npy_array ref_mu = npy_load(ref_dir + "/cfm_step0_mu.npy");
        const float * rm = (const float*)ref_mu.data.data();
        size_t nm = std::min(mu.size(), ref_mu.n_elements());
        ma = 0; rsum = 0;
        for (size_t i = 0; i < nm; ++i) { float d = mu[i] - rm[i]; ma = std::max(ma, std::fabs(d)); rsum += d*d; }
        fprintf(stderr, "  [mu vs step0_mu] max_abs=%.4e rms=%.4e vs ref\n", ma, std::sqrt(rsum / nm));
    }

    // 6) For meanflow: z = randn(80, T_mu); then z[:, prompt_len:] = noised_mels (randn(80, T_speech*2))
    vlog("Initializing CFM noise (seed=%d, meanflow)...\n", seed);
    std::vector<float> z(T_mu * MEL);
    int n_speech_part = 2 * (int)padded.size();
    int prompt_len_in_mu = T_mu - n_speech_part;
    vlog("  T_mu=%d prompt_len_in_mu=%d n_speech_part=%d\n",
            T_mu, prompt_len_in_mu, n_speech_part);
    if (!opts.cfm_z0_override.empty()) {
        // Streaming validation path: use the exact noise Python used for
        // this chunk so we can get bit-exact mel parity.  This override must
        // be the FINAL z passed to the CFM estimator (i.e. including the
        // speech-region overwrite that flow_matching.forward applies when
        // `noised_mels` is not None in meanflow mode).
        if ((int64_t)opts.cfm_z0_override.size() != (int64_t)T_mu * MEL) {
            fprintf(stderr, "error: cfm_z0_override has %zu elements but T_mu*MEL=%d\n",
                    opts.cfm_z0_override.size(), T_mu * MEL);
            return 1;
        }
        std::memcpy(z.data(), opts.cfm_z0_override.data(), z.size() * sizeof(float));
        fprintf(stderr, "  [stream] loaded %zu elems of CFM z0 from cfm_z0_override\n",
                opts.cfm_z0_override.size());
    } else if (debug_mode) {
        npy_array z_npy = npy_load(ref_dir + "/cfm_z0_raw.npy");
        std::memcpy(z.data(), z_npy.data.data(), z.size() * sizeof(float));
        fprintf(stderr, "  [debug] loaded z from cfm_z0_raw.npy\n");
        // Overwrite z[:, prompt_len:] with noised_mels
        npy_array nm_npy = npy_load(ref_dir + "/cfm_noised_mels.npy");
        const float * nm = (const float*)nm_npy.data.data();
        int nm_T = (int)nm_npy.shape[1];
        fprintf(stderr, "  [debug] overlay noised_mels (80, %d) at pos %d\n", nm_T, prompt_len_in_mu);
        // z ne=[T_mu, MEL]; write rows [prompt_len_in_mu .. prompt_len_in_mu+nm_T) of each channel
        for (int m2 = 0; m2 < MEL; ++m2)
            for (int t = 0; t < nm_T; ++t)
                z[m2 * T_mu + (prompt_len_in_mu + t)] = nm[m2 * nm_T + t];
    } else {
        std::mt19937 rng(seed);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        for (size_t i = 0; i < z.size(); ++i) z[i] = gauss(rng);
        // For meanflow production, also resample the non-prompt region independently:
        // (Python: noise = torch.randn(1, 80, speech_tokens.size(-1)*2); z[..., prompt_len:] = noise)
        std::mt19937 rng2(seed + 2);
        std::normal_distribution<float> gauss2(0.0f, 1.0f);
        for (int m2 = 0; m2 < MEL; ++m2)
            for (int t = prompt_len_in_mu; t < T_mu; ++t)
                z[m2 * T_mu + t] = gauss2(rng2);
    }

    // 7) CFM loop: default 2-step meanflow (t_span = [0, 0.5, 1]); streaming
    // callers can override via opts.cfm_steps = 1 for a single Euler jump
    // (~2× CFM speedup at the cost of a small quality degradation — Turbo
    // is meanflow-trained so 1-step is a valid sampling mode, just noisier).
    std::vector<float> t_span;
    const int cfm_steps = opts.cfm_steps > 0 ? opts.cfm_steps : 2;
    t_span.reserve(cfm_steps + 1);
    for (int i = 0; i <= cfm_steps; ++i)
        t_span.push_back((float)i / (float)cfm_steps);
    // QVAC-17872 round-HIFT: persistent CFM estimator graph cache (was
    // local-scope before).  Re-used across synth calls when T matches —
    // multi-synth chunks 2..N skip the 256 MB buf alloc + graph build +
    // gallocr_reserve cost they previously paid every chunk.  Lifetime
    // managed by s3gen_model_cache_release / cache-miss.
    cfm_estimator_cache & cfm_cache = g_cfm_estimator_cache;
    double cfm_t0 = now_ms();
    for (size_t s = 0; s < t_span.size() - 1; ++s) {
        float t = t_span[s], r = t_span[s + 1];
        float dt = r - t;
        vlog("CFM step %zu: t=%g r=%g dt=%g...\n", s, t, r, dt);
        // QVAC-17872 round-HIFT: memoised t-emb pipeline.  Same (t, r)
        // pair always produces the same vector (deterministic functions of
        // t, r and the model weights).  See compute_time_emb_cached.
        auto t_emb = compute_time_emb_cached(m, t, r);

        if (debug_mode) {
            npy_array ref = npy_load(ref_dir + "/cfm_t_mix_call" + std::to_string(s) + ".npy");
            const float * r_ = (const float*)ref.data.data();
            size_t n = std::min(t_emb.size(), ref.n_elements());
            float ma = 0, rsum = 0;
            for (size_t i = 0; i < n; ++i) { float d = t_emb[i] - r_[i]; ma = std::max(ma, std::fabs(d)); rsum += d*d; }
            fprintf(stderr, "    [t_emb step%zu] max_abs=%.4e rms=%.4e vs ref\n", s, ma, std::sqrt(rsum / n));

            // Dump my x_in and see if it matches reference cfm_step{s}_x_in.npy
            npy_array ref_x = npy_load(ref_dir + "/cfm_step" + std::to_string(s) + "_x_in.npy");
            const float * rx = (const float*)ref_x.data.data();
            size_t nx = std::min(z.size(), ref_x.n_elements());
            ma = 0; rsum = 0;
            for (size_t i = 0; i < nx; ++i) { float d = z[i] - rx[i]; ma = std::max(ma, std::fabs(d)); rsum += d*d; }
            fprintf(stderr, "    [x_in step%zu] max_abs=%.4e rms=%.4e vs ref\n", s, ma, std::sqrt(rsum / nx));
        }

        double step_t0 = now_ms();
        auto dxdt = cfm_estimator_forward(m, cfm_cache, z, mu, t_emb, spks, cond, T_mu);
        vlog("  [cfm_step%zu] %.1f ms\n", s, now_ms() - step_t0);

        if (debug_mode) {
            npy_array ref = npy_load(ref_dir + "/cfm_step" + std::to_string(s) + "_dxdt.npy");
            const float * r_ = (const float*)ref.data.data();
            size_t n = std::min(dxdt.size(), ref.n_elements());
            float ma = 0, rsum = 0;
            for (size_t i = 0; i < n; ++i) { float d = dxdt[i] - r_[i]; ma = std::max(ma, std::fabs(d)); rsum += d*d; }
            fprintf(stderr, "    [dxdt step%zu] max_abs=%.4e rms=%.4e vs ref\n", s, ma, std::sqrt(rsum / n));
        }

        // Streaming: dump step0 dxdt for per-chunk bisection.
        if (s == 0 && !opts.dump_mel_path.empty()) {
            std::string base = opts.dump_mel_path;
            if (base.size() > 4 && base.substr(base.size() - 4) == ".npy")
                base.resize(base.size() - 4);
            npy_save_f32(base + "_step0_dxdt.npy", {MEL, (int64_t)T_mu}, dxdt.data());
            fprintf(stderr, "  [stream] dumped step0_dxdt (%d, %d) → %s_step0_dxdt.npy\n",
                    MEL, T_mu, base.c_str());
        }

        for (size_t i = 0; i < z.size(); ++i) z[i] = z[i] + dt * dxdt[i];
    }
    vlog("  [cfm_total] %.1f ms\n", now_ms() - cfm_t0);

    // 8) Slice mel = z[:, mel_len1:] -> shape (80, T_mu - mel_len1).
    //
    // For streaming (finalize=false), also drop the last
    //   pre_lookahead_len * token_mel_ratio = 3 * 2 = 6 mel frames
    // — these aren't "safe" yet (they'd change once the next chunk's tokens
    // provide more right-context).  Matches the trim in Python
    // CausalMaskedDiffWithXvec.inference(..., finalize=False).
    // Beyond-prompt mel span: starts at mel_len1 in z, ends at T_mu.
    // Streaming's 6-frame tail trim is already baked into T_mu above (we
    // shrunk it pre-CFM, matching Python).  Here we only apply the caller's
    // skip offset to drop frames already emitted on prior chunks.
    int T_mel_full = T_mu - mel_len1;
    int skip = opts.skip_mel_frames;
    if (skip < 0) skip = 0;
    if (skip > T_mel_full) {
        fprintf(stderr, "error: skip_mel_frames=%d > T_mel_full=%d\n", skip, T_mel_full);
        return 1;
    }
    int T_mel = T_mel_full - skip;
    vlog("Mel slicing: T_mu=%d mel_len1=%d full=%d skip=%d -> T_mel=%d  "
                    "(finalize=%s, pad_silence=%s)\n",
            T_mu, mel_len1, T_mel_full, skip, T_mel,
            opts.finalize ? "true" : "false",
            opts.append_lookahead_silence ? "true" : "false");
    std::vector<float> mel(MEL * T_mel);
    // z has shape ne=[T_mu, MEL]; grab the slice [mel_len1 + skip, mel_len1 + skip + T_mel).
    const int mel_off = mel_len1 + skip;
    for (int m2 = 0; m2 < MEL; ++m2)
        for (int t = 0; t < T_mel; ++t)
            mel[m2 * T_mel + t] = z[m2 * T_mu + (t + mel_off)];

    if (!opts.dump_mel_path.empty()) {
        std::vector<float> mel_tn((size_t)T_mel * MEL);
        for (int t = 0; t < T_mel; ++t)
            for (int m2 = 0; m2 < MEL; ++m2)
                mel_tn[(size_t)t * MEL + m2] = mel[(size_t)m2 * T_mel + t];
        npy_save_f32(opts.dump_mel_path, {(int64_t)T_mel, MEL}, mel_tn.data());
        fprintf(stderr, "  dumped mel (%d, %d) → %s\n", T_mel, MEL, opts.dump_mel_path.c_str());
    }

    if (debug_mode) {
        npy_array ref_mel = npy_load(ref_dir + "/mel_output.npy");
        const float * r = (const float*)ref_mel.data.data();
        size_t n = std::min(mel.size(), ref_mel.n_elements());
        float ma = 0, rsum = 0, max_ref = 0;
        for (size_t i = 0; i < n; ++i) {
            float d = mel[i] - r[i];
            ma = std::max(ma, std::fabs(d));
            rsum += d * d;
            max_ref = std::max(max_ref, std::fabs(r[i]));
        }
        fprintf(stderr, "  [mel] max_abs=%.4e rms=%.4e max|ref|=%.4e rel=%.4e\n",
                ma, std::sqrt(rsum / n), max_ref, ma / std::max(max_ref, 1e-9f));
    }

    // 9) HiFT
    double hift_t0 = now_ms();
    vlog("Running f0_predictor...\n");
    double t0 = now_ms();
    auto f0 = run_f0_predictor(m_hift, mel, T_mel);
    vlog("  [f0_predictor] %.1f ms\n", now_ms() - t0);
    int upsample = 8 * 5 * 3 * 4;
    int T_wav = T_mel * upsample;
    std::vector<float> f0_up(T_wav);
    for (int i = 0; i < T_mel; ++i)
        for (int j = 0; j < upsample; ++j) f0_up[i * upsample + j] = f0[i];

    vlog("Running SineGen...\n");
    t0 = now_ms();
    std::vector<float> l_linear_w(9);
    ggml_tensor * llw = find_tensor(m_hift, "hift/m_source/l_linear/weight");
    ggml_tensor * llb = find_tensor(m_hift, "hift/m_source/l_linear/bias");
    ggml_backend_tensor_get(llw, l_linear_w.data(), 0, 9 * sizeof(float));
    float l_linear_b;
    ggml_backend_tensor_get(llb, &l_linear_b, 0, sizeof(float));
    auto src = sinegen_source(f0_up, sr, 8, 0.1f, 0.003f, 10.0f, l_linear_w, l_linear_b, (uint32_t)(seed + 1));
    vlog("  [sinegen] %.1f ms\n", now_ms() - t0);

    // Streaming: splice in the previous chunk's source tail so the F0 phase
    // (and hence the vocoded waveform) stays continuous at the chunk seam.
    // Python equivalent: `s[:, :, :cache_source.shape[2]] = cache_source`
    // inside HiFTGenerator.inference.
    if (!opts.hift_cache_source.empty()) {
        size_t n = std::min(opts.hift_cache_source.size(), src.size());
        std::memcpy(src.data(), opts.hift_cache_source.data(), n * sizeof(float));
        vlog("  [sinegen] spliced %zu cache_source samples at head\n", n);
    }
    // Export the tail for the next chunk's cache_source BEFORE running STFT
    // (Python returns `s` unmodified; our tail copy matches that convention).
    if (opts.hift_source_tail_out != nullptr && opts.source_tail_samples > 0) {
        int tail_n = std::min((int)src.size(), opts.source_tail_samples);
        opts.hift_source_tail_out->assign(src.end() - tail_n, src.end());
    }

    vlog("Running STFT...\n");
    t0 = now_ms();
    auto s_stft = run_stft(m_hift, src);
    vlog("  [stft] %.1f ms\n", now_ms() - t0);
    int T_stft = (int)(s_stft.size() / 18);

    vlog("Running HiFT decode...\n");
    t0 = now_ms();
    auto wav = run_hift_decode(m_hift, mel, T_mel, s_stft, T_stft);
    vlog("  [hift_decode] %.1f ms\n", now_ms() - t0);
    vlog("  [hift_total] %.1f ms\n", now_ms() - hift_t0);
    vlog("  wav: %zu samples (%.3fs @ %d Hz)\n", wav.size(), (float)wav.size() / sr, sr);

    // First-chunk / batch-mode: apply raised-cosine fade-in to mask HiFT's
    // resnet cold start.  Length = 2*(sr/50) = 960 samples (40 ms) at 24 kHz.
    // First half is zero, second half is (cos(π→0)+1)/2 (0→1 ramp).
    // Python equivalent: `output_wavs[:, :len(self.trim_fade)] *= self.trim_fade`.
    if (opts.apply_trim_fade) {
        const int n_trim = sr / 50;  // 480 at 24 kHz
        const int fade_len = 2 * n_trim;
        if ((int)wav.size() >= fade_len) {
            for (int i = 0; i < n_trim; ++i) wav[i] = 0.0f;
            for (int i = 0; i < n_trim; ++i) {
                float theta = (float)M_PI * (1.0f - (float)i / (float)n_trim);
                float w = 0.5f * (std::cos(theta) + 1.0f);
                wav[n_trim + i] *= w;
            }
        }
    }

    double pipeline_total = now_ms() - pipeline_t0;
    double audio_ms = 1000.0 * wav.size() / sr;
    fprintf(stderr, "BENCH: S3GEN_INFER_MS=%.0f AUDIO_MS=%.0f\n", pipeline_total, audio_ms);
    vlog("\n=== pipeline: %.1f ms for %.1f ms of audio (RTF=%.2f, %.1fx %s) ===\n",
         pipeline_total, audio_ms,
         pipeline_total / audio_ms,
         audio_ms / pipeline_total >= 1.0 ? audio_ms / pipeline_total : pipeline_total / audio_ms,
         audio_ms >= pipeline_total ? "faster than real-time" : "slower than real-time");

    if (opts.pcm_out) {
        *opts.pcm_out = wav;
    }
    if (!out_path.empty()) {
        write_wav(out_path, wav, sr);
        fprintf(stderr, "Wrote %s\n", out_path.c_str());
    }
    return 0;
}

int s3gen_preload(const std::string & s3gen_gguf_path, int n_gpu_layers) {
    try {
        (void)s3gen_model_cache_get(s3gen_gguf_path, n_gpu_layers, /*verbose=*/false);
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "s3gen_preload: %s\n", e.what());
        return 1;
    }
}

void s3gen_unload() {
    s3gen_model_cache_release();
}
