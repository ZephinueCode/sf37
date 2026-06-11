#define _POSIX_C_SOURCE 200809L

#include "sf37.h"
#include "sf37_image.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static void usage(const char *argv0) {
    printf("usage: %s --model MODEL.gguf [--tokenizer DIR] [--inspect] [--backend cpu|cuda]\n", argv0);
    printf("\n");
    printf("Model/tokenizer:\n");
    printf("  --model FILE            SF37 GGUF model file.\n");
    printf("  --tokenizer DIR         Directory containing tokenizer.json and tokenizer_config.json.\n");
    printf("                          Defaults to the directory containing --model.\n");
    printf("  --hf DIR                Deprecated alias for --tokenizer.\n");
    printf("\n");
    printf("Current first-stage command:\n");
    printf("  --inspect          Validate SF37 metadata, tensor layouts, tokenizer/processor files.\n");
    printf("  --smoke-token ID   Run a CPU single-token decode math smoke test.\n");
    printf("  --smoke-layers N   Number of text layers to execute for --smoke-token (default: 4).\n");
    printf("  --smoke-topk K     Also run final RMSNorm + BF16 lm_head and print top K logits.\n");
    printf("  --smoke-logits     Shortcut for --smoke-topk 10.\n");
    printf("  --cuda-smoke-token ID  Run CUDA single-token decode parity through --smoke-layers.\n");
    printf("  --cuda-bench-token ID  Run CUDA-only decode timing; no CPU parity path.\n");
    printf("  --bench-repeat N       CUDA bench token repetitions (default: 1).\n");
    printf("  --bench-cache N        CUDA bench KV cache rows (default: 1).\n");
    printf("  --bench-logits         Include CUDA final RMSNorm + BF16 lm_head in bench timing.\n");
    printf("  --cuda-layer0-smoke ID  Run real-weight CUDA layer-0 parity smoke test.\n");
    printf("  --cuda-layer0-seq-smoke A B  Run two-token CUDA layer-0 KV/attention parity smoke test.\n");
    printf("  --cuda-layer3-moe-smoke ID  Run isolated real-weight CUDA layer-3 MoE parity smoke test.\n");
    printf("  --cuda-layer-replay-smoke ID L  Run one CUDA layer from the exact CPU prefix hidden.\n");
    printf("\n");
    printf("Tokenizer/session decode:\n");
    printf("  --prompt TEXT           Encode TEXT as a Step-3.7 user chat prompt.\n");
    printf("  --prompt-file FILE      Encode FILE as a Step-3.7 user chat prompt.\n");
    printf("  --raw-prompt-file FILE  Encode FILE as plain tokenizer text.\n");
    printf("  --system TEXT           Optional system message for --prompt/--prompt-file.\n");
    printf("  --think none|on         Chat generation prefix (default on).\n");
    printf("  --gen-tokens N          Greedy decode N tokens with sf37_session.\n");
    printf("  --ctx N                 Session context allocation (default 4096).\n");
    printf("  --image FILE            Local JPEG/PNG path, file:// URL, or data:image URL. May repeat.\n");
    printf("  --image-pixels-f32 FILE Normalized float32 NCHW pixels for native CUDA vision.\n");
    printf("  --image-size 728|504    Square image size for --image-pixels-f32 (default 728).\n");
    printf("  --temp F                Sampling temperature; 0 keeps greedy argmax (default 0).\n");
    printf("  --top-p F               Nucleus sampling cutoff for --temp > 0 (default 1).\n");
    printf("  --min-p F               DS4-style min-p cutoff for --temp > 0 (default 0).\n");
    printf("  --seed N                Sampling RNG seed.\n");
    printf("  --dump-tokens           Print prompt token ids before generation.\n");
    printf("  --snapshot-smoke        Save/restore the prompt KV snapshot before decoding.\n");
    printf("\n");
    printf("Imatrix calibration:\n");
    printf("  --imatrix-dataset FILE  Rendered prompt dataset for routed-MoE imatrix collection.\n");
    printf("  --imatrix-out FILE      Write legacy .dat imatrix for sf37-quantize --imatrix.\n");
    printf("  --imatrix-max-prompts N Stop imatrix collection after N prompts.\n");
    printf("  --imatrix-max-tokens N  Stop imatrix collection after N prompt tokens.\n");
}

static const char *need_value(int argc, char **argv, int *i, const char *arg) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "sf37: %s requires a value\n", arg);
        exit(2);
    }
    return argv[++*i];
}

static int parse_int_range(const char *s, const char *arg, int min, int max) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || *end || v < min || v > max) {
        fprintf(stderr, "sf37: invalid value for %s: %s\n", arg, s);
        exit(2);
    }
    return (int)v;
}

static uint64_t parse_u64(const char *s, const char *arg) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (!s[0] || *end || v == 0) {
        fprintf(stderr, "sf37: invalid value for %s: %s\n", arg, s);
        exit(2);
    }
    return (uint64_t)v;
}

static float parse_float_range(const char *s, const char *arg, float min, float max) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (!s[0] || *end || !isfinite(v) || v < min || v > max) {
        fprintf(stderr, "sf37: invalid value for %s: %s\n", arg, s);
        exit(2);
    }
    return v;
}

static sf37_backend parse_backend(const char *s) {
    if (strcmp(s, "cpu") == 0) return SF37_BACKEND_CPU;
    if (strcmp(s, "cuda") == 0) return SF37_BACKEND_CUDA;
    fprintf(stderr, "sf37: unsupported backend: %s\n", s);
    exit(2);
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = malloc(n + 1u);
    if (!p) {
        fprintf(stderr, "sf37: out of memory\n");
        exit(1);
    }
    memcpy(p, s, n + 1u);
    return p;
}

static void setenv_default(const char *key, const char *value) {
    if (!getenv(key)) setenv(key, value, 0);
}

static void apply_imatrix_streaming_cuda_defaults(void) {
    setenv_default("SF37_CUDA_STARTUP_PRELOAD_GB", "0");
    setenv_default("SF37_CUDA_NO_PREFILL_Q8_CACHE", "1");
    setenv_default("SF37_CUDA_NO_Q8_F16_CACHE", "1");
    setenv_default("SF37_CUDA_NO_Q8_F32_CACHE", "1");
    if (getenv("SF37_CUDA_IMATRIX_KEEP_WEIGHT_CACHE")) {
        setenv_default("SF37_CUDA_WEIGHT_CACHE_LIMIT_GB", "84");
    } else {
        setenv_default("SF37_CUDA_WEIGHT_CACHE_LIMIT_GB", "16");
    }
}

static void *xrealloc_or_die(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1u);
    if (!q) {
        fprintf(stderr, "sf37: out of memory\n");
        exit(1);
    }
    return q;
}

static char *read_file_or_die(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "sf37: open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "sf37: seek %s failed\n", path);
        fclose(fp);
        exit(1);
    }
    long n = ftell(fp);
    if (n < 0) {
        fprintf(stderr, "sf37: tell %s failed\n", path);
        fclose(fp);
        exit(1);
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "sf37: rewind %s failed\n", path);
        fclose(fp);
        exit(1);
    }
    char *buf = malloc((size_t)n + 1u);
    if (!buf) {
        fprintf(stderr, "sf37: out of memory reading %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (n > 0 && fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        fprintf(stderr, "sf37: read %s failed\n", path);
        free(buf);
        fclose(fp);
        exit(1);
    }
    fclose(fp);
    buf[n] = '\0';
    return buf;
}

static void *read_binary_exact_or_die(const char *path, size_t expect) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "sf37: open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "sf37: seek %s failed\n", path);
        fclose(fp);
        exit(1);
    }
    long n = ftell(fp);
    if (n < 0) {
        fprintf(stderr, "sf37: tell %s failed\n", path);
        fclose(fp);
        exit(1);
    }
    if ((size_t)n != expect) {
        fprintf(stderr, "sf37: %s has %ld bytes, expected %zu\n", path, n, expect);
        fclose(fp);
        exit(1);
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "sf37: rewind %s failed\n", path);
        fclose(fp);
        exit(1);
    }
    void *buf = malloc(expect ? expect : 1u);
    if (!buf) {
        fprintf(stderr, "sf37: out of memory reading %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (expect && fread(buf, 1, expect, fp) != expect) {
        fprintf(stderr, "sf37: read %s failed\n", path);
        free(buf);
        fclose(fp);
        exit(1);
    }
    fclose(fp);
    return buf;
}

static char *append_image_placeholder_text(const char *text, int image_size) {
    const int rows = image_size == 728 ? 169 : image_size == 504 ? 81 : 0;
    if (rows <= 0) return NULL;
    const char *start = rows == 169 ? "<im_start>" : "<patch_start>";
    const char *end = rows == 169 ? "<im_end>" : "<patch_end>";
    const char *patch = "<im_patch>";
    if (text && strstr(text, patch)) return xstrdup(text);
    const size_t text_len = text ? strlen(text) : 0;
    const size_t start_len = strlen(start);
    const size_t patch_len = strlen(patch);
    const size_t end_len = strlen(end);
    const size_t extra = (text_len ? 1u : 0u) + start_len +
                         (size_t)rows * patch_len + end_len;
    char *out = malloc(text_len + extra + 1u);
    if (!out) {
        fprintf(stderr, "sf37: out of memory building image prompt\n");
        exit(1);
    }
    char *p = out;
    if (text_len) {
        memcpy(p, text, text_len);
        p += text_len;
        *p++ = '\n';
    }
    memcpy(p, start, start_len);
    p += start_len;
    for (int i = 0; i < rows; i++) {
        memcpy(p, patch, patch_len);
        p += patch_len;
    }
    memcpy(p, end, end_len);
    p += end_len;
    *p = '\0';
    return out;
}

static bool is_rendered_chat_prompt(const char *text) {
    const char *bos = "<｜begin▁of▁sentence｜>";
    return text && strncmp(text, bos, strlen(bos)) == 0;
}

static void encode_prompt_input(sf37_engine *e,
                                const char *system,
                                const char *text,
                                sf37_think_mode think,
                                sf37_tokens *prompt) {
    if (is_rendered_chat_prompt(text)) {
        sf37_tokenize_rendered_chat(e, text, prompt);
    } else {
        sf37_encode_chat_prompt(e, system, text, think, prompt);
    }
}

static void dump_prompt_tokens(sf37_engine *e, const sf37_tokens *prompt) {
    fprintf(stderr, "sf37: prompt tokens: %d\n", prompt ? prompt->len : 0);
    if (!prompt) return;
    for (int i = 0; i < prompt->len; i++) {
        size_t len = 0;
        char *piece = sf37_token_text(e, prompt->v[i], &len);
        fprintf(stderr, "%6d  %8d  ", i, prompt->v[i]);
        for (size_t j = 0; j < len; j++) {
            unsigned char c = (unsigned char)piece[j];
            if (c >= 0x20 && c < 0x7f && c != '\\') fputc(c, stderr);
            else fprintf(stderr, "\\x%02x", (unsigned)c);
        }
        fputc('\n', stderr);
        free(piece);
    }
}

static int run_snapshot_smoke(sf37_session *session, sf37_engine *e) {
    char err[256];
    sf37_session_snapshot snap = {0};
    float *before = malloc((size_t)sf37_engine_vocab_size(e) * sizeof(before[0]));
    float *after = malloc((size_t)sf37_engine_vocab_size(e) * sizeof(after[0]));
    if (!before || !after) {
        fprintf(stderr, "sf37: snapshot smoke allocation failed\n");
        free(after);
        free(before);
        return 1;
    }
    const int vocab = sf37_engine_vocab_size(e);
    if (sf37_session_copy_logits(session, before, vocab) != vocab ||
        sf37_session_save_snapshot(session, &snap, err, sizeof(err)) != 0) {
        fprintf(stderr, "sf37: snapshot save failed: %s\n", err);
        sf37_session_snapshot_free(&snap);
        free(after);
        free(before);
        return 1;
    }
    const int eos = sf37_token_eos(e);
    const int tok0 = sf37_session_argmax_excluding(session, eos);
    if (tok0 < 0 || sf37_session_eval(session, tok0, err, sizeof(err)) != 0) {
        fprintf(stderr, "sf37: snapshot smoke decode failed: %s\n", tok0 < 0 ? "argmax failed" : err);
        sf37_session_snapshot_free(&snap);
        free(after);
        free(before);
        return 1;
    }
    if (sf37_session_load_snapshot(session, &snap, err, sizeof(err)) != 0 ||
        sf37_session_copy_logits(session, after, vocab) != vocab) {
        fprintf(stderr, "sf37: snapshot restore failed: %s\n", err);
        sf37_session_snapshot_free(&snap);
        free(after);
        free(before);
        return 1;
    }
    const int tok1 = sf37_session_argmax_excluding(session, eos);
    float max_abs = 0.0f;
    for (int i = 0; i < vocab; i++) {
        const float d = fabsf(before[i] - after[i]);
        if (d > max_abs) max_abs = d;
    }
    if (tok0 != tok1 || max_abs > 0.0f) {
        fprintf(stderr,
                "sf37: snapshot smoke mismatch: token before=%d after=%d max_logit_abs=%g\n",
                tok0, tok1, max_abs);
        sf37_session_snapshot_free(&snap);
        free(after);
        free(before);
        return 1;
    }
    fprintf(stderr, "sf37: snapshot smoke ok at pos=%d bytes=%llu token=%d\n",
            sf37_session_pos(session),
            (unsigned long long)snap.len,
            tok1);
    sf37_session_snapshot_free(&snap);
    free(after);
    free(before);
    return 0;
}

static void cli_prefill_progress(void *ud, const char *event, int current, int total) {
    (void)ud;
    if (!event || strcmp(event, "prefill_chunk") != 0) return;
    if (total <= 0 || current <= 0 || current == total || (current % 16) == 0) {
        fprintf(stderr, "sf37: processing input tokens: %d/%d\r", current, total);
        if (current == total) fputc('\n', stderr);
        fflush(stderr);
    }
}

typedef struct {
    FILE *fp;
    bool format_thinking;
    bool in_think;
    bool color_open;
    bool use_color;
    bool last_output_newline;
    bool thinking_label_emitted;
    bool answer_label_emitted;
    char pending[16];
    size_t pending_len;
} token_printer;

static bool bytes_has_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n >= plen && memcmp(p, prefix, plen) == 0;
}

static bool bytes_is_partial_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n < plen && memcmp(prefix, p, n) == 0;
}

static void token_printer_reset_color(token_printer *p) {
    if (p->use_color && p->color_open) {
        fputs("\x1b[0m", p->fp);
        p->color_open = false;
    }
}

static void token_printer_set_grey(token_printer *p) {
    if (p->use_color && !p->color_open) {
        fputs("\x1b[90m", p->fp);
        p->color_open = true;
    }
}

static void token_printer_emit_label(token_printer *p, const char *label) {
    token_printer_reset_color(p);
    if (!p->last_output_newline) fputc('\n', p->fp);
    fputs(label, p->fp);
    fputc('\n', p->fp);
    p->last_output_newline = true;
}

static void token_printer_maybe_emit_thinking_label(token_printer *p) {
    if (!p->format_thinking || p->thinking_label_emitted) return;
    token_printer_emit_label(p, "[thinking]");
    p->thinking_label_emitted = true;
}

static void token_printer_maybe_emit_answer_label(token_printer *p) {
    if (!p->format_thinking || p->answer_label_emitted) return;
    token_printer_emit_label(p, "[answer]");
    p->answer_label_emitted = true;
}

static void token_printer_write_char(token_printer *p, char c) {
    if (p->format_thinking && p->in_think) {
        token_printer_maybe_emit_thinking_label(p);
        token_printer_set_grey(p);
    } else {
        if (p->format_thinking) token_printer_maybe_emit_answer_label(p);
        token_printer_reset_color(p);
    }
    fputc((unsigned char)c, p->fp);
    p->last_output_newline = c == '\n';
}

static void token_printer_process(token_printer *p, const char *text, size_t len, bool finish) {
    if (!p->format_thinking) {
        if (len) {
            fwrite(text, 1, len, p->fp);
            p->last_output_newline = text[len - 1] == '\n';
        }
        return;
    }

    const char *think_open = "<think>";
    const char *think_close = "</think>";
    size_t total = p->pending_len + len;
    char *buf = malloc(total ? total : 1u);
    if (!buf) return;
    if (p->pending_len) memcpy(buf, p->pending, p->pending_len);
    if (len) memcpy(buf + p->pending_len, text, len);
    p->pending_len = 0;

    size_t i = 0;
    while (i < total) {
        const char *cur = buf + i;
        size_t rem = total - i;
        if (bytes_has_prefix(cur, rem, think_open)) {
            p->in_think = true;
            i += strlen(think_open);
            continue;
        }
        if (bytes_has_prefix(cur, rem, think_close)) {
            if (p->in_think && !p->last_output_newline) {
                token_printer_reset_color(p);
                fputc('\n', p->fp);
                p->last_output_newline = true;
            }
            p->in_think = false;
            token_printer_reset_color(p);
            i += strlen(think_close);
            continue;
        }
        if (!finish && cur[0] == '<' &&
            (bytes_is_partial_prefix(cur, rem, think_open) ||
             bytes_is_partial_prefix(cur, rem, think_close)))
        {
            if (rem < sizeof(p->pending)) {
                memcpy(p->pending, cur, rem);
                p->pending_len = rem;
            }
            break;
        }
        token_printer_write_char(p, cur[0]);
        i++;
    }

    free(buf);
}

static void token_printer_write_text(token_printer *p, const char *text, size_t len) {
    token_printer_process(p, text, len, false);
}

static void token_printer_finish(token_printer *p) {
    token_printer_process(p, NULL, 0, true);
    token_printer_reset_color(p);
    fflush(p->fp);
}

static int run_session_generate(sf37_engine *e,
                                const char *system,
                                const char *prompt_text,
                                const char *prompt_path,
                                const char *raw_prompt_path,
                                const char **image_paths,
                                int image_path_count,
                                const char *image_pixels_path,
                                int image_size,
                                sf37_think_mode think,
                                int ctx_size,
                                int gen_tokens,
                                float temperature,
                                float top_p,
                                float min_p,
                                uint64_t seed,
                                bool dump_tokens,
                                bool snapshot_smoke) {
    if (!sf37_tokenizer_ready(e)) {
        fprintf(stderr, "sf37: tokenizer is not loaded; pass --tokenizer DIR or place tokenizer.json next to the GGUF\n");
        return 1;
    }
    sf37_tokens prompt = {0};
    char *owned = NULL;
    char *image_prompt = NULL;
    float *image_pixels = NULL;
    sf37_image_batch image_batch = {0};
    sf37_image_features image_features = {0};
    const bool has_image_file = image_path_count > 0;
    const bool has_raw_pixels = image_pixels_path != NULL;
    const bool has_image = has_image_file || has_raw_pixels;
    if (has_image_file && has_raw_pixels) {
        fprintf(stderr, "sf37: use either --image or --image-pixels-f32, not both\n");
        return 1;
    }
    if (has_image_file) {
        char err[256] = {0};
        for (int i = 0; i < image_path_count; i++) {
            if (sf37_image_batch_add_source(&image_batch, image_paths[i], err, sizeof(err)) != 0) {
                fprintf(stderr, "sf37: image preprocess failed for %s: %s\n",
                        image_paths[i], err[0] ? err : "unknown error");
                sf37_image_batch_free(&image_batch);
                return 1;
            }
        }
        sf37_image_features_from_batch(&image_batch, &image_features);
    }
    if (prompt_text) {
        const char *text = prompt_text;
        if (has_image_file) {
            char err[256] = {0};
            image_prompt = sf37_image_batch_apply_placeholders(&image_batch, text, err, sizeof(err));
            if (!image_prompt) {
                fprintf(stderr, "sf37: %s\n", err[0] ? err : "failed to build image prompt");
                sf37_image_batch_free(&image_batch);
                return 1;
            }
            text = image_prompt;
        } else if (has_raw_pixels) {
            image_prompt = append_image_placeholder_text(text, image_size);
            text = image_prompt;
        }
        encode_prompt_input(e, system, text, think, &prompt);
    } else if (prompt_path) {
        owned = read_file_or_die(prompt_path);
        const char *text = owned;
        if (has_image_file) {
            char err[256] = {0};
            image_prompt = sf37_image_batch_apply_placeholders(&image_batch, text, err, sizeof(err));
            if (!image_prompt) {
                fprintf(stderr, "sf37: %s\n", err[0] ? err : "failed to build image prompt");
                free(owned);
                sf37_image_batch_free(&image_batch);
                return 1;
            }
            text = image_prompt;
        } else if (has_raw_pixels) {
            image_prompt = append_image_placeholder_text(text, image_size);
            text = image_prompt;
        }
        encode_prompt_input(e, system, text, think, &prompt);
    } else if (raw_prompt_path) {
        owned = read_file_or_die(raw_prompt_path);
        const char *text = owned;
        if (has_image_file) {
            char err[256] = {0};
            image_prompt = sf37_image_batch_apply_placeholders(&image_batch, text, err, sizeof(err));
            if (!image_prompt) {
                fprintf(stderr, "sf37: %s\n", err[0] ? err : "failed to build image prompt");
                free(owned);
                sf37_image_batch_free(&image_batch);
                return 1;
            }
            text = image_prompt;
        } else if (has_raw_pixels) {
            image_prompt = append_image_placeholder_text(text, image_size);
            text = image_prompt;
        }
        sf37_tokenize_text(e, text, &prompt);
    }
    free(image_prompt);
    free(owned);

    if (prompt.len == 0) {
        fprintf(stderr, "sf37: prompt encoded to zero tokens\n");
        sf37_image_batch_free(&image_batch);
        sf37_tokens_free(&prompt);
        return 1;
    }
    if (has_raw_pixels) {
        const uint64_t nfloat = (uint64_t)3u * (uint32_t)image_size * (uint32_t)image_size;
        if (nfloat > SIZE_MAX / sizeof(float)) {
            fprintf(stderr, "sf37: image is too large\n");
            sf37_image_batch_free(&image_batch);
            sf37_tokens_free(&prompt);
            return 1;
        }
        image_pixels = read_binary_exact_or_die(image_pixels_path, (size_t)nfloat * sizeof(float));
        image_features.pixel_values = image_pixels;
        image_features.images = 1;
        image_features.pixel_channels = 3;
        image_features.pixel_height = (uint32_t)image_size;
        image_features.pixel_width = (uint32_t)image_size;
    }
    if (ctx_size <= prompt.len + gen_tokens) {
        fprintf(stderr,
                "sf37: ctx=%d is too small for prompt=%d plus gen=%d\n",
                ctx_size, prompt.len, gen_tokens);
        free(image_pixels);
        sf37_image_batch_free(&image_batch);
        sf37_tokens_free(&prompt);
        return 1;
    }
    if (dump_tokens) dump_prompt_tokens(e, &prompt);

    sf37_session *session = NULL;
    if (sf37_session_create(&session, e, ctx_size) != 0) {
        fprintf(stderr, "sf37: failed to create session\n");
        free(image_pixels);
        sf37_image_batch_free(&image_batch);
        sf37_tokens_free(&prompt);
        return 1;
    }

    sf37_session_set_progress(session, cli_prefill_progress, NULL);
    char err[256];
    int sync_rc = has_image ?
        sf37_session_sync_multimodal(session, &prompt, &image_features, err, sizeof(err)) :
        sf37_session_sync(session, &prompt, err, sizeof(err));
    if (sync_rc != 0) {
        fprintf(stderr, "sf37: prompt eval failed: %s\n", err);
        sf37_session_free(session);
        free(image_pixels);
        sf37_image_batch_free(&image_batch);
        sf37_tokens_free(&prompt);
        return 1;
    }
    sf37_session_set_progress(session, NULL, NULL);
    if (snapshot_smoke && run_snapshot_smoke(session, e) != 0) {
        sf37_session_free(session);
        free(image_pixels);
        sf37_image_batch_free(&image_batch);
        sf37_tokens_free(&prompt);
        return 1;
    }

    const int eos = sf37_token_eos(e);
    uint64_t rng = seed ? seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    token_printer printer = {
        .fp = stdout,
        .format_thinking = raw_prompt_path == NULL && think == SF37_THINK_ENABLED,
        .in_think = raw_prompt_path == NULL && think == SF37_THINK_ENABLED,
        .use_color = isatty(STDOUT_FILENO) != 0,
        .last_output_newline = true,
    };
    int greedy_token = -1;
    if (temperature <= 0.0f && gen_tokens > 0) {
        greedy_token = sf37_session_argmax(session);
    }
    for (int i = 0; i < gen_tokens; i++) {
        int token = temperature <= 0.0f ?
            greedy_token :
            sf37_session_sample(session, temperature, 0, top_p, min_p, &rng);
        if (token < 0) {
            fprintf(stderr, "sf37: failed to select token at step %d\n", i);
            sf37_session_free(session);
            free(image_pixels);
            sf37_image_batch_free(&image_batch);
            sf37_tokens_free(&prompt);
            return 1;
        }
        if (token == eos) break;
        size_t len = 0;
        char *piece = sf37_token_text(e, token, &len);
        if (piece && len > 0) {
            token_printer_write_text(&printer, piece, len);
            fflush(stdout);
        }
        free(piece);
        if (temperature <= 0.0f) {
            greedy_token = sf37_session_eval_argmax(session, token, err, sizeof(err));
            if (greedy_token < 0) {
                fprintf(stderr, "sf37: decode failed at step %d: %s\n", i, err);
                sf37_session_free(session);
                free(image_pixels);
                sf37_image_batch_free(&image_batch);
                sf37_tokens_free(&prompt);
                return 1;
            }
        } else {
            if (sf37_session_eval(session, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "sf37: decode failed at step %d: %s\n", i, err);
                sf37_session_free(session);
                free(image_pixels);
                sf37_image_batch_free(&image_batch);
                sf37_tokens_free(&prompt);
                return 1;
            }
        }
    }
    token_printer_finish(&printer);
    if (gen_tokens > 0 && !printer.last_output_newline) fputc('\n', stdout);

    sf37_session_free(session);
    free(image_pixels);
    sf37_image_batch_free(&image_batch);
    sf37_tokens_free(&prompt);
    return 0;
}

int main(int argc, char **argv) {
    sf37_engine_options opt = {
        .backend = SF37_BACKEND_CUDA,
        .inspect_only = false,
    };
    int smoke_token = -1;
    int smoke_layers = 4;
    int smoke_topk = 0;
    int cuda_smoke_token = -1;
    int cuda_bench_token = -1;
    int bench_repeat = 1;
    int bench_cache = 1;
    bool bench_logits = false;
    int cuda_layer0_smoke = -1;
    int cuda_layer0_seq0 = -1;
    int cuda_layer0_seq1 = -1;
    int cuda_layer3_moe_smoke = -1;
    int cuda_layer_replay_token = -1;
    int cuda_layer_replay_layer = -1;
    const char *prompt_text = NULL;
    const char *prompt_path = NULL;
    const char *raw_prompt_path = NULL;
    const char **image_paths = NULL;
    int image_path_count = 0;
    int image_path_cap = 0;
    const char *image_pixels_path = NULL;
    int image_size = 728;
    const char *system = NULL;
    sf37_think_mode think_mode = SF37_THINK_ENABLED;
    int gen_tokens = 0;
    int ctx_size = 4096;
    float temperature = SF37_DEFAULT_TEMPERATURE;
    float top_p = SF37_DEFAULT_TOP_P;
    float min_p = SF37_DEFAULT_MIN_P;
    uint64_t seed = 0;
    bool dump_tokens = false;
    bool snapshot_smoke = false;
    const char *imatrix_dataset_path = NULL;
    const char *imatrix_output_path = NULL;
    int imatrix_max_prompts = 0;
    int imatrix_max_tokens = 0;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (strcmp(arg, "--model") == 0) {
            opt.model_path = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--hf") == 0 || strcmp(arg, "--tokenizer") == 0) {
            opt.tokenizer_path = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--inspect") == 0) {
            opt.inspect_only = true;
        } else if (strcmp(arg, "--smoke-token") == 0) {
            smoke_token = parse_int_range(need_value(argc, argv, &i, arg), arg, 0, 128895);
        } else if (strcmp(arg, "--smoke-layers") == 0) {
            smoke_layers = parse_int_range(need_value(argc, argv, &i, arg), arg, 0, 45);
        } else if (strcmp(arg, "--smoke-topk") == 0) {
            smoke_topk = parse_int_range(need_value(argc, argv, &i, arg), arg, 0, 128896);
        } else if (strcmp(arg, "--smoke-logits") == 0) {
            smoke_topk = 10;
        } else if (strcmp(arg, "--cuda-smoke-token") == 0) {
            cuda_smoke_token = parse_int_range(need_value(argc, argv, &i, arg), arg, 0, 128895);
        } else if (strcmp(arg, "--cuda-bench-token") == 0) {
            cuda_bench_token = parse_int_range(need_value(argc, argv, &i, arg), arg, 0, 128895);
        } else if (strcmp(arg, "--bench-repeat") == 0) {
            bench_repeat = parse_int_range(need_value(argc, argv, &i, arg), arg, 1, 1000000);
        } else if (strcmp(arg, "--bench-cache") == 0) {
            bench_cache = parse_int_range(need_value(argc, argv, &i, arg), arg, 1, 262144);
        } else if (strcmp(arg, "--bench-logits") == 0) {
            bench_logits = true;
        } else if (strcmp(arg, "--cuda-layer0-smoke") == 0) {
            cuda_layer0_smoke = parse_int_range(need_value(argc, argv, &i, arg), arg, 0, 128895);
        } else if (strcmp(arg, "--cuda-layer0-seq-smoke") == 0) {
            cuda_layer0_seq0 = parse_int_range(need_value(argc, argv, &i, arg), arg, 0, 128895);
            cuda_layer0_seq1 = parse_int_range(need_value(argc, argv, &i, arg), arg, 0, 128895);
        } else if (strcmp(arg, "--cuda-layer3-moe-smoke") == 0) {
            cuda_layer3_moe_smoke = parse_int_range(need_value(argc, argv, &i, arg), arg, 0, 128895);
        } else if (strcmp(arg, "--cuda-layer-replay-smoke") == 0) {
            cuda_layer_replay_token = parse_int_range(need_value(argc, argv, &i, arg), arg, 0, 128895);
            cuda_layer_replay_layer = parse_int_range(need_value(argc, argv, &i, arg), arg, 0, 44);
        } else if (strcmp(arg, "--prompt") == 0) {
            prompt_text = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--prompt-file") == 0) {
            prompt_path = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--raw-prompt-file") == 0) {
            raw_prompt_path = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--image") == 0 || strcmp(arg, "--image-file") == 0) {
            const char *v = need_value(argc, argv, &i, arg);
            if (image_path_count == image_path_cap) {
                image_path_cap = image_path_cap ? image_path_cap * 2 : 2;
                image_paths = xrealloc_or_die((void *)image_paths,
                                              (size_t)image_path_cap * sizeof(image_paths[0]));
            }
            image_paths[image_path_count++] = v;
        } else if (strcmp(arg, "--image-pixels-f32") == 0) {
            image_pixels_path = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--image-size") == 0) {
            image_size = parse_int_range(need_value(argc, argv, &i, arg), arg, 1, 4096);
        } else if (strcmp(arg, "--system") == 0) {
            system = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--think") == 0) {
            const char *v = need_value(argc, argv, &i, arg);
            if (strcmp(v, "on") == 0 || strcmp(v, "yes") == 0 || strcmp(v, "1") == 0) {
                think_mode = SF37_THINK_ENABLED;
            } else if (strcmp(v, "none") == 0 || strcmp(v, "off") == 0 || strcmp(v, "0") == 0) {
                think_mode = SF37_THINK_NONE;
            } else {
                fprintf(stderr, "sf37: --think must be on or none\n");
                return 2;
            }
        } else if (strcmp(arg, "--gen-tokens") == 0 || strcmp(arg, "-n") == 0) {
            gen_tokens = parse_int_range(need_value(argc, argv, &i, arg), arg, 0, 262144);
        } else if (strcmp(arg, "--ctx") == 0) {
            ctx_size = parse_int_range(need_value(argc, argv, &i, arg), arg, 1, 262144);
        } else if (strcmp(arg, "--temp") == 0 || strcmp(arg, "--temperature") == 0) {
            temperature = parse_float_range(need_value(argc, argv, &i, arg), arg, 0.0f, 100.0f);
        } else if (strcmp(arg, "--top-p") == 0) {
            top_p = parse_float_range(need_value(argc, argv, &i, arg), arg, 0.0f, 1.0f);
        } else if (strcmp(arg, "--min-p") == 0) {
            min_p = parse_float_range(need_value(argc, argv, &i, arg), arg, 0.0f, 1.0f);
        } else if (strcmp(arg, "--seed") == 0) {
            seed = parse_u64(need_value(argc, argv, &i, arg), arg);
        } else if (strcmp(arg, "--dump-tokens") == 0) {
            dump_tokens = true;
        } else if (strcmp(arg, "--snapshot-smoke") == 0) {
            snapshot_smoke = true;
        } else if (strcmp(arg, "--imatrix-dataset") == 0) {
            imatrix_dataset_path = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--imatrix-out") == 0) {
            imatrix_output_path = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--imatrix-max-prompts") == 0) {
            imatrix_max_prompts = parse_int_range(need_value(argc, argv, &i, arg), arg, 0, 100000000);
        } else if (strcmp(arg, "--imatrix-max-tokens") == 0) {
            imatrix_max_tokens = parse_int_range(need_value(argc, argv, &i, arg), arg, 0, 100000000);
        } else if (strcmp(arg, "--backend") == 0) {
            opt.backend = parse_backend(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            free(image_paths);
            return 0;
        } else {
            fprintf(stderr, "sf37: unknown argument: %s\n", arg);
            usage(argv[0]);
            free(image_paths);
            return 2;
        }
    }
    if (!opt.model_path) {
        fprintf(stderr, "sf37: --model is required\n");
        usage(argv[0]);
        free(image_paths);
        return 2;
    }
    int prompt_modes = (prompt_text != NULL) + (prompt_path != NULL) + (raw_prompt_path != NULL);
    if ((imatrix_dataset_path != NULL) != (imatrix_output_path != NULL)) {
        fprintf(stderr, "sf37: --imatrix-dataset and --imatrix-out must be specified together\n");
        free(image_paths);
        return 2;
    }
    if (imatrix_dataset_path && prompt_modes) {
        fprintf(stderr, "sf37: imatrix collection cannot be combined with prompt generation\n");
        free(image_paths);
        return 2;
    }
    if (prompt_modes > 1) {
        fprintf(stderr, "sf37: specify only one prompt input mode\n");
        free(image_paths);
        return 2;
    }
    if (prompt_modes && gen_tokens <= 0) {
        fprintf(stderr, "sf37: prompt decode requires --gen-tokens N\n");
        free(image_paths);
        return 2;
    }
    if (image_pixels_path && !prompt_modes) {
        fprintf(stderr, "sf37: --image-pixels-f32 requires a prompt input mode\n");
        free(image_paths);
        return 2;
    }
    if (image_path_count > 0 && !prompt_modes) {
        fprintf(stderr, "sf37: --image requires a prompt input mode\n");
        free(image_paths);
        return 2;
    }
    if (image_path_count > 0 && image_pixels_path) {
        fprintf(stderr, "sf37: use either --image or --image-pixels-f32, not both\n");
        free(image_paths);
        return 2;
    }
    if (image_pixels_path && image_size != 728 && image_size != 504) {
        fprintf(stderr, "sf37: --image-size must be 728 or 504\n");
        free(image_paths);
        return 2;
    }
    if (imatrix_dataset_path && opt.backend == SF37_BACKEND_CUDA) {
        apply_imatrix_streaming_cuda_defaults();
    }

    sf37_engine *e = NULL;
    if (sf37_engine_open(&e, &opt) != 0) {
        free(image_paths);
        return 1;
    }
    sf37_engine_summary(e, prompt_modes ? stderr : stdout);

    if (imatrix_output_path) {
        int rc = sf37_engine_collect_imatrix(e,
                                             imatrix_dataset_path,
                                             imatrix_output_path,
                                             ctx_size,
                                             imatrix_max_prompts,
                                             imatrix_max_tokens);
        sf37_engine_close(e);
        free(image_paths);
        return rc == 0 ? 0 : 1;
    }

    if (prompt_modes) {
        int rc = run_session_generate(e, system, prompt_text, prompt_path, raw_prompt_path,
                                      image_paths, image_path_count,
                                      image_pixels_path, image_size,
                                      think_mode, ctx_size, gen_tokens,
                                      temperature, top_p, min_p, seed,
                                      dump_tokens, snapshot_smoke);
        sf37_engine_close(e);
        free(image_paths);
        return rc == 0 ? 0 : 1;
    }
    free(image_paths);

    if (cuda_layer0_seq0 >= 0) {
        int rc = sf37_engine_cuda_layer0_seq_smoke(e, cuda_layer0_seq0, cuda_layer0_seq1, stdout);
        sf37_engine_close(e);
        return rc == 0 ? 0 : 1;
    }

    if (cuda_layer0_smoke >= 0) {
        int rc = sf37_engine_cuda_layer0_smoke(e, cuda_layer0_smoke, stdout);
        sf37_engine_close(e);
        return rc == 0 ? 0 : 1;
    }

    if (cuda_layer3_moe_smoke >= 0) {
        int rc = sf37_engine_cuda_layer3_moe_smoke(e, cuda_layer3_moe_smoke, stdout);
        sf37_engine_close(e);
        return rc == 0 ? 0 : 1;
    }

    if (cuda_layer_replay_token >= 0) {
        int rc = sf37_engine_cuda_layer_replay_smoke(e, cuda_layer_replay_token,
                                                     cuda_layer_replay_layer, stdout);
        sf37_engine_close(e);
        return rc == 0 ? 0 : 1;
    }

    if (cuda_smoke_token >= 0) {
        int rc = sf37_engine_cuda_smoke_decode(e, cuda_smoke_token, smoke_layers, smoke_topk, stdout);
        sf37_engine_close(e);
        return rc == 0 ? 0 : 1;
    }

    if (cuda_bench_token >= 0) {
        int rc = sf37_engine_cuda_bench_decode(e, cuda_bench_token, smoke_layers,
                                               bench_repeat, bench_cache,
                                               bench_logits, stdout);
        sf37_engine_close(e);
        return rc == 0 ? 0 : 1;
    }

    if (smoke_token >= 0) {
        int rc = sf37_engine_smoke_decode(e, smoke_token, smoke_layers, smoke_topk, stdout);
        sf37_engine_close(e);
        return rc == 0 ? 0 : 1;
    }

    if (!opt.inspect_only) {
        sf37_log(stderr, SF37_LOG_WARNING,
                 "decode runtime is not enabled in this first-stage build; rerun with --inspect");
        sf37_engine_close(e);
        return 2;
    }
    sf37_engine_close(e);
    return 0;
}
