#define _POSIX_C_SOURCE 200809L

#include "sf37.h"

#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const char *model_path;
    const char *tokenizer_path;
    const char *tokens_path;
    const char *prompt_path;
    const char *raw_prompt_path;
    const char *system;
    const char *csv_path;
    sf37_backend backend;
    int ctx_start;
    int ctx_max;
    int ctx_alloc;
    int step_incr;
    int gen_tokens;
    int repeat_token;
    int decode_token;
    int eos_token;
    bool think;
    bool verbose;
    bool warm_weights;
} bench_config;

static double bench_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s --model MODEL.gguf [--tokenizer DIR] [options]\n"
            "\n"
            "Input:\n"
            "  --tokenizer DIR         Directory containing tokenizer.json and tokenizer_config.json.\n"
            "                          Defaults to the directory containing --model.\n"
            "  --hf DIR                Deprecated alias for --tokenizer.\n"
            "  --prompt-file FILE      Encode as Step-3.7 chat prompt.\n"
            "  --raw-prompt-file FILE  Encode as plain tokenizer text.\n"
            "  --tokens-file FILE      Whitespace-separated token ids.\n"
            "  --token ID              Repeat one token to ctx-max.\n"
            "\n"
            "Benchmark:\n"
            "  --backend cpu|cuda      Default cuda in CUDA builds, cpu otherwise.\n"
            "  --ctx-start N           First context frontier (default 1).\n"
            "  --ctx-max N             Last context frontier (default 1).\n"
            "  --ctx-alloc N           Session context allocation.\n"
            "  --step-incr N           Frontier increment (default ctx-max).\n"
            "  --gen-tokens N          Greedy decode tokens per frontier (default 1).\n"
            "  --decode-token ID       Repeat this token during generation and skip per-token logits.\n"
            "  --eos ID                Token to exclude during greedy decode.\n"
            "  --think none|on         Chat generation prefix (default on).\n"
            "  --csv FILE              Write CSV to file instead of stdout.\n"
            "  --warm-weights          Run one untimed token first to populate CUDA weight cache.\n"
            "  --verbose               Print model summary.\n",
            argv0);
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "sf37-bench: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static int parse_int_arg(const char *s, const char *opt, int min, int max) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || *end || v < min || v > max) {
        fprintf(stderr, "sf37-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static sf37_backend parse_backend(const char *s, const char *opt) {
    if (!strcmp(s, "cpu")) return SF37_BACKEND_CPU;
    if (!strcmp(s, "cuda")) return SF37_BACKEND_CUDA;
    fprintf(stderr, "sf37-bench: invalid value for %s: %s\n", opt, s);
    exit(2);
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "sf37-bench: open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "sf37-bench: seek %s failed\n", path);
        fclose(fp);
        exit(1);
    }
    long n = ftell(fp);
    if (n < 0) {
        fprintf(stderr, "sf37-bench: tell %s failed\n", path);
        fclose(fp);
        exit(1);
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "sf37-bench: rewind %s failed\n", path);
        fclose(fp);
        exit(1);
    }
    char *buf = malloc((size_t)n + 1u);
    if (!buf) {
        fprintf(stderr, "sf37-bench: out of memory reading %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (n > 0 && fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        fprintf(stderr, "sf37-bench: read %s failed\n", path);
        free(buf);
        fclose(fp);
        exit(1);
    }
    fclose(fp);
    buf[n] = '\0';
    return buf;
}

static void parse_tokens_file(const char *path, sf37_tokens *out) {
    char *text = read_file(path);
    const char *p = text;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
        if (!*p) break;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p || v < 0 || v > INT_MAX) {
            fprintf(stderr, "sf37-bench: invalid token id near: %.32s\n", p);
            free(text);
            exit(2);
        }
        sf37_tokens_push(out, (int)v);
        p = end;
    }
    free(text);
}

static bench_config parse_options(int argc, char **argv) {
#ifdef SF37_CPU_ONLY
    const sf37_backend default_backend = SF37_BACKEND_CPU;
#else
    const sf37_backend default_backend = SF37_BACKEND_CUDA;
#endif
    bench_config c = {
        .backend = default_backend,
        .ctx_start = 1,
        .ctx_max = 1,
        .step_incr = 0,
        .gen_tokens = 1,
        .repeat_token = -1,
        .decode_token = -1,
        .eos_token = -1,
        .think = true,
    };
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(argv[0]);
            exit(0);
        } else if (!strcmp(arg, "--model") || !strcmp(arg, "-m")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--hf") || !strcmp(arg, "--tokenizer")) {
            c.tokenizer_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--tokens-file")) {
            c.tokens_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            c.prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--raw-prompt-file")) {
            c.raw_prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--system")) {
            c.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--backend")) {
            c.backend = parse_backend(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-start")) {
            c.ctx_start = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1, 262144);
        } else if (!strcmp(arg, "--ctx-max")) {
            c.ctx_max = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1, 262144);
        } else if (!strcmp(arg, "--ctx-alloc")) {
            c.ctx_alloc = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1, 262144);
        } else if (!strcmp(arg, "--step-incr")) {
            c.step_incr = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 1, 262144);
        } else if (!strcmp(arg, "--gen-tokens") || !strcmp(arg, "-n")) {
            c.gen_tokens = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 0, 262144);
        } else if (!strcmp(arg, "--token")) {
            c.repeat_token = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 0, 128895);
        } else if (!strcmp(arg, "--decode-token")) {
            c.decode_token = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 0, 128895);
        } else if (!strcmp(arg, "--eos")) {
            c.eos_token = parse_int_arg(need_arg(&i, argc, argv, arg), arg, 0, 128895);
        } else if (!strcmp(arg, "--think")) {
            const char *v = need_arg(&i, argc, argv, arg);
            if (!strcmp(v, "on") || !strcmp(v, "yes") || !strcmp(v, "1")) c.think = true;
            else if (!strcmp(v, "none") || !strcmp(v, "off") || !strcmp(v, "0")) c.think = false;
            else {
                fprintf(stderr, "sf37-bench: --think must be on or none\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--csv")) {
            c.csv_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--warm-weights")) {
            c.warm_weights = true;
        } else if (!strcmp(arg, "--verbose")) {
            c.verbose = true;
        } else {
            fprintf(stderr, "sf37-bench: unknown option: %s\n", arg);
            usage(argv[0]);
            exit(2);
        }
    }
    if (!c.model_path) {
        fprintf(stderr, "sf37-bench: --model is required\n");
        exit(2);
    }
    if (c.ctx_start > c.ctx_max) {
        fprintf(stderr, "sf37-bench: --ctx-start must be <= --ctx-max\n");
        exit(2);
    }
    if (c.step_incr == 0) c.step_incr = c.ctx_max;
    if (c.ctx_alloc == 0) c.ctx_alloc = c.ctx_max + c.gen_tokens + 1;
    if (c.ctx_alloc <= c.ctx_max + c.gen_tokens) {
        fprintf(stderr, "sf37-bench: --ctx-alloc must be greater than ctx-max + gen-tokens\n");
        exit(2);
    }
    return c;
}

static int next_frontier(const bench_config *c, int cur) {
    if (cur >= c->ctx_max) return c->ctx_max;
    int next = cur + c->step_incr;
    if (next <= cur) next = c->ctx_max;
    if (next > c->ctx_max) next = c->ctx_max;
    return next;
}

static void build_prompt_tokens(sf37_engine *engine, const bench_config *cfg, sf37_tokens *prompt) {
    if (cfg->tokens_path) {
        parse_tokens_file(cfg->tokens_path, prompt);
        return;
    }
    if (cfg->repeat_token >= 0) {
        for (int i = 0; i < cfg->ctx_max; i++) sf37_tokens_push(prompt, cfg->repeat_token);
        return;
    }
    if (!sf37_tokenizer_ready(engine)) {
        fprintf(stderr, "sf37-bench: tokenizer is not loaded; pass --tokenizer, place tokenizer.json next to the GGUF, or use --tokens-file/--token\n");
        exit(1);
    }
    if (cfg->prompt_path) {
        char *text = read_file(cfg->prompt_path);
        sf37_encode_chat_prompt(engine, cfg->system, text,
                                cfg->think ? SF37_THINK_ENABLED : SF37_THINK_NONE,
                                prompt);
        free(text);
        return;
    }
    if (cfg->raw_prompt_path) {
        char *text = read_file(cfg->raw_prompt_path);
        sf37_tokenize_text(engine, text, prompt);
        free(text);
        return;
    }
    fprintf(stderr, "sf37-bench: specify --prompt-file, --raw-prompt-file, --tokens-file, or --token\n");
    exit(2);
}

int main(int argc, char **argv) {
    bench_config cfg = parse_options(argc, argv);
    sf37_engine_options opt = {
        .model_path = cfg.model_path,
        .tokenizer_path = cfg.tokenizer_path,
        .backend = cfg.backend,
        .inspect_only = false,
    };
    sf37_engine *engine = NULL;
    if (sf37_engine_open(&engine, &opt) != 0) return 1;
    sf37_engine_timing_info engine_timing = {0};
    sf37_engine_timing(engine, &engine_timing);
    if (cfg.verbose) sf37_engine_summary(engine, stderr);

    sf37_tokens prompt = {0};
    const double tokenizer_t0 = bench_now_sec();
    build_prompt_tokens(engine, &cfg, &prompt);
    const double tokenizer_t1 = bench_now_sec();
    const double tokenizer_sec = tokenizer_t1 - tokenizer_t0;
    if (prompt.len < cfg.ctx_max) {
        fprintf(stderr,
                "sf37-bench: prompt has %d tokens, need at least --ctx-max=%d\n",
                prompt.len, cfg.ctx_max);
        sf37_tokens_free(&prompt);
        sf37_engine_close(engine);
        return 1;
    }
    if (cfg.eos_token < 0) cfg.eos_token = sf37_token_eos(engine);

    sf37_session *session = NULL;
    if (sf37_session_create(&session, engine, cfg.ctx_alloc) != 0) {
        fprintf(stderr, "sf37-bench: failed to create session\n");
        sf37_tokens_free(&prompt);
        sf37_engine_close(engine);
        return 1;
    }
    if (cfg.warm_weights && prompt.len > 0) {
        char err[256];
        const double warm_t0 = bench_now_sec();
        if (sf37_session_eval(session, prompt.v[0], err, sizeof(err)) != 0) {
            fprintf(stderr, "sf37-bench: warm-weight token failed: %s\n", err);
            sf37_session_free(session);
            sf37_tokens_free(&prompt);
            sf37_engine_close(engine);
            return 1;
        }
        sf37_session_rewind(session, 0);
        const double warm_t1 = bench_now_sec();
        fprintf(stderr, "sf37-bench: warm-weight token complete in %.3fs\n", warm_t1 - warm_t0);
    }

    FILE *out = stdout;
    if (cfg.csv_path) {
        out = fopen(cfg.csv_path, "wb");
        if (!out) {
            fprintf(stderr, "sf37-bench: open %s: %s\n", cfg.csv_path, strerror(errno));
            sf37_session_free(session);
            sf37_tokens_free(&prompt);
            sf37_engine_close(engine);
            return 1;
        }
    }
    fprintf(out,
            "ctx_tokens,tokenizer_sec,model_map_sec,preload_sec,"
            "prefill_tokens,prefill_sec,prefill_tps,"
            "gen_tokens,decode_sec,lm_head_sec,sampling_sec,gen_sec,gen_tps,kvcache_bytes\n");
    fflush(out);

    char err[256];
    int previous = 0;
    int rc = 0;
    sf37_session_snapshot snap = {0};
    for (int frontier = cfg.ctx_start; ; frontier = next_frontier(&cfg, frontier)) {
        sf37_tokens prefix = {
            .v = prompt.v,
            .len = frontier,
            .cap = frontier,
        };
        const double prefill_t0 = bench_now_sec();
        if (sf37_session_sync(session, &prefix, err, sizeof(err)) != 0) {
            fprintf(stderr, "sf37-bench: prefill to %d failed: %s\n", frontier, err);
            rc = 1;
            break;
        }
        const double prefill_t1 = bench_now_sec();
        const int prefill_tokens = frontier - previous;

        if (cfg.gen_tokens > 0) {
            if (sf37_session_save_snapshot(session, &snap, err, sizeof(err)) != 0) {
                fprintf(stderr, "sf37-bench: snapshot at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        }

        double decode_sec = 0.0;
        double lm_head_sec = 0.0;
        double sampling_sec = 0.0;
        const double gen_t0 = bench_now_sec();
        for (int i = 0; i < cfg.gen_tokens; i++) {
            int token = cfg.decode_token;
            if (token < 0) {
                const double sample_t0 = bench_now_sec();
                token = sf37_session_argmax_excluding(session, cfg.eos_token);
                const double sample_t1 = bench_now_sec();
                sampling_sec += sample_t1 - sample_t0;
                if (token < 0) {
                    fprintf(stderr, "sf37-bench: failed to choose next token at frontier %d\n", frontier);
                    rc = 1;
                    break;
                }
            }
            const double decode_t0 = bench_now_sec();
            if (sf37_session_eval_no_logits(session, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "sf37-bench: decode at frontier %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
            const double decode_t1 = bench_now_sec();
            decode_sec += decode_t1 - decode_t0;

            if (cfg.decode_token >= 0) continue;
            const double lm_t0 = bench_now_sec();
            if (sf37_session_output_logits(session, err, sizeof(err)) != 0) {
                fprintf(stderr, "sf37-bench: lm_head at frontier %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
            const double lm_t1 = bench_now_sec();
            lm_head_sec += lm_t1 - lm_t0;
        }
        const double gen_t1 = bench_now_sec();
        if (rc != 0) break;

        if (cfg.gen_tokens > 0) {
            if (sf37_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) {
                fprintf(stderr, "sf37-bench: restore at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        } else {
            sf37_session_rewind(session, frontier);
        }

        const double prefill_sec = prefill_t1 - prefill_t0;
        const double gen_sec = gen_t1 - gen_t0;
        fprintf(out,
                "%d,%.6f,%.6f,%.6f,%d,%.6f,%.2f,%d,%.6f,%.6f,%.6f,%.6f,%.2f,%llu\n",
                frontier,
                tokenizer_sec,
                engine_timing.model_map_sec,
                engine_timing.preload_sec,
                prefill_tokens,
                prefill_sec,
                prefill_sec > 0.0 ? (double)prefill_tokens / prefill_sec : 0.0,
                cfg.gen_tokens,
                decode_sec,
                lm_head_sec,
                sampling_sec,
                gen_sec,
                gen_sec > 0.0 ? (double)cfg.gen_tokens / gen_sec : 0.0,
                (unsigned long long)(snap.len ? snap.len : sf37_session_kv_bytes(session)));
        fflush(out);

        previous = frontier;
        if (frontier >= cfg.ctx_max) break;
    }

    if (out != stdout) fclose(out);
    sf37_session_snapshot_free(&snap);
    sf37_session_free(session);
    sf37_tokens_free(&prompt);
    sf37_engine_close(engine);
    return rc;
}
