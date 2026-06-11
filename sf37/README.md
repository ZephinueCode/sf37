# SuperFast 3.7 for Step 3.7 Flash: Custom Quantized Inference Engine for Step 3.7 Flash

**SuperFast 3.7**, or **SF37**, is a narrow native inference engine for
**Step 3.7 Flash** and the custom `Q3GU-Q2D` SF37 GGUF quantization. It is
inspired by the DS4 engine style: model-specific runtime, custom CUDA kernels,
disk KV snapshots, OpenAI-compatible serving, Responses API serving, Anthropic
Messages serving, tool-call formatting, and practical local deployment defaults.

This is **alpha software**. It is not a general GGUF runner, and it is not meant
to load arbitrary Step/GGUF files. Use the matching SF37 GGUF and tokenizer files.

## Status

- CUDA decode and batch prefill are implemented.
- DGX Spark / GB10 CUDA build target is supported with `sm_121`.
- H20 / Hopper class CUDA builds are supported with `sm_90`.
- OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages server
  endpoints are implemented.
- Tool-call output, reasoning/content stream splitting, disk KV cache, session
  snapshots, and basic live tool memory are implemented.
- MTP is intentionally not implemented.
- Multimodal local JPEG/PNG input is implemented for CUDA: SF37 follows the
  official Step 3.7 image patcher, emits 504 patch crops plus the 728 main image,
  and runs the native CUDA vision encoder/projector. This is still alpha quality.

## Model

ModelScope repository:

```text
https://www.modelscope.cn/models/Zephinue/Step-3.7-Flash-MM-Q3GU-Q2D/
```

We recommend using the imatrix calibrated version ``imatrix-calib1m`` for deployment. Recommended directory layout:

```text
models/sf37/
  Step-3.7-Flash-MM-SF37-Q3GU-Q2D-imatrix-calib1m.gguf
  tokenizer.json
  tokenizer_config.json
```

When `tokenizer.json` and `tokenizer_config.json` are next to the GGUF, you only
need to pass `--model`. If they live elsewhere, pass `--tokenizer DIR`.

`--hf DIR` is accepted as a deprecated alias for `--tokenizer DIR`.

## Build

The default `make` target only prints help. Choose an explicit backend.

### CUDA, explicit architecture

Example for H20 / Hopper:

```sh
make cuda
```

### DGX Spark / GB10

Spark uses the `sm_121` target by default:

```sh
make cuda-spark
```

### Generic CUDA

```sh
make cuda-generic \
  NVCC=/path/to/nvcc \
  CUDA_HOME=/path/to/cuda
```

### CPU diagnostics

The CPU path is for inspection, tokenizer checks, and correctness diagnostics.
It is not the intended way to run this model.

```sh
make cpu
```

### Tests

```sh
make test

make cuda-test
```

## CLI

Inspect the GGUF and tokenizer:

```sh
./sf37 \
  --backend cpu \
  --inspect \
  --model models/sf37/Step-3.7-Flash-MM-SF37-Q3GU-Q2D.gguf
```

Generate with CUDA:

```sh
./sf37 \
  --backend cuda \
  --model models/sf37/Step-3.7-Flash-MM-SF37-Q3GU-Q2D.gguf \
  --prompt "Give me a short hello world C++ example." \
  --gen-tokens 256 \
  --ctx 8192 \
  --think on
```

Generate with a local image:

```sh
./sf37 \
  --backend cuda \
  --model models/sf37/Step-3.7-Flash-MM-SF37-Q3GU-Q2D.gguf \
  --prompt "Describe this image." \
  --image /path/to/image.png \
  --gen-tokens 256 \
  --ctx 4096 \
  --think on
```

`--image` accepts local JPEG/PNG paths, `file://` URLs, and `data:image/...;base64`
URLs. Online `http://` and `https://` image fetching is intentionally not
implemented. The lower-level debug path `--image-pixels-f32 FILE --image-size
728|504` is still available for pre-normalized NCHW float32 tensors.

If the tokenizer files are not next to the GGUF:

```sh
./sf37 \
  --backend cuda \
  --model /path/to/Step-3.7-Flash-MM-SF37-Q3GU-Q2D.gguf \
  --tokenizer /path/to/tokenizer_dir \
  --prompt "中国的首都是北京还是上海？" \
  --gen-tokens 128
```

## Server

Start a local server:

```sh
./sf37-server \
  --backend cuda \
  --model models/sf37/Step-3.7-Flash-MM-SF37-Q3GU-Q2D.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx 32768 \
  --max-tokens 2048 \
  --think on \
  --kv-disk-dir ./sf37-kv-cache \
  --kv-disk-space-mb 32768
```

If the tokenizer files are elsewhere, add:

```sh
--tokenizer /path/to/tokenizer_dir
```

Available endpoints:

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
POST /v1/responses
POST /v1/messages
```

OpenAI Chat Completions example:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "sf37",
    "messages": [
      {"role": "user", "content": "Write a tiny C++ hello world program."}
    ],
    "temperature": 0,
    "max_tokens": 512,
    "stream": true
  }'
```

OpenAI-compatible local image example:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "sf37",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this image."},
          {"type": "image_url", "image_url": {"url": "file:///path/to/image.png"}}
        ]
      }
    ],
    "temperature": 0,
    "max_tokens": 256,
    "stream": true
  }'
```

Responses `input_image` and Anthropic `image` blocks are also accepted when the
image source is a local path, `file://` URL, or `data:image/...;base64` URL.

OpenAI Responses example:

```sh
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "sf37",
    "input": "中国的首都是北京还是上海？",
    "temperature": 0,
    "max_output_tokens": 256,
    "stream": true
  }'
```

Anthropic Messages example:

```sh
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: local' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model": "sf37",
    "messages": [
      {"role": "user", "content": "Explain what a mutex is in one paragraph."}
    ],
    "max_tokens": 512,
    "temperature": 0,
    "stream": true
  }'
```

## Disk KV Cache

The server can persist prompt/session KV snapshots to disk. This is useful for
agent workflows that repeatedly resume large shared prefixes.

Common options:

```text
--kv-disk-dir DIR
--kv-disk-space-mb N
--kv-cache-min-tokens N
--kv-cache-cold-max-tokens N
--kv-cache-continued-interval-tokens N
```

For normal local use, start with:

```sh
--kv-disk-dir ./sf37-kv-cache --kv-disk-space-mb 32768
```

## Performance Notes

- Spark defaults are conservative: limited startup preload plus layer-aware
  on-demand residency, not full model copy.
- Long prompt speed comes from CUDA batch prefill.
- Greedy decode avoids full logits CPU readback by doing the lm_head argmax on
  GPU and reading back only the selected token.
- BF16 `lm_head`, embeddings, and vision projector are preserved for quality.

Useful profiling toggles:

```sh
SF37_CUDA_WEIGHT_CACHE_VERBOSE=1
SF37_CUDA_PREFILL_PROFILE=1
SF37_CUDA_MOE_PROFILE=1
```

Fallback toggles are available for debugging:

```sh
SF37_CUDA_NO_BATCH_PREFILL=1
SF37_CUDA_NO_BATCH_ATTENTION=1
SF37_CUDA_NO_BATCH_ROUTER=1
SF37_CUDA_NO_BATCH_MOE=1
SF37_CUDA_NO_QLOW_Q8K=1
SF37_CUDA_NO_DIRECT_MOE_DOWN_SUM=1
SF37_CUDA_NO_LM_HEAD_GPU_ARGMAX=1
```

## Development

Clean build products before packaging or publishing source:

```sh
make clean
```

This removes local binaries, object files, and test executables.
