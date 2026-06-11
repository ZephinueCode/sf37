# SF37 GGUF tools

`sf37-quantize` is the Step-3.7-Flash HF-safetensors to GGUF writer used by
the SF37 engine prototype.

It currently writes a model-specific GGUF with HF tensor names and reversed
GGUF dimensions. The quantization policy is:

- routed MoE `gate_proj` and `up_proj`: SF37 asymmetric Q3, stored under GGUF
  type id `Q3_K` with an SF37-specific byte layout;
- routed MoE `down_proj`: SF37 asymmetric Q2, stored under GGUF type id
  `Q2_K` with an SF37-specific byte layout;
- rank-2 non-routed matrices: `Q8_0`;
- `lm_head.weight`, `model.embed_tokens.weight`, and `vit_large_projector.weight`:
  BF16;
- routers, norms, biases, F32 tensors, and non-matrix tensors: original dtype.

Build:

```sh
make -C step3_7/sf37/gguf-tools
```

Dry run:

```sh
step3_7/sf37/gguf-tools/sf37-quantize \
  --hf step3_7/checkpoint/bf16 \
  --out step3_7/quant/Step-3.7-Flash-MM-SF37-Q3GU-Q2D.gguf \
  --dry-run
```

Full write:

```sh
step3_7/sf37/gguf-tools/sf37-quantize \
  --hf step3_7/checkpoint/bf16 \
  --out step3_7/quant/Step-3.7-Flash-MM-SF37-Q3GU-Q2D.gguf
```
