# SF37 Imatrix Calibration

SF37 imatrix calibration targets the routed MoE tensors only:

- `model.layers.N.moe.gate_proj.weight`
- `model.layers.N.moe.up_proj.weight`
- `model.layers.N.moe.down_proj.weight`

Gate/up entries collect squared FFN-normalized activations. Down entries collect
the squared routed SwiGLU mid row after router weighting. The `.dat` file uses
the llama.cpp legacy imatrix layout, with one packed entry per routed tensor:

```text
entry length = n_expert * n_columns
```

## Build A Small Calibration Dataset

```sh
python3 gguf-tools/imatrix/dataset/build_sf37_imatrix_dataset.py \
  --out-dir gguf-tools/imatrix/dataset \
  --tokenizer ../checkpoint/bf16 \
  --max-prompts 600 \
  --max-tokens 800000 \
  --shards-per-source 1
```

The default sources are:

- code: `/mnt/share-bos/data/core-data-share/text_sft/coding/OpenCodeReasoning-2_wds`
- IF/English: `/mnt/share-bos/data/core-data-share/text_sft/mimo7bsftr10515v3amthinkingif`
- Chinese: `/mnt/share-bos/data/core-data-share/text_sft/chinese-distill-glm`

Use a smoke dataset first:

```sh
python3 gguf-tools/imatrix/dataset/build_sf37_imatrix_dataset.py \
  --out-dir /tmp/sf37-imatrix-smoke \
  --max-prompts 30 \
  --max-tokens 32768 \
  --shards-per-source 1
```

## Collect Imatrix

Collection is CUDA-only and uses SF37's layer-aware mapped prefill path. The
CLI applies conservative streaming defaults for `--imatrix-*` collection:

```text
SF37_CUDA_STARTUP_PRELOAD_GB=0
SF37_CUDA_WEIGHT_CACHE_LIMIT_GB=16
SF37_CUDA_NO_PREFILL_Q8_CACHE=1
SF37_CUDA_NO_Q8_F16_CACHE=1
SF37_CUDA_NO_Q8_F32_CACHE=1
```

The collector also evicts model-weight cache after every layer. This keeps H20
calibration from turning into a full 16-bit or full expanded-cache residency
run.

For H20 calibration, the faster recommended mode is still startup-streamed and
still keeps expanded Q8 caches disabled, but it lets the already-quantized GGUF
weights stay warm under an 84 GiB cache budget. This does not load the BF16
checkpoint into GPU memory:

```sh
SF37_CUDA_IMATRIX_KEEP_WEIGHT_CACHE=1 ./sf37 ...
```

Override `SF37_CUDA_WEIGHT_CACHE_LIMIT_GB` if the local device needs a tighter
or looser cache budget.

```sh
./sf37 \
  --backend cuda \
  --model ../quant/Step-3.7-Flash-MM-SF37-Q3GU-Q2D.gguf \
  --tokenizer ../checkpoint/bf16 \
  --ctx 8192 \
  --imatrix-dataset gguf-tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out ../quant/imatrix/Step-3.7-Flash-MM-SF37-Q3GU-Q2D-imatrix-v1.dat \
  --imatrix-max-prompts 600 \
  --imatrix-max-tokens 800000
```

Useful smoke:

```sh
./sf37 \
  --backend cuda \
  --model ../quant/Step-3.7-Flash-MM-SF37-Q3GU-Q2D.gguf \
  --tokenizer ../checkpoint/bf16 \
  --ctx 4096 \
  --imatrix-dataset /tmp/sf37-imatrix-smoke/rendered_prompts.txt \
  --imatrix-out /tmp/sf37-smoke.imatrix.dat \
  --imatrix-max-prompts 2 \
  --imatrix-max-tokens 4096
```

## Regenerate GGUF

```sh
cd gguf-tools
make

./sf37-quantize \
  --hf /mnt/share-bos/user/wangxiaoce/step3_7/checkpoint/bf16 \
  --out /mnt/share-bos/user/wangxiaoce/step3_7/quant/Step-3.7-Flash-MM-SF37-Q3GU-Q2D-imatrix-v1.gguf \
  --imatrix /mnt/share-bos/user/wangxiaoce/step3_7/quant/imatrix/Step-3.7-Flash-MM-SF37-Q3GU-Q2D-imatrix-v1.dat \
  --threads 24 \
  --overwrite
```

The output keeps the same runtime format and size class as the alpha GGUF:
Q3 asym gate/up, Q2 asym down, Q8_0 rank-2 tensors, and BF16 embedding/lm_head
/ vision projector.
