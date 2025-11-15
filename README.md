# Int4 Triton Kernels

> tl;dr: we got 4× weight memory savings with per-pack int4 quantization, a working BF16×Int4 GEMM in Triton, and honest benchmarks vs cuBLAS. \
Spoiler: cuBLAS is still a monster.

---

## Int4 Quantization & Packing

First step was to teach the model weights how to live on a diet.

In [`quant_i4_pack8.py`](triton_int4/triton_kernels/quant_i4_pack8.py) we implemented:

- quantization of a 2D FP16 matrix to int4 with one scale per group of 8 values (per-pack quantization),
- packing of 8 int4 values into a single `int32` (or into `int8` when needed).

This gives a clean **4× reduction in weight memory**: FP16 → packed int4 + scales.

Roundtrip and memory behavior are checked in [`tests/test_quant.py`](tests/test_quant.py):

- we verify that the packed tensor uses ~25% of the original bytes,
- we dequantize back and assert the mean error is small/consistent for random FP16 inputs.

Net result: a lightweight int4 “codec” for model weights that is easy to plug into other components.

---

## BF16 × Int4 Matmul (X16 @ W4ᵀ)

Next, we needed to actually *use* those compressed weights.

In [`matmul_bf16_i4_pack8.py`](triton_int4/triton_kernels/matmul_bf16_i4_pack8.py) we wrote a Triton kernel that:

- takes BF16 activations,
- reads the int4-packed weight matrix plus per-pack scales,
- unpacks and dequantizes weights on the fly inside the matmul loop,
- accumulates the result in FP32.

This is wrapped in `matmul_bf16_i4`, which is also used by a quantized linear layer [`Int4PackedLinear`](triton_int4/quant_layer.py). That layer takes an existing `nn.Linear`, quantizes its FP16 weights once at init time, and then replaces the matmul with the BF16×Int4 kernel in `forward`.

Correctness is again validated in [`tests/test_quant.py`](tests/test_quant.py):

- comparing kernel output vs a reference path that explicitly dequantizes and calls `torch.matmul`,
- checking an end-to-end `Int4PackedLinear` against the original `nn.Linear` (within an error budget appropriate for int4).

So at this point we have a full pipeline: FP16 weights → packed int4 + scales → BF16×Int4 GEMM.

---

## Benchmarks vs FP16 GEMM (LLaMA-3.2-1B Shapes)

Finally, we benchmarked:

- **Int4 path**: `(X16 @ W4ᵀ)` via our Triton kernel `matmul_bf16_i4`,
- **FP16/BF16 baseline**: `(X16 @ W16ᵀ)` via `torch.matmul` (cuBLAS).

Benchmarking is in [`bench.py`](triton_int4/bench.py). Shapes are taken from the config of `unsloth/Llama-3.2-1B-Instruct`:

- typical linear layers with sizes like `(2048, 2048)`, `(2048, 8192)`, `(8192, 2048)`,
- number of tokens (rows in X): `M ∈ {128, 512, 2048}`.

Weights `W` are random FP16, quantized to int4 with `quantize_i4_pack8`. Activations `X` are random BF16. Both kernels are warmed up and timed with CUDA events.

### H100 Results

Final numbers on an H100 GPU:

| Tokens (M) | Out (N) | In (K) | Int4 GEMM (ms) | FP16 GEMM (ms) | Speedup (FP16 / Int4) |
|-----------:|--------:|-------:|----------------:|----------------:|-----------------------:|
| 128        | 2048    | 2048   | 0.0807          | 0.0209          | 0.26×                  |
| 512        | 2048    | 2048   | 0.1399          | 0.0211          | 0.15×                  |
| 2048       | 2048    | 2048   | 0.5001          | 0.0534          | 0.11×                  |
| 128        | 2048    | 8192   | 0.3133          | 0.0285          | 0.09×                  |
| 512        | 2048    | 8192   | 0.5450          | 0.0512          | 0.09×                  |
| 2048       | 2048    | 8192   | 2.0084          | 0.1503          | 0.07×                  |
| 128        | 8192    | 2048   | 0.1407          | 0.0234          | 0.17×                  |
| 512        | 8192    | 2048   | 0.5025          | 0.0460          | 0.09×                  |
| 2048       | 8192    | 2048   | 1.8509          | 0.1616          | 0.09×                  |

So yes, the int4 path gives **4× smaller weights**, but on raw matmul speed it’s currently **slower than cuBLAS BF16/FP16** by about 4-15×, depending on the shape.

Which is exactly what you expect when:

- cuBLAS is tuned to death for BF16/FP16 Tensor Cores,
- and your custom kernel is doing on-the-fly bit unpacking and scaling for int4 without using native int4 Tensor Core instructions or hyper-optimized layouts.

From here, the interesting part is not “can we beat cuBLAS in three files of Triton”, but “what does this quantization scheme look like end-to-end when plugged into LLaMA’s linear layers and measured on perplexity + throughput” - that’s what the next tasks will cover.

## Int4 Linear Layers in LLaMA-3.2-1B

After the kernels were in place, the next step was to actually *swap them into a real model*.


In [`quant_layer.py`](triton_int4/quant_layer.py):

- `Int4PackedLinear` wraps an existing `nn.Linear`, quantizes its FP16 weights with [`quantize_i4_pack8`](triton_int4/triton_kernels/quant_i4_pack8.py), and uses [`matmul_bf16_i4`](triton_int4/triton_kernels/matmul_bf16_i4_pack8.py) in `forward`.
- `replace_linear_with_int4` walks the module tree and replaces every `torch.nn.Linear` with `Int4PackedLinear`.

Applied to `unsloth/Llama-3.2-1B-Instruct`, all dense layers (attention and MLP projections) are now int4-packed and computed through the custom Triton GEMM; everything else in the model stays the same.

---

## WikiText-2 Perplexity & Speed

To evaluate the quantized model end-to-end, [`eval_wikitext2.py`](triton_int4/eval_wikitext2.py) runs WikiText-2 (`wikitext-2-raw-v1`, `test` split) in next-token-prediction mode:

- sequences are chunked to `seq_len=256`,
- batch size = 4,
- metrics: perplexity and tokens per second (CUDA-timed).

Final numbers (Triton autotune **disabled** for steady-state speed):

| mode | seq_len | batch_size | perplexity | tokens/s  |
|------|--------:|-----------:|-----------:|----------:|
| fp16 |     256 |          4 |   21.0905  | 32711.13  |
| int4 |     256 |          4 |   22.9485  |  8438.13  |

So with all linears quantized to int4 we get ~4× smaller weights, about **+9%** worse perplexity on WikiText-2, and an end-to-end throughput that is roughly **3.9× slower** than the original fp16 model on this setup.
