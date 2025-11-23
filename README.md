# Int4 Triton Kernels

> tl;dr: we got ≈4× weight memory savings with group-wise int4 quantization, a working BF16×Int4 GEMM in Triton, and honest benchmarks vs cuBLAS. \
Spoiler: cuBLAS is still a monster.

---

## Quantization Schemes: Row-wise vs Group-wise

There is a small playground for different int4 schemes in
[`quant_schemes.py`](triton_int4/quant/quant_schemes.py) and
[`quant_bench.py`](triton_int4/quant/quant_bench.py):

- **Symmetric row-wise** int4 (one scale per row),
- **Asymmetric row-wise** int4 (scale + zero-point per row),
- **Symmetric group-wise** int4 with different group sizes `G` along the columns.

For a representative LLaMA-3.2-1B weight matrix of shape **2048×8192**, we get:

| Scheme                    | MAE    | Memory reduction vs FP16 |
|---------------------------|--------|--------------------------|
| sym row-wise              | 0.1415 | 4.00×                    |
| asym row-wise             | 0.1265 | 3.99×                    |
| sym group-wise (G = 8)    | 0.0558 | 2.00×                    |
| sym group-wise (G = 128)  | 0.1002 | 3.76×                    |
| sym group-wise (G = 256)  | 0.1083 | 3.88×                    |
| sym group-wise (G = 512)  | 0.1158 | 3.94×                    |

So there’s a clear trade-off:

- small groups -> better MAE, but more metadata (less compression),
- large groups / row-wise -> slightly worse MAE, but **≈4×** compression including scales.

The main pipeline uses **symmetric group-wise int4 with `G = 256`** as a compromise: almost 4× smaller than FP16, noticeably better MAE than pure row-wise.

---

## Int4 Quantization & Packing

In [`quant_i4_pack8.py`](triton_int4/triton_kernels/quant_i4_pack8.py) we implement:

- symmetric **group-wise** quantization of a 2D FP16 matrix to int4 with one scale per group of `G` columns (default `G = 256`),
- packing of 8 int4 values into a single `int32` (or into `int8` when needed).

The packed weight tensor itself uses ~25% of the original FP16 bytes (4× smaller).
With group-wise scales (`G = 256`) we get **≈3.9–4.0× total weight memory reduction** on LLaMA-3.2-1B-style shapes.

Roundtrip and memory behavior are checked in [`tests/test_quant.py`](tests/test_quant.py):

- we verify that the packed weight tensor uses ~25% of the original bytes,
- we dequantize back and assert the mean error is small/consistent for random FP16 inputs.

Net result: a compact int4 “codec” for model weights that is easy to plug into other components.

---

## BF16 × Int4 Matmul (X16 @ W4ᵀ)

In [`matmul_bf16_i4_pack8.py`](triton_int4/triton_kernels/matmul_bf16_i4_pack8.py) we provide a Triton kernel that:

- takes BF16 activations,
- reads the int4-packed weight matrix plus **group-wise** scales,
- unpacks and dequantizes weights on the fly inside the matmul loop,
- accumulates the result in FP32.

This is wrapped in `matmul_bf16_i4`, which is also used by the quantized linear layer [`Int4PackedLinear`](triton_int4/quant_layer.py). That layer takes an existing `nn.Linear`, quantizes its FP16 weights once at init time, and then replaces the matmul with the BF16×Int4 kernel in `forward`.

Correctness is validated in [`tests/test_quant.py`](tests/test_quant.py):

- comparing kernel output vs a reference path that explicitly dequantizes and calls `torch.matmul`,
- checking an end-to-end `Int4PackedLinear` against the original `nn.Linear` (within an error budget appropriate for int4).

So we have a full pipeline: FP16 weights -> packed int4 + group-wise scales -> BF16×Int4 GEMM.

---

## Benchmarks vs FP16 GEMM (LLaMA-3.2-1B Shapes)

We benchmarked:

- **Int4 path**: `(X16 @ W4ᵀ)` via `matmul_bf16_i4`,
- **FP16/BF16 baseline**: `(X16 @ W16ᵀ)` via `torch.matmul` (cuBLAS).

Benchmarking is in [`bench.py`](triton_int4/bench.py). Shapes are taken from the config of `unsloth/Llama-3.2-1B-Instruct`:

- typical linear layers with sizes like `(2048, 2048)`, `(2048, 8192)`, `(8192, 2048)`,
- number of tokens (rows in X): `M ∈ {128, 512, 2048}`.

Weights `W` are random FP16, quantized to int4 with `quantize_i4_pack8`. Activations `X` are random BF16. Both kernels are warmed up and timed with CUDA events.

### H100 Results

Final numbers on an H100 GPU:

| Tokens (M) | Out (N) | In (K) | Int4 GEMM (ms) | FP16 GEMM (ms) | Speedup (FP16 / Int4) |
|-----------:|--------:|-------:|----------------:|----------------:|-----------------------:|
| 128        | 2048    | 2048   | 0.0760          | 0.0273          | 0.36×                  |
| 512        | 2048    | 2048   | 0.1208          | 0.0271          | 0.22×                  |
| 2048       | 2048    | 2048   | 0.4350          | 0.0544          | 0.12×                  |
| 128        | 2048    | 8192   | 0.3089          | 0.0287          | 0.09×                  |
| 512        | 2048    | 8192   | 0.5330          | 0.0498          | 0.09×                  |
| 2048       | 2048    | 8192   | 1.9880          | 0.1515          | 0.08×                  |
| 128        | 8192    | 2048   | 0.1211          | 0.0269          | 0.22×                  |
| 512        | 8192    | 2048   | 0.4361          | 0.0457          | 0.10×                  |
| 2048       | 8192    | 2048   | 1.6819          | 0.1629          | 0.10×                  |

So the int4 path gives **≈4× smaller weights**, but on raw matmul speed it is currently **slower than cuBLAS BF16/FP16** by roughly **3–12×**, depending on the shape.

This is pretty expected:

- cuBLAS is heavily tuned for BF16/FP16 Tensor Cores,
- our custom kernel does on-the-fly bit unpacking + scaling for int4, without using native int4 Tensor Core instructions or specialized layouts.

---

## Int4 Linear Layers in LLaMA-3.2-1B

In [`quant_layer.py`](triton_int4/quant_layer.py):

- `Int4PackedLinear` wraps an existing `nn.Linear`, quantizes its FP16 weights with [`quantize_i4_pack8`](triton_int4/triton_kernels/quant_i4_pack8.py), and uses [`matmul_bf16_i4`](triton_int4/triton_kernels/matmul_bf16_i4_pack8.py) in `forward`.
- `replace_linear_with_int4` walks the module tree and replaces every `torch.nn.Linear` with `Int4PackedLinear`.

Applied to `unsloth/Llama-3.2-1B-Instruct`, all dense layers (attention and MLP projections) are now int4-packed and computed through the custom Triton GEMM; everything else in the model stays the same.

---

## WikiText-2 Perplexity & Speed

To evaluate the quantized model end-to-end, [`eval_wikitext2.py`](triton_int4/eval_wikitext2.py) runs WikiText-2 (`wikitext-2-raw-v1`, `test` split) in next-token-prediction mode:

- sequences are chunked to `seq_len = 256`,
- batch size = 4,
- metrics: perplexity and tokens per second (CUDA-timed),
- Triton autotune is disabled to measure steady-state kernel performance.

Final numbers:

| mode | seq_len | batch_size | perplexity | tokens/s  |
|------|--------:|-----------:|-----------:|----------:|
| fp16 |     256 |          4 |   21.0905  | 27879.49  |
| int4 |     256 |          4 |   29.0775  |  9749.81  |

With all linears quantized to int4 we get **≈4× smaller weights** (for the chosen group-wise scheme), a higher perplexity on WikiText-2 (about **+38%** relative), and an end-to-end throughput that is roughly **2.9× slower** than the original fp16 model on this setup.
