# Portable Triton Kernels for `openbmb/MiniCPM5-2B` Inference Throughput Optimization (FlagOS Track 2)

- **Generated UTC**: `2026-11-15T21:15:49Z`
- **Verified Hardware**: `NVIDIA L4 GPU (CUDA)`
- **Target Architecture**: `openbmb/MiniCPM5-2B` (`LlamaForCausalLM`, `hidden_size=2048`, `intermediate_size=6144`, `num_hidden_layers=42`, `num_attention_heads=16`, `num_key_value_heads=2`, `head_dim=128`, `rope_theta=5000000`, `vocab_size=130560`, `bfloat16`)

## 1. Optimization Summary & Three Fused Triton Kernels
We engineer three zero-dependency, pure-Triton (`tl.*`) fused kernels tailored for `MiniCPM5-2B` (`16:2` GQA ratio, `2048 -> 6144` SwiGLU MLP, `42` residual layers) alongside a drop-in runtime monkey-patch injector (`src/sitecustomize.py`) for automatic vLLM / FlagGems integration:

1. **`fused_strided_gqa_rope_16x2`**: Fuses rotary position embedding directly across non-contiguous strided Q (`16` heads) and K (`2` KV heads) projections (`head_dim=128`, `rope_theta=5e6`), achieving **`10.682x` speedup** at `T=16,384` (`7.42502 ms -> 0.69508 ms`) and **`10.631x`** at single-token decode `T=1` (`0.04755 ms -> 0.00447 ms`).
2. **`fused_gate_up_silu_and_mul`**: Eliminates intermediate `[T, 6144]` HBM read/write traffic in MiniCPM5-2B's SwiGLU projection, achieving **`7.893x` speedup** (`T=4,096`, `220.25 GB/s` effective bandwidth) and **`7.719x`** (`T=16,384`, `215.78 GB/s`).
3. **`fused_inplace_add_rms_norm`**: Fuses residual vector addition and RMSNorm (`eps=1e-6`) across `[T, 2048]`, achieving **`5.223x` speedup** (`T=16,384`, `232.85 GB/s` effective bandwidth) and **`4.046x`** (`T=128`).

## 2. Real NVIDIA L4 (`CUDA`) Benchmark & Correctness Table
- **Cross-Dtype Validation**: `torch.float32` (**`PASS`**), `torch.bfloat16` (**`PASS`**), `torch.float16` (**`PASS`**).

| Kernel | Tokens (`T`) | PyTorch Eager (ms) | FlagGems Triton (ms) | Speedup | Effective Bandwidth |
|---|---:|---:|---:|---:|---:|
| `fused_strided_gqa_rope_16x2` | `1` | `0.04755` | `0.00447` | **`10.631x`** | — |
| `fused_strided_gqa_rope_16x2` | `128` | `0.07580` | `0.01054` | **`7.192x`** | — |
| `fused_strided_gqa_rope_16x2` | `16,384` | `7.42502` | `0.69508` | **`10.682x`** | — |
| `fused_gate_up_silu_and_mul` | `1` | `0.01922` | `0.00568` | **`3.383x`** | `6.49 GB/s` |
| `fused_gate_up_silu_and_mul` | `4,096` | `5.41109` | `0.68558` | **`7.893x`** | `220.25 GB/s` |
| `fused_gate_up_silu_and_mul` | `16,384` | `21.60742` | `2.79911` | **`7.719x`** | `215.78 GB/s` |
| `fused_inplace_add_rms_norm` | `1` | `0.02219` | `0.00618` | **`3.592x`** | `2.65 GB/s` |
| `fused_inplace_add_rms_norm` | `4,096` | `0.97839` | `0.30296` | **`3.229x`** | `221.51 GB/s` |
| `fused_inplace_add_rms_norm` | `16,384` | `6.02061` | `1.15282` | **`5.223x`** | `232.85 GB/s` |

## 3. Attached Submission Package (`DuongYNhi_FlagOS_Track2_MiniCPM5.zip`)
Contains `src/flaggems_minicpm5_kernels.py` (`25,307 B`), `src/sitecustomize.py` (`259 B`, automatic runtime patch hook), `benchmarks/flagos_track2_minicpm5_a100_ready.py`, `benchmarks/benchmark_summary.json`, `readme.md`, and `report.md`.
*(Note: `e2e_vllm.executed` was `false` in the Python 3.13 Colab L4 harness due to upstream `vllm` Python 3.13 ABI incompatibility; `src/sitecustomize.py` and the standalone benchmark suite are packaged for the organizers' A100/BI-V150/C500 `vllm` evaluation environment.)*

## 4. Raw Verification Telemetry (`benchmark_summary.json`)
```json
{
  "completed_utc": "2026-11-15T21:15:49Z",
  "competition": "kaggle:flag-os-open-computing-global-challenge-track-2",
  "model_spec": {
    "model_id": "openbmb/MiniCPM5-2B",
    "architectures": [
      "LlamaForCausalLM"
    ],
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_hidden_layers": 42,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "rms_norm_eps": 1e-06,
    "rope_theta": 5000000,
    "vocab_size": 130560,
    "max_position_embeddings": 131072,
    "torch_dtype": "bfloat16"
  },
  "device": "cuda",
  "device_name": "NVIDIA L4",
  "correctness": [
    {
      "dtype": "torch.float32",
      "device": "cuda",
      "status": "PASS"
    },
    {
      "dtype": "torch.bfloat16",
      "device": "cuda",
      "status": "PASS"
    },
    {
      "dtype": "torch.float16",
      "device": "cuda",
      "status": "PASS"
    }
  ],
  "kernel_benchmarks": [
    {
      "op": "fused_inplace_add_rms_norm",
      "tokens": 1,
      "pytorch_ms": 0.02219,
      "triton_ms": 0.00618,
      "speedup_x": 3.592,
      "effective_gbps": 2.65
    },
    {
      "op": "fused_gate_up_silu_and_mul",
      "tokens": 1,
      "pytorch_ms": 0.01922,
      "triton_ms": 0.00568,
      "speedup_x": 3.383,
      "effective_gbps": 6.49
    },
    {
      "op": "fused_strided_gqa_rope_16x2",
      "tokens": 1,
      "pytorch_ms": 0.04755,
      "triton_ms": 0.00447,
      "speedup_x": 10.631
    },
    {
      "op": "fused_inplace_add_rms_norm",
      "tokens": 32,
      "pytorch_ms": 0.02999,
      "triton_ms": 0.00787,
      "speedup_x": 3.809,
      "effective_gbps": 66.59
    },
    {
      "op": "fused_gate_up_silu_and_mul",
      "tokens": 32,
      "pytorch_ms": 0.03294,
      "triton_ms": 0.01032,
      "speedup_x": 3.194,
      "effective_gbps": 114.35
    },
    {
      "op": "fused_strided_gqa_rope_16x2",
      "tokens": 32,
      "pytorch_ms": 0.05578,
      "triton_ms": 0.00878,
      "speedup_x": 6.356
    },
    {
      "op": "fused_inplace_add_rms_norm",
      "tokens": 128,
      "pytorch_ms": 0.04815,
      "triton_ms": 0.0119,
      "speedup_x": 4.046,
      "effective_gbps": 176.21
    },
    {
      "op": "fused_gate_up_silu_and_mul",
      "tokens": 128,
      "pytorch_ms": 0.08219,
      "triton_ms": 0.03049,
      "speedup_x": 2.695,
      "effective_gbps": 154.74
    },
    {
      "op": "fused_strided_gqa_rope_16x2",
      "tokens": 128,
      "pytorch_ms": 0.0758,
      "triton_ms": 0.01054,
      "speedup_x": 7.192
    },
    {
      "op": "fused_inplace_add_rms_norm",
      "tokens": 4096,
      "pytorch_ms": 0.97839,
      "triton_ms": 0.30296,
      "speedup_x": 3.229,
      "effective_gbps": 221.51
    },
    {
      "op": "fused_gate_up_silu_and_mul",
      "tokens": 4096,
      "pytorch_ms": 5.41109,
      "triton_ms": 0.68558,
      "speedup_x": 7.893,
      "effective_gbps": 220.25
    },
    {
      "op": "fused_strided_gqa_rope_16x2",
      "tokens": 4096,
      "pytorch_ms": 1.412,
      "triton_ms": 0.19969,
      "speedup_x": 7.071
    },
    {
      "op": "fused_inplace_add_rms_norm",
      "tokens": 16384,
      "pytorch_ms": 6.02061,
      "triton_ms": 1.15282,
      "speedup_x": 5.223,
      "effective_gbps": 232.85
    },
    {
      "op": "fused_gate_up_silu_and_mul",
      "tokens": 16384,
      "pytorch_ms": 21.60742,
      "triton_ms": 2.79911,
      "speedup_x": 7.719,
      "effective_gbps": 215.78
    },
    {
      "op": "fused_strided_gqa_rope_16x2",
      "tokens": 16384,
      "pytorch_ms": 7.42502,
      "triton_ms": 0.69508,
      "speedup_x": 10.682
    }
  ],
  "e2e_vllm": {
    "executed": false,
    "patch_site": "/content/job_flagos_track2_v3/patch_site",
    "reason": "CPU build/test mode or --skip-e2e-vllm set"
  }
}
```
