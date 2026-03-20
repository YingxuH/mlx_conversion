# MLX MERaLiON Evaluation Results

Evaluation of the MLX Apple Silicon port of MERaLiON-2 AudioLLM against the reference HuggingFace transformers implementation on 5 ASR datasets.

## Setup

- **Reference**: HuggingFace MERaLiON-2-10B (bfloat16, CUDA/CPU)
- **MLX**: MERaLiON-2-10B 4-bit quantized (mlx-lm compatible, group_size=64)
- **Hardware**: Apple M4 Pro, 24GB RAM
- **Prompt format**: `"Instruction: Please transcribe this speech. \nFollow the text instruction based on the following audio: <SpeechHere>"` (matches HF eval training format)

## WER Results

| Dataset | HF 10B bf16 | MLX 10B-4bit | Relative diff | Method | N (evaluated) |
|---|---|---|---|---|---|
| idpc_short_ASR_v2 | 0.1488 | 0.1783 | +19.9% | multi-chunk | 32 |
| ytb_asr_batch1 | 0.0973 | 0.0750 | **-22.9%** | multi-chunk | 32 |
| ytb_asr_batch2 | 0.1129 | 0.1290 | +14.3% | multi-chunk | 473 (full) |
| ytb_asr_batch3_malay | 0.2059 | 0.3247 | +57.7% | multi-chunk | 200 (full) |
| ytb_asr_batch3_tamil_v2 | 0.4564 | 0.6612 | +44.9% | smart-chunk | 184 (full) |

### MERaLiON-2-3B (MLX, fp16) — for reference

| Dataset | MLX 3B | N |
|---|---|---|
| idpc_short_ASR_v2 | 0.3211 | 32 |
| ytb_asr_batch1 | 0.1418 | 32 |
| ytb_asr_batch2 | 0.2210 | 473 (full) |
| ytb_asr_batch3_malay | 0.8467 | 200 (full) |
| ytb_asr_batch3_tamil_v2 | 1.0111 | 184 (full) |

## Key Findings

### English ASR: MLX matches HF well
- `ytb_asr_batch2` (473 samples, all single 30s clips): MLX 4-bit WER **0.1290** vs HF **0.1129** — only 14% relative worse
- `ytb_asr_batch1`: MLX actually **better** (0.0750 vs 0.0973), likely sample variance on 32 samples
- English ASR quality is largely preserved under 4-bit quantization

### Malay: Degraded but functional
- Multi-chunk WER **0.3247** vs HF **0.2059** (57% relative degradation)
- Malay clips average 55s (range 32–95s), all require multi-chunk processing
- 4-bit quantization introduces systematic errors in non-English tokens

### Tamil: Significant degradation
- Smart-chunk WER **0.6612** vs HF **0.4564** (45% relative degradation)
- Tamil clips average 94s (range 30–315s); transcripts average 806 tokens (40% exceed 512 tokens)
- Two compounding factors: quantization quality loss + long audio processing challenges
- 3B model essentially fails (WER >1.0)

## Inference Strategy

### Problem: Short-chunk Hallucination
The 4-bit quantized model hallucinates when given mostly-silent padded audio (e.g., a 4s clip padded to 30s). Unlike the bf16 model, it cannot reliably generate EOS and instead produces repetitive hallucinated text.

HF eval processes each 30s chunk independently, which works fine with bf16 but fails with 4-bit.

### Solution: Smart Chunking
```
infer_one(audio_array):
  if len(audio) ≤ 30s:
      → single-chunk inference (identical to HF)
  else:
      → split at 30s boundaries
      → if last segment < 10s: merge it into the preceding segment
      → run inference on each segment independently
      → concatenate text outputs
```

The short-tail merge prevents short-chunk hallucination while keeping segments manageable.

### When to use multi-chunk vs smart-chunk

| Scenario | Recommended | Reason |
|---|---|---|
| Audio ≤ 30s | N/A (same) | Single chunk |
| Audio 30–90s (e.g., Malay) | Multi-chunk | Avoids short-tail; transcripts fit in 512 tokens |
| Audio >90s (e.g., Tamil) | Smart-chunk | Multi-chunk truncates at 512 tokens (Tamil avg 806 tokens) |

The current `eval_mlx_asr.py` default uses smart-chunking (independent 30s segments with short-tail merge), which is the safer general-purpose choice.

### Performance
- 10B-4bit model: ~22s per sample on M4 Pro (after JIT warmup)
- First inference: ~8–15 minutes for Metal JIT kernel compilation
- Model footprint: 17.21 GB (bf16) → 6.07 GB (4-bit), 65% reduction

## Running the Evaluation

```bash
# Full eval, 10B 4-bit model (default)
python client_eval/eval_mlx_asr.py

# Specific datasets
python client_eval/eval_mlx_asr.py --datasets ytb_asr_batch2 ytb_asr_batch3_malay

# Quick smoke test (first 8 samples)
python client_eval/eval_mlx_asr.py --max-samples 8

# 3B model
python client_eval/eval_mlx_asr.py --model-size 3b
```

## Conclusion

The MLX 4-bit port of MERaLiON-2-10B achieves good English ASR quality on Apple Silicon (within 14% of HF baseline). Multilingual performance (Malay, Tamil) degrades more significantly under 4-bit quantization. The 3B model is not recommended for multilingual tasks.

For production use on Apple Silicon, the 10B 4-bit model is the right choice, with the understanding that Malay/Tamil WER will be ~45–58% relatively worse than the server-side bf16 model.
