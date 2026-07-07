# SLT Experiment Notes
_Last updated: 2026-07-07_

---

## Currently Running (as of 2026-07-07 ~13:30 CDT)

| Exp | Job name | Status | Notes |
|-----|----------|--------|-------|
| 45 | `train-exp45-how2sign-multistream-ctc-bpe500` | Running (~2h in) | How2Sign interleaved multistream, BPE-500. Needed by exp57. |
| 56 | `train-exp56-phoenix-gloss2text-nllb200` | Running (~5m in) | NLLB-200-distilled-600M Gloss→German. Needed by exp59. Fixed OOM (added `gpu.memory Gt 15000`). |
| 58 | `train-exp58-phoenix-interleaved-multistream-no-fp16` | Running (~2h in) | Interleaved multistream + velocity, fp16 removed. Fix for exp50/52 collapse. |

---

## Pending (auto-submit via background watcher at `/tmp/submit_dependent_jobs.sh`)

| Exp | Waits for | Purpose |
|-----|-----------|---------|
| 57 | exp45 checkpoint | Full-weight transfer How2Sign → PHOENIX encoder (interleaved multistream, same arch as exp45, LR halved to 1e-4) |
| 59 | exp56 checkpoint (+ exp25 must exist) | End-to-end Sign→German joint: exp25 encoder + NLLB decoder, 3-phase training |

> **Note:** The background watcher (`PID 3909`) lives in the current terminal session.
> If the session is closed, submit manually:
> ```bash
> kubectl apply -f nautilius/train-exp57-phoenix-multistream-from-exp45.yaml
> kubectl apply -f nautilius/train-exp59-phoenix-sign2gloss2text-joint.yaml
> ```
> (Only after their respective dependencies have a `best_model.pt` on the cluster.)

---

## Recent Fixes Applied This Session

- **exp45/56/58**: Extended node blacklist to include `k8s-haosu-05/10/19`, `k8s-4090-01`
- **exp56**: Added `nvidia.com/gpu.memory Gt 15000` after OOM on a 10.57 GB card (NLLB-600M needs >15 GB)

---

## Experiment Lineage (exp56–59)

```
exp45 (How2Sign, interleaved BPE-500)
  └─► exp57: transfer encoder to PHOENIX (full-weight, same arch)

exp56 (NLLB-200 Gloss→German)
  └─► exp59: joint Sign→Gloss→German (encoder from exp25 + NLLB decoder from exp56)

exp58: interleaved multistream + velocity, no fp16 (standalone fix for exp50/52)
```

### Baselines to beat
| Task | Best exp | BLEU-4 |
|------|----------|--------|
| PHOENIX Gloss→German | exp33 (mBART-50) | ~16.33 |
| PHOENIX Sign→German (multistream) | exp25 (sequential) | ~12.74 |

---

## Known Issues / Graveyard

| Exp | Problem |
|-----|---------|
| 50, 52 | fp16 NaN/Inf in ConformerBlocks → GradScaler skipped every batch, train loss = 0.0 |
| 55 | BART-large Gloss→German → val BLEU-4 ~14.86 (below mBART-50 baseline) |
| 56 (first run) | OOM on 10.57 GB GPU — fixed with `gpu.memory Gt 15000` |

---

## Next Steps (after current jobs finish)

1. **Check exp56 BLEU** vs exp33 (mBART-50, 16.33) and exp55 (BART-large, 14.86) — determines if NLLB is worth using in exp59
2. **Check exp58** — if it trains without collapse, compare vs exp25 (12.74) to confirm interleaved + velocity works
3. **Check exp57** — does full-weight transfer from How2Sign help vs scratch (exp25) or partial flat transfer (exp54)?
4. **exp59 (joint)** — final eval: end-to-end Sign→German BLEU-4 vs pipeline upper bound
5. If exp56 < exp33 → consider swapping exp59's decoder back to mBART-50
6. If exp58 collapses again → investigate ConformerBlock instability more deeply (gradient clipping? smaller LR?)
