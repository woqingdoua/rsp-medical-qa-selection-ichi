# ICHI protocol and reproduction

## Data and permissions

Obtain AfriMed-QA, MedExpQA, MedExQA, PubMedQA and MedMCQA from their original distributors under the applicable terms. This repository does not redistribute benchmark text or model weights. MedGemma access and model terms are separate from code access.

Each authorized dataset directory contains `pool.jsonl`, `dev.jsonl` and `test.jsonl`. A normalized record has `uid`, `question`, `options` (consecutive letter-to-text mapping), `answer_letter`, `rationale`, and optional `domain` (`medical` for this study). UIDs are bookkeeping only. `configs/ichi.json` lists the study sizes.

`src/prepare_additional_medical.py` contains the four additional-dataset normalization/split procedures. `src/prepare_data.py` contains MedMCQA preparation. Exact historical MedMCQA reconstruction also needs the prior-pilot exclusion identities, which are not redistributed. Do not use an earlier large-pool raw-data script in place of the study's 512/256/4096 partition. Legacy data-helper comments about rationale-visible acquisition describe other consumers, not the question/options-only ICHI selector.

These are study-specific partitions, not uniformly official test splits. MedExQA and PubMedQA use internal reshuffles; MedMCQA comes from source training records. MedExpQA is reported after excluding four normalized-question pool-overlap test rows, reducing 125 to121. One within-test duplicate remains. The filter does not reconstruct development or pool data, remove all contamination, or retrain any branch.

## Fresh scoring and selection

The following commands document the released interface; they were not executed as experiments for this release. They require an authorized GPU and your own matching data. Run from the repository root:

```bash
python run_ichi.py score --dataset medexpqa \
  --data-dir data/additional_medical/medexpqa \
  --output-dir runs/example/score --cache-dir model_cache \
  --budget 128 --seed 13

python selection.py --scores runs/example/score/components/scores.jsonl \
  --output-dir runs/example/rsp --budget 128 --method rsp_auto

python selection.py --scores runs/example/score/components/scores.jsonl \
  --output-dir runs/example/drop --budget 128 --method rsp_drop

python selection.py --scores runs/example/score/components/scores.jsonl \
  --output-dir runs/example/random --budget 128 --method random4b
```

Scoring fixes the paper's K8, final-block output, absolute sigma0.02, max length1024 and batch size4. Selection restores pool-index order. The scorer's inherited `rsp` column is **not** the automatic-RMS method; `selection.py` derives the final `rsp_auto` column from stored `rsp_flip` and `rsp_drop`. The `rsp_drop` column is raw D. Random selection reuses iid keys already emitted by the scorer and takes the highest B within the entropy pool, not a fresh unrestricted draw.

Scoring at a smaller budget does not provide missing components outside its entropy pool. Never treat null components as zero. To reuse a larger scored pool at smaller budgets, reconstruct that budget's top-min(4B,N) candidates and recompute RMS scales within that pool. The historical experiments reused retained components and supplemented missing candidates while preserving the inherited Gaussian stream; see the archive and provenance guide. A fresh scoring run does not recreate an unpinned historical execution.

For full-pool analysis, explicitly score the complete bank using `run_ichi.py score --full-pool ...` and select using `selection.py --full-pool ...`. Full-pool RSP computes its coefficient across all N candidates, so within a dataset/seed that coefficient is independent of B. Changing traversal to construct a fresh cache is not a claim to reproduce the archived historical Gaussian assignments.

## Training and evaluation

```bash
python run_ichi.py train --dataset medexpqa \
  --data-dir data/additional_medical/medexpqa \
  --selection-dir runs/example/rsp --output-dir runs/example/rsp_train \
  --cache-dir model_cache --budget 128 --seed 13
```

Repeat with separate output directories for `drop` and `random`, and seeds13/23/37. Each branch creates a fresh adapter on the raw backbone. This launcher refuses existing output directories and checks frozen UID order and selected membership before training. It does not continue a trained Original adapter. Use the same seed and frozen study recipe across methods.

The exact retained serializer/trainer is `src/train_original_rsp_raw.py`. Cleaning removes only its three explicit answer-tail patterns. Question/options prefix head and tail are retained under length pressure, followed by rationale right-truncation, while reserving instruction and full answer target. Prompt and rationale labels are `-100`. Only the leading-space option letter, newline and EOS are supervised. There is no weighted CoT loss. Evaluation uses question/options without auxiliary text and a restricted next-letter softmax, not free generation.

Training writes `result.json` and `predictions.npz` into your untracked output directory. The release launcher skips adapter saving. Raw trainer CLI defaults include legacy budgets and score-rank ordering; use the launcher or provide every paper-specific option explicitly.

For MedExpQA reporting, apply the normalized-question pool-overlap exclusion to the original 125-row predictions exactly once. `archive/canonical_pool_order_baselines_20260917/aggregate_canonical_baselines.py` records the original filter. Other datasets keep their full study tests. Do not compare raw trainer125-item metrics with the paper's121-item metrics. Existing release CSVs are already filtered and must not be filtered a second time.

## Baselines

Question/options-only implementations are entropy and negative margin in the raw scorer; BALD/BatchBALD/Noise-Stability/SIFT in `src/score_original_rsp_extra_baselines.py`; and NAIT in `src/score_latest_nait.py`. BALD uses eight hidden-dropout passes (probability0.1), not parameter dropout. Noise-Stability uses eight relative-RMS0.03 perturbations and cosine k-center. SIFT's uncertainty/diversity adaptation and NAIT's development-question PCA orientation are not exact reproductions of their cited originals.

Auxiliary-text-aware RSR and ASLEC-DROP use `src/score_latest_static_baselines.py`; TEMP uses `src/train_latest_temp_direction.py` (development explanations) and `src/score_latest_temp.py`. RSR rank clipping is100 and ASLEC step segmentation is deterministic. TEMP performs its recorded calibration and auxiliary LoRA-direction procedure; it is not the main selector's coefficient calculation. Refer to retained CLI options and metadata rather than substituting another paper's parameters.

`src/rationale_baselines.py` is a shared dependency for these comparators. Its older `score_rsp` function describes a different rationale-visible algorithm and is **not** the ICHI RSP entry point. The static scorer also emits an unused Complexity-aware FT column, which is not part of this paper's reported comparator set. Historical training-order preparation and aggregation are retained under `archive/canonical_pool_order_baselines_20260917`.

## Reproducibility limits

The historical model repository ID is known, but not every resolved model revision, precision and package version was retained. A later cached revision appears in the budget manifest; this does not prove that earlier runs used identical weights. Requested GPU classes were not uniform across comparator families. Source hashes establish file provenance, not hardware-level determinism. Full private run trees and per-item arrays are not redistributed, so some archival scripts require those retained inputs and are not standalone end-to-end launchers.
