# RSP and RSP-Drop for medical QA data selection

Research code for **Perturbation-Based Data Selection for Medical Question Answering**, by Yuexin Wu, Dayou Yu, and Vasile Rus. This is the **ICHI extended-study implementation**, submitted to IEEE ICHI 2027; acceptance is not asserted.

**[Read the ICHI preprint PDF](paper/ichi-preprint.pdf)** · [Method and reproduction](docs/REPRODUCIBILITY.md) · [Source and result provenance](docs/PROVENANCE.md) · [Paper configuration](configs/ichi.json)

The ICHI preprint was submitted to arXiv as a **new submission** (8275811), separately from the GenAI4Health article. The PDF above is the actual arXiv-compiled manuscript, updated to include this code repository link. The update was resubmitted successfully; its status at the update check was **on hold** (moderation checks), without a public arXiv identifier. Its public abstract link can be added after announcement; the GenAI4Health link below is **not** the ICHI paper link.

## Relationship to the GenAI4Health study

The prior paper, [An Empirical Study of Development-Tuned Representation Sensitivity for Medical QA Data Selection](https://arxiv.org/abs/2609.09684v2), appeared as a non-archival GenAI4Health @ NeurIPS 2026 workshop study. Its code is kept in a [separate repository](https://github.com/woqingdoua/rsp-medical-qa-selection). This repository does not replace that code or article.

| Aspect | GenAI4Health study | This ICHI study |
| --- | --- | --- |
| Signals | Flip and non-flipping margin Drop | Same two underlying signals |
| Mixing coefficient | Selected using labeled development performance | Automatic RMS scale balancing from unlabeled candidate scores |
| Methods | Development-tuned RSP | Automatic-RMS RSP and contraction-only RSP-Drop |
| Retained-set budgets | Primary B = 128 | B = 64, 128, 256 |
| Additional analyses | Development tuning audit | Component support, selected-set comparison, entropy-prefilter ablation |

Both studies use the same five benchmark families and MedGemma backbone. Cross-paper numerical differences are not a controlled estimate of the benefit of automatic scaling.

## Method

The task is offline selection from an existing question bank, not acquiring new labels or certifying explanation quality. For each candidate, the unadapted `google/medgemma-4b-it` receives only the **question and options**. Neither its answer label nor its auxiliary explanation is used in RSP scoring.

1. Compute the clean answer distribution as a next-token softmax restricted to legal option letters.
2. Keep the top `min(4B,N)` candidates by predictive entropy.
3. Add absolute Gaussian noise (`sigma = 0.02`) to the final decoder block output for `K = 8` passes.
4. Compute **Flip**: the fraction of passes whose predicted letter differs from the clean prediction. Compute **Drop**: positive contraction of the answer-probability margin on non-flipping passes, averaged over **all K passes**, with zero contribution from flipped passes.
5. For RSP, compute `a_F = RMS(F)` and `a_D = RMS(D)` within the eligible pool, then `lambda0 = a_F / (a_D + 1e-8)` and rank by `F + lambda0 * D`. When `a_F <= 1e-8`, rank directly by raw `D`.
6. **RSP-Drop** uses the same components and entropy pool, but ranks only by `D`.
7. Retain B items, break exact ties by original pool index, and restore ascending pool-index order before training.

In the full-pool ablation, RSP recomputes RMS scales over the **complete bank** separately for each dataset and seed; that coefficient is shared across budgets within that dataset–seed pair. Full-pool RSP-Drop directly ranks all candidates by D.

Noise is instantiated over the final-block output tensor. Since no later attention block follows the intervention and answer scoring uses final-position logits, only that final position affects the answer distribution. This is an output-representation sensitivity heuristic, not a robustness or clinical-safety guarantee.

## Downstream adaptation

Each branch starts from the raw MedGemma checkpoint with a **fresh LoRA adapter**, without answer-only warm-up. Selected training inputs contain the question, options, and auxiliary explanation. **Only the final-answer target tokens receive loss**; explanation tokens are context, not a weighted CoT-generation target. Testing provides question/options without the explanation.

The paper uses rank-16 LoRA (alpha 32, dropout 0.05), two epochs, learning rate `1e-4`, cosine decay, 3% warm-up, microbatch 1, accumulation 8, length 1024, and branch seeds 13/23/37. This is **not** the later experimental `L_answer + 0.1 L_CoT` training family. Retaining B items equalizes item count, not explanation length, acquisition access, or compute cost.

## Quick start

Use Python 3.11 or later. Synthetic tests and aggregate checks need only the standard library and do not download data or models:

```bash
python -m unittest discover -s tests -v
python analysis/summarize_results.py
python analysis/verify_release.py
```

For an authorized CUDA environment, install the functional dependencies and obtain your own authorized MedGemma access:

```bash
pip install -r requirements.txt
python run_ichi.py --help
python run_ichi.py score --help
```

See [the reproduction guide](docs/REPRODUCIBILITY.md) for separate scoring, CPU selection, training and evaluation commands. No experiment is launched on import or by running the tests. Historical dependency versions and model revisions were not fully pinned; `requirements.txt` is not a recovered lockfile.

## Existing results

The release retains unrounded, already evaluated per-run metrics and deterministic summaries. At B = 128, the five-dataset equal-weight means are:

| Method | Accuracy (%) | Macro-F1 (%) |
| --- | ---: | ---: |
| Entropy-prefiltered Random | 51.42 | 45.11 |
| RSP | 53.00 | 47.55 |
| RSP-Drop | 53.60 | 48.28 |

These are descriptive averages, not significance tests or a universal ordering. RSP-Drop has higher primary-budget means on MedExpQA, MedExQA and MedMCQA; RSP has higher means on AfriMed-QA and PubMedQA. The [per-dataset primary summary](results/primary_summary.csv) includes sample SD across three seeds. [Budget summaries](results/budget_summary.csv) cover all 135 RSP/RSP-Drop/Random branches; [paired entropy-ablation rows](results/entropy_ablation_paired.csv) cover only settings where the prefilter actually excludes candidates.

The pool-matched Random baseline samples within the same entropy-restricted pool. It is not unrestricted full-bank random unless `N <= 4B`. Baselines labeled BALD, BatchBALD, SIFT, NAIT, RSR, ASLEC-DROP and TEMP are study-specific inspired implementations, not exact reproductions of their source papers. RSR, ASLEC-DROP and TEMP read auxiliary explanations during candidate scoring; RSP and RSP-Drop do not.

## Scope and release contents

This is an exploratory medical-knowledge benchmark study, not a clinical deployment evaluation. Four benchmark families appear among MedGemma's documented post-training sources; exact item overlap remains unresolved. MedExpQA metrics exclude four pool-overlap test questions (121 retained), but the post-hoc filter does not rebuild the splits. There is no matched answer-only adaptation control identifying a causal benefit of explanation conditioning, and no validated rule for choosing RSP versus RSP-Drop on unseen data.

Included: source code, paper configuration, retained aggregate metrics, historical source fingerprints, CPU tests, and the identified ICHI preprint. Excluded: model/adapter weights, benchmark question/explanation text, per-item prediction arrays, selection UID lists, secrets, and original private git history. Read [license status](LICENSE_STATUS.md) before reuse. No new scoring, training, checkpoint evaluation, or scientific experiment was performed to prepare this release.

## Citation

```bibtex
@unpublished{wu2026perturbation,
  title={Perturbation-Based Data Selection for Medical Question Answering},
  author={Wu, Yuexin and Yu, Dayou and Rus, Vasile},
  year={2026},
  note={Preprint submitted to IEEE ICHI 2027},
  url={https://github.com/woqingdoua/rsp-medical-qa-selection-ichi/blob/main/paper/ichi-preprint.pdf}
}
```

Please also cite the [GenAI4Health prior work](https://arxiv.org/abs/2609.09684v2) when discussing development-tuned RSP. Its public identifier must not be assigned to this extended manuscript.
