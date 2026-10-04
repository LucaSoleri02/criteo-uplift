# From Conversion Prediction to Incremental Ad Impact — the Criteo Uplift case

## The question

An ad budget is fixed, so the money question in online advertising is not *"who will visit if shown the ad?"* but *"who changes their behavior because of the ad?"* These are different people. Standard practice targets high scorers from a response model — but the users most likely to visit are often those who would have visited anyway ("sure-things"), on whom spend buys nothing. This project tests, on a large randomized benchmark, whether **uplift targeting** — ranking users by estimated incremental effect instead of predicted outcome — actually earns more per euro.

## The dataset

[Criteo Uplift v2.1](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) (Diemert et al., *A Large Scale Benchmark for Uplift Modeling*, AdKDD @ KDD 2018): **13,979,592 rows**, one per user, pooled from randomized advertising incrementality tests where a random 15% of users was held out from targeting. Per row: 12 anonymized pre-treatment features `f0–f11`, the randomized `treatment` flag, and outcomes `visit` (4.70% base rate), `conversion` (0.29%), and `exposure` (whether the user was actually reached — post-treatment, 3.6% of treated). Because assignment was randomized and verified balanced (max |SMD| 0.049 across all 12 features), causal contrasts are first-class citizens here, not an add-on.

## The pipeline in one line

**Preprocess & trust-check** → **response prediction** (the practitioner baseline) → **causal effect of the ad** → **who does the ad work on?** (uplift) → **which targeting policy earns more at equal budget?** → **what experiment would settle it in production?**

## Main results

1. **Advertising works, but the dose is dilute.** Assignment to targeting raises visits by **+1.03 pp** [+1.01, +1.06] over a 3.82% control rate (+27% relative) and conversions by +0.115 pp (+59% relative) — precise causal ITT effects, stable across folds. But only **3.6%** of targeted users were ever actually exposed; an exposure Wald IV puts the effect among the truly reached at ≈ **+28.7 pp** of visit — which is why the average effect looks small.
2. **Great prediction ≠ good targeting.** A gradient-boosting response model predicts visits superbly (AUC 0.947, top-10% lift 7.8×) — yet it ranks sure-things and persuadables identically, because the outcome it optimizes is not the outcome spend should buy.
3. **The effect is concentrated in a top slice.** Uplift (CATE) models (S- and T-learner, both agreeing) show the top decile of estimated incrementality has an observed effect of **+6.4 pp — ~6× the population average.** Below that decile the response-style ranking carries no extra incremental value.
4. **Uplift targeting wins at every realistic budget (the headline).** Same estimator, randomized arm contrast inside the selected slice, extra visits per 1,000 targeted:

   | budget | Response (status quo) | Uplift | Random |
   |---|---|---|---|
   | 5% | 52.9 | **90.6** (1.7×) | 9.7 |
   | 10% | 46.2 | **64.0** (1.4×) | 9.6 |
   | 20% | 36.2 | 38.1 (tie) | 9.7 |

   A deliberately untouched held-out test split confirmed the exact ordering (68.3 vs 56.0 vs 10.8 at the 10% budget; uplift's CI excludes response's). Uplift also captures **46% of all incremental visits with a 5% budget**, vs 27% for response targeting.
5. **The honest limit.** Visit-trained uplift slices *do* produce conversion lift, but the uplift **advantage does not transfer to conversions** (response ≥ uplift at every budget). A visit-effect ranking licenses no conversion-targeting claim — and conversions are the outcome revenue cares about. This is what the proposed A/B test must settle.
6. **Robust & honest by construction.** Duplicates (1.26M exact rows — a profiled benchmark artifact, surprisingly concentrated among treated non-visitors) materially dilute naive contrasts; dedup sensitivity moves the ITT to +1.49 pp while every policy ordering holds. All magnitudes are benchmark-scale (the data is sub-sampled); cross-policy *comparisons* — the decision-relevant read — are unaffected.

**One-sentence thesis:** *predicting who converts is not the same as identifying who should receive the ad — on this randomized benchmark, uplift targeting captured ~18% more of total incremental visits than response targeting at the same budget, and the production A/B experiment to validate it is fully specified.*

## Where everything is

```
criteo-uplift/
├── 01_preprocessing_eda_rules.ipynb        E1 — DQ, randomization checks, duplicate anatomy (11/11 checks PASS)
├── 02_response_prediction_topk_lift.ipynb  E2 — response baseline: LR + HistGB, both outcomes (AUC 0.947)
├── 03_ate_exposure_iv_exploration.ipynb    E3 — causal ITT effects + exposure IV nuance
├── 04_uplift_cate_models.ipynb             E4 — S/T-learner CATEs, Qini/AUUC, decile validation
├── 05_policy_value_eval_testset_uplift.ipynb E5 — the hero: policy comparison at 5/10/20% + one-shot test confirm
├── 06_sensitivity_segments_expdesign.ipynb E6/E7 — dedup sensitivity, uplift segments, A/B test spec
│
├── results/                                one results.md per notebook: decision, estimates, diagnostics, limitations
├── COMPREHENSIVE_RESULTS.md                all six stages in one place: pipeline, findings, fixes, open gaps
├── PLAN.md                                 the analysis contract frozen before modeling (splits, metrics, budgets)
│
├── outputs/
│   ├── figures/                            E1–E7 analysis figures (titles are the takeaways)
│   ├── tables/                             every CSV behind those figures
│   ├── deck/                               presentation-grade figures (PNG 2× + SVG) for the slide deck
│   └── notebooks_html/                     all notebooks exported to HTML for quick reading
│
├── src/
│   ├── config.py                           paths & constants
│   ├── core.py                             deterministic splits/folds, ATE bootstrap, policy-value estimator
│   ├── viz.py                              plotting style + savers
│   ├── precompute_hash.py                  rebuilds the hashed parquet cache from the raw CSV
│   └── deck_figures.py                     rebuilds outputs/deck/ from outputs/tables/ alone (no data needed)
│
├── requirements-deck.txt                   only what deck_figures.py needs
└── .gitignore                              data & model artifacts stay out of git
```

**Where to look, by intent:**
- Want the story fast? → `COMPREHENSIVE_RESULTS.md`, then the HTML notebooks.
- Want to check one claim? → the matching `results/results_0X_*.md`, verified against its CSV in `outputs/tables/`.
- Want to reuse the charts? → `outputs/deck/`, or `python src/deck_figures.py` (runs from CSVs, no dataset needed).
- Want to regenerate everything? → see below.

## Reproducing from scratch

1. Obtain `criteo-research-uplift-v2.1.csv.gz` (Criteo AI Lab, or HF mirror `criteo/criteo-uplift`) into the repo root.
2. `python src/precompute_hash.py` → builds `data/interim/criteo_uplift_v2_hashed.parquet`, adding the deterministic full-row hash every split and fold derives from (`md5(full row) % 1000` → 70/15/15 train/val/test). Full-row hashing guarantees exact duplicates never straddle splits — no leakage through the 1.26M dupes.
3. Run notebooks `01 → 06` in order. Deck figures are independent: `pip install -r requirements-deck.txt && python src/deck_figures.py`.

## Scope & limitations (the honest section)

- **No economic layer.** The dataset has no cost or revenue fields; "value" here = incremental visits. The fully specified E7 A/B test (uplift vs response targeting, equal budget, ~2,400 users/arm per the power sketch) is where economics enters.
- **Sparse conversions** → conversion results are secondary and the conversion-transfer finding is under-powered at small slices (reported as such).
- **Uplift model diversity** was deliberately limited to S- vs T-learner; this is a decision-grade benchmark, not a learner leaderboard.
- **`exposure` is post-treatment** — used only for the IV annotation, never as a feature; leakage rules are documented per feature in `PLAN.md`.
