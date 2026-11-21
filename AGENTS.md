# AGENTS.md

Rules for anyone (human or AI agent) changing this repository. Read this
before editing code. `README.md` explains how to use Skolfie; this file
explains how to work on it.

## What Skolfie is

An AI-generated-text detector for student essays and other longer English
texts, built for schools. Verdicts: Human / AI / Mixed, with reasons a
teacher can read. Accuracy on student writing and a low false-positive rate
on real students (including English learners) matter more than anything
else. News-domain results are not a goal.

## Layout

| Path | What lives there |
|---|---|
| `skolfie/data/` | corpus schema, dataset loaders (`sources/`), AI-essay generation (`generation/`), fairness sets, file parsers |
| `skolfie/features/` | stylometric features, Burrows' Delta, language-model and Binoculars features, feature cache |
| `skolfie/models/` | feature model (LightGBM), transformer model, stacker, baselines, legacy ensemble |
| `skolfie/eval/` | splits, leak checks, metrics, reports, fairness, LOFO |
| `skolfie/mixed/`, `skolfie/explain/`, `skolfie/robust/` | Mixed detection by chunks, explanations, evasion attacks and input normalisation |
| `skolfie/service.py`, `skolfie/cli.py`, `skolfie/app/` | `Detector` (the one entry point for checking a text), CLI, Gradio app |
| `scripts/` | command-line steps the notebooks call |
| `notebooks/` | Colab notebooks that run the pipeline on the user's Google Drive |
| `configs/` | split configurations |
| `data/ai_essays/` | AI essays shipped in the repo (gzipped JSONL) |
| `website/` | static marketing site (GitHub Pages) |
| `tests/` | pytest suite |

## Contracts you must not break

Trained models and Colab runs depend on these. Changing one needs an
explicit decision from the project owner, not a cleanup.

1. **Pickled models load by import path.** `models_v3` and older `.joblib`
   files reference classes and functions by module path (for example
   `skolfie.models.feature_model.FeatureModel`,
   `skolfie.models.stacker`, `skolfie.models.transformer_model`,
   `skolfie.features.*`, `skolfie.robust.normalize`). Do not rename, move or
   delete anything a pickle can reference, and keep attribute names that
   `__setstate__`/`predict` read.
2. **Model behaviour.** A refactor must give the same scores for the same
   input and the same model file.
   Input normalisation is versioned (`skolfie.robust.normalize`,
   `CURRENT_NORMALIZER`): a model stores the version it was fitted with in
   `normalizer`, and files without it use version 1. Version 2 (models
   fitted from 2026-11-21 on) also folds typography and replaces corpus
   anonymisation tags. Change what a version does only by adding a new one.
3. **Notebook interfaces.** `retrain_modern.ipynb`, `app.ipynb`,
   `generate_modern.ipynb` and `generate_gemini_batch.ipynb` are what the
   owner runs. Keep their parameters, Drive paths and the script flags and
   functions they call.
4. **Data formats.** Corpus parquet schema (`skolfie/data/schema.py`),
   `manifest.json`, split JSON, essay JSONL fields, `known_limitations.json`,
   report file names.

## Code rules

- Python 3.10+, `ruff check .` clean (line length 100, rules E, F, I).
- No dead code: no unused functions, parameters, imports, constants or
  commented-out code. Delete it; git keeps history.
- One implementation per job. Before adding a helper, search for an existing
  one and reuse it.
- Keep functions small and named for what they do. Comments explain why,
  not what.
- Code, comments and identifiers in English.
- Every behaviour change comes with a test. `pytest -q` must pass before a
  merge; CI runs ruff and pytest on Python 3.11 and 3.12.
- Heavy optional dependencies (`torch`, `transformers`, `lightgbm`,
  `gradio`, API clients) are imported lazily inside the function that needs
  them, so the core install and the tests stay light.

## Cost, privacy and secrets

- **GPU only where needed.** Language-model scoring and transformer
  training run on a Colab GPU runtime. Everything else (corpus, harvesting,
  splits, leak checks, LightGBM, stacking, reports, API essay generation)
  runs on CPU. Notebooks keep GPU and CPU stages apart.
- **No student text leaves the machine.** Checking a text never calls an
  external AI API. APIs are used only to generate training essays.
- **No secrets in the repo.** API keys and tokens come from Colab secrets
  or environment variables, never from files in git.
- Respect dataset licences: record each source's licence in its loader and
  keep non-commercial data out of anything sold.

## Workflow

- Work on a branch and open a pull request; direct pushes to `main` are
  blocked. The owner does not need to review: once ruff and the tests pass,
  merge it.
- Significant decisions (a retrain, spending money or Colab units, new data
  sources, pricing, website claims, architecture changes) go through the
  board review described in the project instructions first.
