# Claude student essays (fresh source)

1,000 English student-style essays written on 2026-10-29 by Claude models for
retraining Skolfie on current AI text (`notebooks/retrain_modern.ipynb`):

| file | generator | essays | role |
|---|---|---|---|
| `claude-opus-5-5.jsonl.gz` | claude-opus-5-5 | 375 | train / val / test |
| `claude-sonnet-5-5.jsonl.gz` | claude-sonnet-5-5 | 375 | train / val / test |
| `claude-haiku-5-5.jsonl.gz` | claude-haiku-5-5 | 250 | held out: unseen-model test only |

Prompts come from `skolfie.data.generation.student.student_prompt_plan(1000)`:
600 on the 15 PERSUADE 2.0 assignments (grade 6-12 personas, some asking the
model to "sound like a real student") and 400 college essays and personal
narratives. Records are fresh-source JSONL made by
`scripts/import_claude_essays.py` (same format as `scripts/generate_ai_texts.py`).
The retrain notebook uses these copies when `My Drive/skolfie/data/raw/fresh/`
has no `.jsonl` files.
