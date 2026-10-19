# Student Query Classification System

A small AI problem-design project for SWYNEX Task 1. It classifies student-support queries into `Academics`, `Fees`, `Account`, `Scholarship`, or `General`.

## Files
- `AI_Problem_Definition.md` — problem, target users, data source, constraints, evaluation and limitations.
- `data/student_queries.csv` — 150 synthetic, manually labelled example queries.
- `src/train_evaluate.py` — reproducible TF-IDF + Logistic Regression baseline.
- `requirements.txt` — Python dependencies.

## Run locally
Use Python 3.10 or newer.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install -r requirements.txt
python src/train_evaluate.py
```

The script prints accuracy, macro precision/recall/F1, a classification report, and a confusion matrix. Report the metrics actually produced by your run; do not claim the success targets were achieved without checking.

## Important note
The dataset is synthetic and intentionally small. This project demonstrates an AI workflow; it is not production-ready and has not been validated on real student queries. See the problem definition for constraints and responsible-use notes.
