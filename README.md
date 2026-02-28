# StudentDA - Learning Analytics Based Student Risk Prediction

This project provides a practical machine learning pipeline to **identify students at risk of academic underperformance early** using institutional and LMS activity data.

## Models Included
The training pipeline includes the three models you asked for:
- **Logistic Regression** (interpretable baseline)
- **Random Forest** (robust non-linear ensemble)
- **XGBoost** (high-performance gradient boosting)

The system trains all three models, compares them using stratified cross-validation, and automatically saves the best-performing model by ROC-AUC.

## Suggested Input Features
Use a CSV dataset where each row is one student record and one target column indicates risk.

Typical features:
- Attendance percentage
- Internal assessment scores
- Assignment submission and marks
- Quiz/lab performance
- LMS login frequency
- Time spent on LMS
- Content access count
- Forum interactions
- Historical GPA / previous semester result

Target column example:
- `at_risk` (0 = not at risk, 1 = at risk)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train and Compare Models
```bash
python src/train_models.py \
  --data data/student_data.csv \
  --target at_risk \
  --id-columns student_id
```

## Outputs
After training, artifacts are saved in `artifacts/`:
- `leaderboard.csv` - CV comparison of Logistic Regression, Random Forest, and XGBoost
- `test_metrics.json` - holdout test metrics for the selected best model
- `best_model.joblib` - full preprocessing + model pipeline for inference

## Accuracy Improvement Tips
To improve prediction accuracy beyond a basic setup:
1. Balance classes (SMOTE / class weights / threshold tuning)
2. Perform feature engineering (engagement trends, lag-based features)
3. Tune hyperparameters (GridSearchCV / Optuna)
4. Use time-aware validation if semester sequence matters
5. Track precision-recall trade-offs to reduce false negatives (missed at-risk students)

---
If you want, I can next add:
- a Streamlit dashboard for faculty/admin roles,
- feature importance explainability (SHAP),
- and an API endpoint for real-time risk scoring.
