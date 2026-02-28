"""Generate a realistic synthetic dataset for student risk modeling."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEPARTMENTS = ["CSE", "ECE", "MECH", "CIVIL", "EEE", "MBA"]


def generate(n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    attendance = rng.normal(78, 12, n_rows).clip(35, 100)
    internal = rng.normal(68, 14, n_rows).clip(20, 100)
    assignment = rng.normal(72, 13, n_rows).clip(15, 100)
    quiz = rng.normal(70, 15, n_rows).clip(10, 100)
    lms_logins = rng.poisson(8, n_rows).clip(0, 30)
    lms_hours = rng.normal(6, 3, n_rows).clip(0, 20)
    content_views = rng.poisson(45, n_rows).clip(0, 200)
    forum_posts = rng.poisson(4, n_rows).clip(0, 40)
    previous_gpa = rng.normal(7.1, 1.1, n_rows).clip(3, 10)
    department = rng.choice(DEPARTMENTS, n_rows)
    hostel = rng.choice(["yes", "no"], n_rows, p=[0.55, 0.45])

    risk_score = (
        0.06 * (75 - attendance)
        + 0.05 * (65 - internal)
        + 0.04 * (70 - assignment)
        + 0.03 * (70 - quiz)
        + 0.02 * (6 - lms_logins)
        + 0.02 * (5 - lms_hours)
        + 0.015 * (7 - previous_gpa)
        + rng.normal(0, 0.5, n_rows)
    )

    probs = 1 / (1 + np.exp(-risk_score))
    at_risk = (probs > 0.5).astype(int)

    return pd.DataFrame(
        {
            "student_id": np.arange(10001, 10001 + n_rows),
            "attendance_pct": attendance.round(2),
            "internal_score": internal.round(2),
            "assignment_avg": assignment.round(2),
            "quiz_avg": quiz.round(2),
            "lms_logins_per_week": lms_logins,
            "lms_hours_per_week": lms_hours.round(2),
            "content_views": content_views,
            "forum_posts": forum_posts,
            "previous_gpa": previous_gpa.round(2),
            "department": department,
            "hostel_resident": hostel,
            "at_risk": at_risk,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/student_data.csv")
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} rows to {output}")


if __name__ == "__main__":
    main()
