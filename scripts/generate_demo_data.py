"""Generate a synthetic dataset for demo purposes.

If a user does not have a real dataset for the student academic risk
early-warning system, this script generates a synthetic one that mirrors
the expected schema: 2000 samples with 15 numeric features named ``f0``
through ``f14``, a multi-class ``Academic_Risk_Class`` target and a
regression target ``Final_Academic_Score`` (0-100).

The generated CSV is written to ``data/college_students_academic_performance_dataset.csv``.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

try:
    from sklearn.datasets import make_classification
except ImportError:
    print("scikit-learn is required to generate demo data.")
    sys.exit(1)


def main():
    n_samples = 2000
    n_features = 15
    n_classes = 3
    seed = 42

    print("Generating synthetic dataset (n={}, features={}, classes={}) ...".format(
        n_samples, n_features, n_classes))

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(5, n_features // 2),
        n_redundant=2,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=2,
        weights=[0.4, 0.35, 0.25],
        class_sep=1.2,
        flip_y=0.05,
        random_state=seed,
    )

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.5, size=X.shape)
    X = X + noise

    feature_cols = ["f{}".format(i) for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df["Academic_Risk_Class"] = y.astype(int)

    # Build a regression target in [0, 100] that correlates with the class.
    class_baseline = {0: 78.0, 1: 62.0, 2: 42.0}
    reg_target = np.array([class_baseline[int(c)] for c in y])
    reg_target = reg_target + rng.normal(0.0, 6.0, size=n_samples)
    reg_target = np.clip(reg_target, 0.0, 100.0)
    df["Final_Academic_Score"] = reg_target.round(2)

    # Reorder columns so features come first, then targets.
    ordered_cols = feature_cols + ["Academic_Risk_Class", "Final_Academic_Score"]
    df = df[ordered_cols]

    data_dir = os.path.join(_PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(
        data_dir, "college_students_academic_performance_dataset.csv"
    )
    df.to_csv(out_path, index=False)

    print("Saved {} rows x {} columns to {}".format(
        df.shape[0], df.shape[1], out_path))
    print("Class distribution:")
    counts = df["Academic_Risk_Class"].value_counts().sort_index()
    for cls, count in counts.items():
        print("  Class {}: {} ({:.1%})".format(cls, count, count / len(df)))
    print("Final_Academic_Score: min={:.2f}, mean={:.2f}, max={:.2f}".format(
        df["Final_Academic_Score"].min(),
        df["Final_Academic_Score"].mean(),
        df["Final_Academic_Score"].max(),
    ))


if __name__ == "__main__":
    main()
