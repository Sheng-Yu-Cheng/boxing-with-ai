#!/usr/bin/env python3
r"""
train_punch_classifier.py

Train a small trajectory classifier from MediaPipe punch segments.

Example:

    python .\scripts\vision\train_punch_classifier.py `
      --dataset .\data\punch_dataset `
      --out .\models\punch_classifier.joblib `
      --hand right

This trains on trajectory features extracted from .npz files recorded by
record_punch_dataset.py.

Labels should include:
    right_straight
    right_hook
    right_uppercut
    negative
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.punch_vision_common import load_segment_npz, extract_punch_features


def load_dataset(dataset_dir: Path, hand: str, resample_len: int):
    paths = sorted(dataset_dir.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found under {dataset_dir}")

    X = []
    y = []
    used_paths = []
    feature_names = None

    for p in paths:
        frames, label, sample_hand, meta = load_segment_npz(p)

        # The label is authoritative, but hand can be used for feature extraction.
        feat_hand = hand if hand in ("left", "right") else sample_hand
        if feat_hand not in ("left", "right"):
            feat_hand = "right"

        feats, names = extract_punch_features(frames, hand=feat_hand, resample_len=resample_len)

        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("Feature name mismatch")

        X.append(feats)
        y.append(label)
        used_paths.append(str(p))

    return np.vstack(X).astype(np.float32), np.array(y), feature_names, used_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RadarBox punch trajectory classifier")
    parser.add_argument("--dataset", default="data/punch_dataset")
    parser.add_argument("--out", default="models/punch_classifier.joblib")
    parser.add_argument("--hand", choices=["left", "right", "auto"], default="right")
    parser.add_argument("--resample-len", type=int, default=16)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--model", choices=["rf", "extratrees"], default="rf")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    X, y, feature_names, paths = load_dataset(dataset_dir, args.hand, args.resample_len)

    labels, counts = np.unique(y, return_counts=True)
    print()
    print("[Trainer] dataset:", dataset_dir)
    print("[Trainer] samples:", len(y))
    print("[Trainer] features:", X.shape[1])
    print("[Trainer] labels:")
    for lab, c in zip(labels, counts):
        print(f"  {lab}: {c}")

    min_count = int(np.min(counts))
    stratify = y if min_count >= 2 and len(labels) >= 2 else None

    if args.model == "rf":
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=args.random_state,
        )
    else:
        clf = ExtraTreesClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=args.random_state,
        )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])

    if len(y) >= 8 and min_count >= 2 and len(labels) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=stratify,
        )

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        print()
        print("[Trainer] holdout classification report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        print("[Trainer] confusion matrix labels:", list(pipe.classes_))
        print(confusion_matrix(y_test, y_pred, labels=pipe.classes_))

        # Cross-validation if feasible.
        n_splits = min(5, min_count)
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)
            scores = cross_val_score(pipe, X, y, cv=cv)
            print(f"[Trainer] CV accuracy: mean={scores.mean():.3f}, std={scores.std():.3f}, splits={n_splits}")

    else:
        print("[Trainer] too few samples for a reliable holdout report; fitting all data.")

    # Fit final model on all data.
    pipe.fit(X, y)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": pipe,
        "feature_names": feature_names,
        "hand": args.hand,
        "resample_len": args.resample_len,
        "labels": list(pipe.classes_),
        "training_paths": paths,
        "training_label_counts": {str(l): int(c) for l, c in zip(labels, counts)},
    }
    joblib.dump(bundle, out)

    meta_path = out.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "out": str(out),
                "dataset": str(dataset_dir),
                "samples": int(len(y)),
                "features": int(X.shape[1]),
                "labels": list(pipe.classes_),
                "training_label_counts": bundle["training_label_counts"],
                "hand": args.hand,
                "resample_len": args.resample_len,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("[Trainer] saved:", out)
    print("[Trainer] meta :", meta_path)


if __name__ == "__main__":
    main()
