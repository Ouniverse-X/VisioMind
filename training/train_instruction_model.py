from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def _read(path: Path) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            texts.append(record["text"])
            labels.append(record["intent"])
    return texts, labels


def _baseline(text: str) -> str:
    lowered = text.casefold()
    if any(
        token in lowered
        for token in (
            "摆放失败",
            "放错",
            "格外",
            "重新",
            "recover",
            "failed placement",
            "incorrectly placed",
            "re-place",
        )
    ):
        return "recover_placement"
    if any(token in lowered for token in ("停止", "停下", "stop", "halt", "cancel")):
        return "stop"
    if any(token in lowered for token in ("识别", "检查", "inspect", "locate", "find")):
        return "inspect"
    if any(
        token in lowered for token in ("附近", "靠近", "旁边", "move near", "navigate", "approach")
    ):
        return "move_near"
    if any(token in lowered for token in ("放进", "装入", "归位", "inside", "into", "store")):
        return "transfer_inside"
    if any(token in lowered for token in ("放到", "摆在", "顶部", " on ", "on top", "set ")):
        return "transfer_on_top"
    return "pick_up"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()

    train_texts, train_labels = _read(args.train)
    test_texts, test_labels = _read(args.test)
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=30000,
    )
    train_features = vectorizer.fit_transform(train_texts)
    classifier = LogisticRegression(
        C=6.0,
        max_iter=1500,
        class_weight="balanced",
        random_state=202607,
    )
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(vectorizer.transform(test_texts))
    baseline_predictions = [_baseline(text) for text in test_texts]
    labels = sorted(set(train_labels))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "model_version": "industrial-char-tfidf-logreg-v2-recovery",
        "training_seed": 202607,
        "labels": labels,
    }
    joblib.dump(payload, args.output, compress=3)
    metrics = {
        "model_version": payload["model_version"],
        "train_samples": len(train_texts),
        "test_samples": len(test_texts),
        "train_class_distribution": dict(Counter(train_labels)),
        "test_class_distribution": dict(Counter(test_labels)),
        "split_policy": "held-out paraphrase templates",
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "macro_f1": float(f1_score(test_labels, predictions, average="macro")),
        "baseline_accuracy": float(accuracy_score(test_labels, baseline_predictions)),
        "baseline_macro_f1": float(f1_score(test_labels, baseline_predictions, average="macro")),
        "labels": labels,
        "confusion_matrix": confusion_matrix(test_labels, predictions, labels=labels).tolist(),
        "classification_report": classification_report(
            test_labels, predictions, labels=labels, output_dict=True, zero_division=0
        ),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "path": str(args.output.name),
        "sha256": _sha256(args.output),
        "bytes": args.output.stat().st_size,
        "format": "joblib",
        "license": "project-generated",
    }
    (args.output.parent / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
