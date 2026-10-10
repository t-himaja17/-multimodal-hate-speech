
import sys
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from src.multimodal_hate.data.loaders.hateful_memes import load_hateful_memes
from src.multimodal_hate.training.dataset import (
    MultimodalHateSpeechDataset,
    multimodal_collate_fn,
)
from src.multimodal_hate.models.multimodal_model import MultimodalHateSpeechModel
from src.multimodal_hate.models.sarcasm.sarcasm_bert import SarcasmBERT
from src.multimodal_hate.models.sarcasm.sentiment import SentimentReversal
from src.multimodal_hate.evaluation.evaluator import MultimodalEvaluator

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "hateful_memes"
TARGET_GROUPS = ["race", "religion", "gender", "sexuality"]


def create_model(device):
    sarcasm_encoder = SarcasmBERT(
        model_name="bert-base-uncased",
        representation_dim=768,
        max_length=128,
        dropout=0.1,
    )

    sentiment_tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    sentiment_module = SentimentReversal(
        tokenizer=sentiment_tokenizer,
        model=sentiment_model,
        device=device,
    )

    model = MultimodalHateSpeechModel(
        sarcasm_encoder=sarcasm_encoder,
        sentiment_module=sentiment_module,
        fusion_dim=512,
        sarcasm_gate_dim=64,
        num_fusion_layers=2,
        num_heads=8,
        dropout=0.1,
    )
    return model.to(device)


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def main():
    checkpoint_path = PROJECT_ROOT / "artifacts" / "checkpoints_multitask_sarcasm_priority" / "best.pt"
    official_file = DATA_ROOT / "dev.jsonl"
    weak_file = DATA_ROOT / "dev_weak_labels.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("TARGET-GROUP EVALUATION")
    print("=" * 65)
    print("Device:", device)
    print("Checkpoint:", checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not official_file.exists() or not weak_file.exists():
        raise FileNotFoundError("Required dev annotation file is missing.")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = create_model(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    evaluator = MultimodalEvaluator(model=model, device=device)

    def evaluate_file(path):
        samples = load_hateful_memes(str(path), str(DATA_ROOT))
        dataset = MultimodalHateSpeechDataset(samples)
        loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            collate_fn=multimodal_collate_fn,
            pin_memory=(device.type == "cuda"),
        )
        return samples, evaluator.evaluate(loader)

    print("\nEvaluating official dev set...")
    official_samples, official_results = evaluate_file(official_file)
    print("Official samples:", len(official_samples))
    print("Official hate metrics:")
    print(json.dumps(official_results.get("metrics", {}).get("hate"), indent=2))

    print("\nEvaluating weakly annotated dev set...")
    weak_samples, weak_results = evaluate_file(weak_file)

    # Confirm that the weak-label file matches the official dev set.
    official_by_id = {str(s.sample_id): s for s in official_samples}
    weak_by_id = {str(s.sample_id): s for s in weak_samples}

    if official_by_id.keys() != weak_by_id.keys():
        raise ValueError("The official and weak-label sample IDs do not match.")

    for sample_id, sample in weak_by_id.items():
        official = official_by_id[sample_id]
        if sample.image_path != official.image_path:
            raise ValueError(f"Image mismatch for sample ID {sample_id}")
        if sample.hate_label != official.hate_label:
            raise ValueError(f"Hate-label mismatch for sample ID {sample_id}")

    predictions = weak_results.get("predictions", {})
    targets = weak_results.get("targets", {})
    masks = weak_results.get("label_masks", {})

    if "target" not in predictions or "target" not in targets:
        raise KeyError(
            "The evaluator did not return target predictions/targets. "
            "Inspect the returned keys before computing target metrics."
        )

    probabilities = as_numpy(predictions["target"])
    true_targets = as_numpy(targets["target"])

    # Only samples with target annotations are valid for this evaluation.
    target_mask = masks.get("target")
    if target_mask is not None:
        annotated = as_numpy(target_mask).astype(bool).reshape(-1)
    else:
        annotated = true_targets.sum(axis=1) > 0

    if true_targets.ndim != 2 or true_targets.shape[1] != len(TARGET_GROUPS):
        raise ValueError(
            f"Expected target labels shaped [N, 4], got {true_targets.shape}"
        )
    if probabilities.shape != true_targets.shape:
        raise ValueError(
            f"Prediction shape {probabilities.shape} does not match "
            f"target shape {true_targets.shape}"
        )

    count = int(annotated.sum())
    print(f"Weak-label samples with target annotations: {count}/{len(weak_samples)}")

    if count == 0:
        raise ValueError("No annotated target rows were found by the evaluator.")

    y_true = true_targets[annotated].astype(int)
    y_prob = probabilities[annotated]
    y_pred = (y_prob >= 0.5).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    print("\n" + "=" * 65)
    print("TARGET-GROUP METRICS (WEAK-LABEL EVALUATION)")
    print("=" * 65)
    print(f"{'Group':<12}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Support':>12}")

    group_results = {}
    for i, group in enumerate(TARGET_GROUPS):
        row = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        group_results[group] = row
        print(
            f"{group:<12}{precision[i]:>12.4f}"
            f"{recall[i]:>12.4f}{f1[i]:>12.4f}{support[i]:>12}"
        )

    report = {
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": str(device),
        "official_dev_samples": len(official_samples),
        "weak_label_samples": len(weak_samples),
        "target_annotated_samples": count,
        "target_group_order": TARGET_GROUPS,
        "target_group_metrics": group_results,
        "evaluation_note": (
            "Target-group metrics use weakly supervised annotations; "
            "they are not definitive ground-truth measurements."
        ),
    }

    output_path = PROJECT_ROOT / "artifacts" / "target_group_evaluation_multitask.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nReport saved to:", output_path)
    print("Existing checkpoints were not modified.")


if __name__ == "__main__":
    main()

