import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


import argparse
import torch

from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from src.multimodal_hate.data.loaders.hateful_memes import (
    load_hateful_memes,
)

from src.multimodal_hate.training.dataset import (
    MultimodalHateSpeechDataset,
    multimodal_collate_fn,
)

from src.multimodal_hate.models.multimodal_model import (
    MultimodalHateSpeechModel,
)

from src.multimodal_hate.models.sarcasm.sarcasm_bert import (
    SarcasmBERT,
)

from src.multimodal_hate.models.sarcasm.sentiment import (
    SentimentReversal,
)

from src.multimodal_hate.evaluation.evaluator import (
    MultimodalEvaluator,
)


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hateful_memes"
)

DEV_FILE = DATA_ROOT / "dev.jsonl"
IMAGE_DIR = DATA_ROOT


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "artifacts/"
            "checkpoints_fusion2/"
            "best.pt"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    return parser.parse_args()


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(device):

    sarcasm_tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased"
    )

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


    # IMPORTANT:
    # This MUST match the checkpoint architecture.
    model = MultimodalHateSpeechModel(
        sarcasm_encoder=sarcasm_encoder,
        sentiment_module=sentiment_module,
        fusion_dim=512,
        sarcasm_gate_dim=64,
        num_fusion_layers=2,
        num_heads=8,
        dropout=0.1,
    )


    model.to(device)

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()


    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if (
        args.device == "cuda"
        and torch.cuda.is_available()
    ):

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")


    checkpoint_path = (
        PROJECT_ROOT
        / args.checkpoint
    )


    print("=" * 70)
    print("2-LAYER MODEL EVALUATION")
    print("=" * 70)

    print(
        "DEVICE:",
        device,
    )

    print(
        "CHECKPOINT:",
        checkpoint_path,
    )


    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )


    print()
    print(
        "CHECKPOINT EPOCH:",
        checkpoint.get("epoch"),
    )


    print(
        "CHECKPOINT BEST VALIDATION LOSS:",
        checkpoint.get("best_val_loss"),
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = create_model(
        device
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()


    # --------------------------------------------------------
    # OFFICIAL DEV DATA
    # --------------------------------------------------------

    dev_samples = load_hateful_memes(
        str(DEV_FILE),
        str(IMAGE_DIR),
    )


    print()
    print(
        "DEV SAMPLES:",
        len(dev_samples),
    )


    # --------------------------------------------------------
    # VERIFY OFFICIAL LABELS
    # --------------------------------------------------------

    non_hate = 0
    hate = 0
    missing = 0


    for sample in dev_samples:

        if sample.hate_label is None:

            missing += 1

        elif int(sample.hate_label) == 0:

            non_hate += 1

        elif int(sample.hate_label) == 1:

            hate += 1


    print()
    print("=" * 70)
    print("OFFICIAL DEV LABELS")
    print("=" * 70)

    print(
        "Non-Hate:",
        non_hate,
    )

    print(
        "Hate:",
        hate,
    )

    print(
        "Missing:",
        missing,
    )


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dev_dataset = MultimodalHateSpeechDataset(
        dev_samples
    )


    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=multimodal_collate_fn,
        pin_memory=True,
    )


    # --------------------------------------------------------
    # EVALUATOR
    # --------------------------------------------------------

    evaluator = MultimodalEvaluator(
        model=model,
        device=device,
    )


    print()
    print("=" * 70)
    print("RUNNING OFFICIAL DEV EVALUATION")
    print("=" * 70)


    results = evaluator.evaluate(
        dev_loader
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("2-LAYER MODEL RESULTS")
    print("=" * 70)


    if isinstance(results, dict):

        for key, value in results.items():

            print(
                f"{key}: {value}"
            )

    else:

        print(results)


    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()