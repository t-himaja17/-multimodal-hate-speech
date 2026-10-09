"""
Training pipeline for the sarcasm-aware multimodal hate-speech model.

Member 4 ownership.

Pipeline:

    DataLoader
        ↓
    MultimodalHateSpeechModel
        ↓
    MultimodalTotalLoss
        ↓
    Backpropagation
        ↓
    Optimizer
        ↓
    Validation
        ↓
    Checkpointing

Supports:

    - mixed precision
    - gradient accumulation
    - gradient clipping
    - multiple training phases
    - validation
    - early stopping
    - checkpoint saving
    - loss tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from multimodal_hate.training.losses import MultimodalTotalLoss


# ============================================================
# TRAINING RESULT
# ============================================================


@dataclass
class EpochResult:
    """Metrics collected for one epoch."""

    loss: float
    hate_loss: float
    sarcasm_loss: float
    contrastive_loss: float
    target_loss: float
    batches: int


@dataclass
class TrainingHistory:
    """Complete training history."""

    train: list[EpochResult] = field(default_factory=list)
    validation: list[EpochResult] = field(default_factory=list)


# ============================================================
# TRAINER
# ============================================================


class MultimodalTrainer:
    """
    Trainer for MultimodalHateSpeechModel.

    Parameters
    ----------
    model:
        End-to-end multimodal hate-speech model.

    optimizer:
        PyTorch optimizer.

    loss_fn:
        MultimodalTotalLoss instance.

    device:
        Training device.

    gradient_accumulation_steps:
        Number of batches accumulated before optimizer update.

    mixed_precision:
        Enable automatic mixed precision.

    gradient_clipping:
        Maximum gradient norm.

    checkpoint_dir:
        Directory used for checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[torch.device | str] = None,
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = False,
        gradient_clipping: Optional[float] = None,
        checkpoint_dir: str | Path = "artifacts/checkpoints",
    ) -> None:
        if gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be greater than zero."
            )

        if gradient_clipping is not None and gradient_clipping <= 0:
            raise ValueError(
                "gradient_clipping must be greater than zero."
            )

        self.model = model

        self.optimizer = optimizer

        self.loss_fn = (
            loss_fn
            if loss_fn is not None
            else MultimodalTotalLoss()
        )

        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            device = torch.device(device)

        self.device = device

        self.gradient_accumulation_steps = (
            gradient_accumulation_steps
        )

        self.mixed_precision = (
            mixed_precision
            and self.device.type == "cuda"
        )

        self.gradient_clipping = gradient_clipping

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model.to(self.device)
        self.loss_fn.to(self.device)

        self.history = TrainingHistory()

        # CUDA AMP scaler.
        #
        # Newer PyTorch versions may use torch.amp instead,
        # but GradScaler remains broadly compatible.
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.mixed_precision
        )

    # ========================================================
    # DEVICE
    # ========================================================

    def _move_batch_to_device(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Move tensor values to the training device.

        Text and sample IDs remain Python objects.
        """

        moved = {}

        for key, value in batch.items():

            if isinstance(value, Tensor):
                moved[key] = value.to(
                    self.device,
                    non_blocking=True,
                )

            else:
                moved[key] = value

        return moved

    # ========================================================
    # FORWARD + LOSS
    # ========================================================

    def _forward_loss(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Tensor]:
        """
        Run model forward pass and calculate all losses.
        """

        outputs = self.model(
            images=batch["image"],
            texts=batch["text"],
        )

        losses = self.loss_fn(
            hate_logits=outputs.logits["hate"],
            hate_targets=batch["hate_target"],
            sarcasm_logits=outputs.logits["sarcasm"],
            sarcasm_targets=batch["sarcasm_target"],
            target_logits=outputs.logits["target"],
            target_targets=batch["target_target"],
            image_representation=outputs.image_representation,
            text_representation=outputs.text_representation,
        )

        return losses

    # ========================================================
    # TRAIN ONE EPOCH
    # ========================================================

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> EpochResult:
        """
        Train the model for one epoch.
        """

        self.model.train()

        self.optimizer.zero_grad(
            set_to_none=True
        )

        total_loss = 0.0
        total_hate = 0.0
        total_sarcasm = 0.0
        total_contrastive = 0.0
        total_target = 0.0

        num_batches = 0

        for batch_index, batch in enumerate(
            dataloader
        ):

            batch = self._move_batch_to_device(
                batch
            )

            with torch.cuda.amp.autocast(
                enabled=self.mixed_precision
            ):

                losses = self._forward_loss(
                    batch
                )

                loss = (
                    losses["total"]
                    / self.gradient_accumulation_steps
                )

            if self.mixed_precision:

                self.scaler.scale(
                    loss
                ).backward()

            else:

                loss.backward()

            should_update = (
                (batch_index + 1)
                % self.gradient_accumulation_steps
                == 0
            )

            is_last_batch = (
                batch_index + 1
                == len(dataloader)
            )

            if should_update or is_last_batch:

                if self.gradient_clipping is not None:

                    if self.mixed_precision:

                        self.scaler.unscale_(
                            self.optimizer
                        )

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clipping,
                    )

                if self.mixed_precision:

                    self.scaler.step(
                        self.optimizer
                    )

                    self.scaler.update()

                else:

                    self.optimizer.step()

                self.optimizer.zero_grad(
                    set_to_none=True
                )

            total_loss += (
                losses["total"]
                .detach()
                .item()
            )

            total_hate += (
                losses["hate"]
                .detach()
                .item()
            )

            total_sarcasm += (
                losses["sarcasm"]
                .detach()
                .item()
            )

            total_contrastive += (
                losses["contrastive"]
                .detach()
                .item()
            )

            total_target += (
                losses["target"]
                .detach()
                .item()
            )

            num_batches += 1

        if num_batches == 0:

            raise ValueError(
                "Training dataloader produced zero batches."
            )

        return EpochResult(
            loss=total_loss / num_batches,
            hate_loss=total_hate / num_batches,
            sarcasm_loss=total_sarcasm / num_batches,
            contrastive_loss=(
                total_contrastive / num_batches
            ),
            target_loss=total_target / num_batches,
            batches=num_batches,
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    @torch.no_grad()
    def validate_epoch(
        self,
        dataloader: DataLoader,
    ) -> EpochResult:
        """
        Evaluate the model without gradient updates.
        """

        self.model.eval()

        total_loss = 0.0
        total_hate = 0.0
        total_sarcasm = 0.0
        total_contrastive = 0.0
        total_target = 0.0

        num_batches = 0

        for batch in dataloader:

            batch = self._move_batch_to_device(
                batch
            )

            with torch.cuda.amp.autocast(
                enabled=self.mixed_precision
            ):

                losses = self._forward_loss(
                    batch
                )

            total_loss += (
                losses["total"]
                .detach()
                .item()
            )

            total_hate += (
                losses["hate"]
                .detach()
                .item()
            )

            total_sarcasm += (
                losses["sarcasm"]
                .detach()
                .item()
            )

            total_contrastive += (
                losses["contrastive"]
                .detach()
                .item()
            )

            total_target += (
                losses["target"]
                .detach()
                .item()
            )

            num_batches += 1

        if num_batches == 0:

            raise ValueError(
                "Validation dataloader produced zero batches."
            )

        return EpochResult(
            loss=total_loss / num_batches,
            hate_loss=total_hate / num_batches,
            sarcasm_loss=total_sarcasm / num_batches,
            contrastive_loss=(
                total_contrastive / num_batches
            ),
            target_loss=total_target / num_batches,
            batches=num_batches,
        )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        best_validation_loss: Optional[float] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> Path:
        """
        Save model, optimizer and training state.
        """

        checkpoint_path = (
            self.checkpoint_dir / filename
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": (
                self.model.state_dict()
            ),
            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),
            "best_validation_loss": (
                best_validation_loss
            ),
            "history": self.history,
        }

        if extra is not None:
            checkpoint["extra"] = extra

        if self.mixed_precision:
            checkpoint["scaler_state_dict"] = (
                self.scaler.state_dict()
            )

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        return checkpoint_path

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
    ) -> dict[str, Any]:
        """
        Load a training checkpoint.
        """

        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        if (
            self.mixed_precision
            and "scaler_state_dict" in checkpoint
        ):
            self.scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

        return checkpoint

    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        epochs: int,
        early_stopping_patience: Optional[int] = None,
    ) -> TrainingHistory:
        """
        Train the model and perform validation.

        Parameters
        ----------
        train_loader:
            Training DataLoader.

        validation_loader:
            Validation DataLoader.

        epochs:
            Number of epochs.

        early_stopping_patience:
            Number of epochs without validation improvement
            before stopping.
        """

        if epochs <= 0:
            raise ValueError(
                "epochs must be greater than zero."
            )

        if (
            early_stopping_patience is not None
            and early_stopping_patience <= 0
        ):
            raise ValueError(
                "early_stopping_patience must be "
                "greater than zero."
            )

        best_validation_loss = float("inf")

        epochs_without_improvement = 0

        for epoch in range(epochs):

            train_result = self.train_epoch(
                train_loader
            )

            validation_result = (
                self.validate_epoch(
                    validation_loader
                )
            )

            self.history.train.append(
                train_result
            )

            self.history.validation.append(
                validation_result
            )

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train={train_result.loss:.4f} | "
                f"val={validation_result.loss:.4f}"
            )

            print(
                "  train components: "
                f"hate={train_result.hate_loss:.4f}, "
                f"sarcasm={train_result.sarcasm_loss:.4f}, "
                f"contrastive="
                f"{train_result.contrastive_loss:.4f}, "
                f"target="
                f"{train_result.target_loss:.4f}"
            )

            print(
                "  val components: "
                f"hate={validation_result.hate_loss:.4f}, "
                f"sarcasm={validation_result.sarcasm_loss:.4f}, "
                f"contrastive="
                f"{validation_result.contrastive_loss:.4f}, "
                f"target="
                f"{validation_result.target_loss:.4f}"
            )

            # ------------------------------------------------
            # Save latest checkpoint
            # ------------------------------------------------

            self.save_checkpoint(
                filename="latest.pt",
                epoch=epoch + 1,
                best_validation_loss=(
                    best_validation_loss
                ),
            )

            # ------------------------------------------------
            # Best checkpoint
            # ------------------------------------------------

            if (
                validation_result.loss
                < best_validation_loss
            ):

                best_validation_loss = (
                    validation_result.loss
                )

                epochs_without_improvement = 0

                self.save_checkpoint(
                    filename="best.pt",
                    epoch=epoch + 1,
                    best_validation_loss=(
                        best_validation_loss
                    ),
                )

                print(
                    "  ✓ New best checkpoint saved."
                )

            else:

                epochs_without_improvement += 1

            # ------------------------------------------------
            # Early stopping
            # ------------------------------------------------

            if (
                early_stopping_patience is not None
                and epochs_without_improvement
                >= early_stopping_patience
            ):

                print(
                    "Early stopping triggered."
                )

                break

        return self.history