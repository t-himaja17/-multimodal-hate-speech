import pytest
import torch

from multimodal_hate.models.fusion.projection import (
    ModalityProjection,
    ProjectionAlignment,
)

from multimodal_hate.models.fusion.sarcasm_bias import (
    SarcasmConditionedAttentionBias,
)

from multimodal_hate.models.fusion.fusion_transformer import (
    CrossModalFusionLayer,
    SarcasmAwareFusionTransformer,
)

from multimodal_hate.models.classification.heads import (
    ClassificationHead,
    HateClassificationHead,
    SarcasmClassificationHead,
    TargetGroupClassificationHead,
    MultitaskClassificationHeads,
)


# ============================================================
# PROJECTION TESTS
# ============================================================

def test_image_projection_shape():
    module = ModalityProjection(input_dim=1024, fusion_dim=512)
    x = torch.randn(2, 257, 1024)
    output = module(x)
    assert output.shape == (2, 257, 512)


def test_text_projection_shape():
    module = ModalityProjection(input_dim=768, fusion_dim=512)
    x = torch.randn(2, 128, 768)
    output = module(x)
    assert output.shape == (2, 128, 512)


def test_projection_alignment_shapes():
    module = ProjectionAlignment(
        image_input_dim=1024,
        text_input_dim=768,
        fusion_dim=512,
    )

    image = torch.randn(4, 257, 1024)
    text = torch.randn(4, 128, 768)

    projected_image, projected_text = module(image, text)

    assert projected_image.shape == (4, 257, 512)
    assert projected_text.shape == (4, 128, 512)


def test_projection_alignment_supports_different_sequence_lengths():
    module = ProjectionAlignment(
        image_input_dim=1024,
        text_input_dim=768,
        fusion_dim=512,
    )

    image = torch.randn(2, 10, 1024)
    text = torch.randn(2, 37, 768)

    projected_image, projected_text = module(image, text)

    assert projected_image.shape == (2, 10, 512)
    assert projected_text.shape == (2, 37, 512)


def test_modality_embeddings_are_learnable():
    module = ProjectionAlignment(
        image_input_dim=1024,
        text_input_dim=768,
        fusion_dim=512,
    )

    assert isinstance(module.img_token, torch.nn.Parameter)
    assert isinstance(module.txt_token, torch.nn.Parameter)
    assert module.img_token.requires_grad
    assert module.txt_token.requires_grad


def test_projection_gradients_flow():
    module = ProjectionAlignment(
        image_input_dim=1024,
        text_input_dim=768,
        fusion_dim=512,
    )

    image = torch.randn(2, 10, 1024, requires_grad=True)
    text = torch.randn(2, 12, 768, requires_grad=True)

    projected_image, projected_text = module(image, text)

    loss = projected_image.mean() + projected_text.mean()
    loss.backward()

    assert image.grad is not None
    assert text.grad is not None
    assert module.img_token.grad is not None
    assert module.txt_token.grad is not None


def test_projection_rejects_wrong_image_dimension():
    module = ProjectionAlignment(
        image_input_dim=1024,
        text_input_dim=768,
        fusion_dim=512,
    )

    image = torch.randn(2, 10, 768)
    text = torch.randn(2, 10, 768)

    with pytest.raises(ValueError):
        module(image, text)


def test_projection_rejects_wrong_text_dimension():
    module = ProjectionAlignment(
        image_input_dim=1024,
        text_input_dim=768,
        fusion_dim=512,
    )

    image = torch.randn(2, 10, 1024)
    text = torch.randn(2, 10, 512)

    with pytest.raises(ValueError):
        module(image, text)


def test_projection_rejects_batch_mismatch():
    module = ProjectionAlignment(
        image_input_dim=1024,
        text_input_dim=768,
        fusion_dim=512,
    )

    image = torch.randn(2, 10, 1024)
    text = torch.randn(3, 10, 768)

    with pytest.raises(ValueError):
        module(image, text)


def test_projection_rejects_non_3d_image_input():
    module = ModalityProjection(
        input_dim=1024,
        fusion_dim=512,
    )

    image = torch.randn(2, 1024)

    with pytest.raises(ValueError):
        module(image)


# ============================================================
# SARCASM-CONDITIONED ATTENTION BIAS TESTS
# ============================================================

def test_sarcasm_bias_shape():
    module = SarcasmConditionedAttentionBias(
        gate_dim=64,
        num_heads=8,
    )

    sarcasm_gate = torch.randn(4, 64)
    bias = module(sarcasm_gate)

    assert bias.shape == (4, 8, 1, 1)


def test_sarcasm_bias_works_with_non_512_gate_dimension():
    module = SarcasmConditionedAttentionBias(
        gate_dim=37,
        num_heads=8,
    )

    sarcasm_gate = torch.randn(2, 37)
    bias = module(sarcasm_gate)

    assert bias.shape == (2, 8, 1, 1)


def test_sarcasm_bias_broadcasts_to_attention_logits():
    module = SarcasmConditionedAttentionBias(
        gate_dim=64,
        num_heads=8,
    )

    sarcasm_gate = torch.randn(2, 64)

    attention_logits = torch.randn(
        2,
        8,
        10,
        20,
    )

    output = module.apply(
        attention_logits,
        sarcasm_gate,
    )

    assert output.shape == attention_logits.shape


def test_sarcasm_bias_changes_attention_logits():
    module = SarcasmConditionedAttentionBias(
        gate_dim=64,
        num_heads=8,
    )

    sarcasm_gate = torch.randn(2, 64)

    attention_logits = torch.zeros(
        2,
        8,
        10,
        20,
    )

    output = module.apply(
        attention_logits,
        sarcasm_gate,
    )

    assert not torch.allclose(
        output,
        attention_logits,
    )


def test_sarcasm_bias_gradient_flow():
    module = SarcasmConditionedAttentionBias(
        gate_dim=64,
        num_heads=8,
    )

    sarcasm_gate = torch.randn(
        2,
        64,
        requires_grad=True,
    )

    bias = module(sarcasm_gate)

    loss = bias.mean()
    loss.backward()

    assert sarcasm_gate.grad is not None
    assert module.projection.weight.grad is not None
    assert module.projection.bias.grad is not None


def test_sarcasm_bias_rejects_wrong_gate_shape():
    module = SarcasmConditionedAttentionBias(
        gate_dim=64,
        num_heads=8,
    )

    sarcasm_gate = torch.randn(2, 5, 64)

    with pytest.raises(ValueError):
        module(sarcasm_gate)


def test_sarcasm_bias_rejects_wrong_gate_dimension():
    module = SarcasmConditionedAttentionBias(
        gate_dim=64,
        num_heads=8,
    )

    sarcasm_gate = torch.randn(2, 32)

    with pytest.raises(ValueError):
        module(sarcasm_gate)


def test_sarcasm_bias_rejects_wrong_attention_head_count():
    module = SarcasmConditionedAttentionBias(
        gate_dim=64,
        num_heads=8,
    )

    sarcasm_gate = torch.randn(2, 64)

    attention_logits = torch.randn(
        2,
        4,
        10,
        20,
    )

    with pytest.raises(ValueError):
        module.apply(
            attention_logits,
            sarcasm_gate,
        )


def test_sarcasm_bias_rejects_batch_mismatch():
    module = SarcasmConditionedAttentionBias(
        gate_dim=64,
        num_heads=8,
    )

    sarcasm_gate = torch.randn(3, 64)

    attention_logits = torch.randn(
        2,
        8,
        10,
        20,
    )

    with pytest.raises(ValueError):
        module.apply(
            attention_logits,
            sarcasm_gate,
        )


# ============================================================
# CROSS-MODAL FUSION TESTS
# ============================================================

def test_cross_modal_fusion_layer_shapes():
    module = CrossModalFusionLayer(
        fusion_dim=512,
        num_heads=8,
        dropout=0.1,
        sarcasm_gate_dim=64,
    )

    image_tokens = torch.randn(2, 10, 512)
    text_tokens = torch.randn(2, 12, 512)
    sarcasm_gate = torch.randn(2, 64)

    output_image, output_text = module(
        image_tokens,
        text_tokens,
        sarcasm_gate,
    )

    assert output_image.shape == (2, 10, 512)
    assert output_text.shape == (2, 12, 512)


def test_cross_modal_fusion_supports_different_sequence_lengths():
    module = CrossModalFusionLayer(
        fusion_dim=512,
        num_heads=8,
        dropout=0.1,
        sarcasm_gate_dim=64,
    )

    image_tokens = torch.randn(2, 17, 512)
    text_tokens = torch.randn(2, 43, 512)
    sarcasm_gate = torch.randn(2, 64)

    output_image, output_text = module(
        image_tokens,
        text_tokens,
        sarcasm_gate,
    )

    assert output_image.shape == (2, 17, 512)
    assert output_text.shape == (2, 43, 512)


def test_fusion_transformer_default_configuration():
    module = SarcasmAwareFusionTransformer()

    assert module.fusion_dim == 512
    assert module.num_layers == 4
    assert module.num_heads == 8
    assert module.dropout == 0.1
    assert len(module.layers) == 4


def test_fusion_transformer_output_shapes():
    module = SarcasmAwareFusionTransformer(
        fusion_dim=512,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        sarcasm_gate_dim=64,
    )

    image_tokens = torch.randn(2, 10, 512)
    text_tokens = torch.randn(2, 15, 512)
    sarcasm_gate = torch.randn(2, 64)

    fused_tokens, pooled = module(
        image_tokens,
        text_tokens,
        sarcasm_gate,
    )

    assert fused_tokens.shape == (2, 25, 512)
    assert pooled.shape == (2, 512)


def test_fusion_transformer_preserves_batch_size():
    module = SarcasmAwareFusionTransformer(
        sarcasm_gate_dim=32,
    )

    image_tokens = torch.randn(4, 8, 512)
    text_tokens = torch.randn(4, 13, 512)
    sarcasm_gate = torch.randn(4, 32)

    fused_tokens, pooled = module(
        image_tokens,
        text_tokens,
        sarcasm_gate,
    )

    assert fused_tokens.size(0) == 4
    assert pooled.size(0) == 4


def test_fusion_transformer_supports_tbd_gate_dimension():
    module = SarcasmAwareFusionTransformer(
        sarcasm_gate_dim=37,
    )

    image_tokens = torch.randn(2, 6, 512)
    text_tokens = torch.randn(2, 9, 512)
    sarcasm_gate = torch.randn(2, 37)

    fused_tokens, pooled = module(
        image_tokens,
        text_tokens,
        sarcasm_gate,
    )

    assert fused_tokens.shape == (2, 15, 512)
    assert pooled.shape == (2, 512)


def test_fusion_transformer_gradient_flow():
    module = SarcasmAwareFusionTransformer(
        sarcasm_gate_dim=64,
    )

    image_tokens = torch.randn(
        2,
        6,
        512,
        requires_grad=True,
    )

    text_tokens = torch.randn(
        2,
        8,
        512,
        requires_grad=True,
    )

    sarcasm_gate = torch.randn(
        2,
        64,
        requires_grad=True,
    )

    fused_tokens, pooled = module(
        image_tokens,
        text_tokens,
        sarcasm_gate,
    )

    loss = fused_tokens.mean() + pooled.mean()
    loss.backward()

    assert image_tokens.grad is not None
    assert text_tokens.grad is not None
    assert sarcasm_gate.grad is not None


def test_fusion_transformer_rejects_wrong_image_dimension():
    module = SarcasmAwareFusionTransformer(
        fusion_dim=512,
        sarcasm_gate_dim=64,
    )

    image_tokens = torch.randn(2, 10, 256)
    text_tokens = torch.randn(2, 10, 512)
    sarcasm_gate = torch.randn(2, 64)

    with pytest.raises(ValueError):
        module(
            image_tokens,
            text_tokens,
            sarcasm_gate,
        )


def test_fusion_transformer_rejects_wrong_text_dimension():
    module = SarcasmAwareFusionTransformer(
        fusion_dim=512,
        sarcasm_gate_dim=64,
    )

    image_tokens = torch.randn(2, 10, 512)
    text_tokens = torch.randn(2, 10, 256)
    sarcasm_gate = torch.randn(2, 64)

    with pytest.raises(ValueError):
        module(
            image_tokens,
            text_tokens,
            sarcasm_gate,
        )


def test_fusion_transformer_rejects_batch_mismatch():
    module = SarcasmAwareFusionTransformer(
        sarcasm_gate_dim=64,
    )

    image_tokens = torch.randn(2, 10, 512)
    text_tokens = torch.randn(3, 10, 512)
    sarcasm_gate = torch.randn(2, 64)

    with pytest.raises(ValueError):
        module(
            image_tokens,
            text_tokens,
            sarcasm_gate,
        )


def test_cross_modal_layer_has_required_attention_blocks():
    module = CrossModalFusionLayer(
        fusion_dim=512,
        num_heads=8,
        dropout=0.1,
        sarcasm_gate_dim=64,
    )

    assert isinstance(
        module.image_to_text,
        torch.nn.MultiheadAttention,
    )

    assert isinstance(
        module.text_to_image,
        torch.nn.MultiheadAttention,
    )

    assert isinstance(
        module.fused_self_attention,
        torch.nn.MultiheadAttention,
    )


def test_all_four_layers_have_sarcasm_bias():
    module = SarcasmAwareFusionTransformer(
        sarcasm_gate_dim=64,
    )

    assert len(module.layers) == 4

    for layer in module.layers:
        assert hasattr(layer, "sarcasm_bias")
        assert layer.sarcasm_bias.gate_dim == 64
        assert layer.sarcasm_bias.num_heads == 8


# ============================================================
# CLASSIFICATION HEAD TESTS
# ============================================================

def test_generic_classification_head_shape():
    """Generic head should produce the configured number of logits."""
    module = ClassificationHead(
        input_dim=512,
        output_dim=3,
        dropout=0.1,
    )

    x = torch.randn(4, 512)

    output = module(x)

    assert output.shape == (4, 3)


def test_hate_classification_head_shape():
    """Hate head must produce two class logits."""
    module = HateClassificationHead(
        input_dim=512,
        dropout=0.1,
    )

    x = torch.randn(4, 512)

    output = module(x)

    assert output.shape == (4, 2)


def test_sarcasm_classification_head_shape():
    """Sarcasm head must produce two class logits."""
    module = SarcasmClassificationHead(
        input_dim=512,
        dropout=0.1,
    )

    x = torch.randn(4, 512)

    output = module(x)

    assert output.shape == (4, 2)


def test_target_group_classification_head_shape():
    """Target head must produce five independent target logits."""
    module = TargetGroupClassificationHead(
        input_dim=512,
        dropout=0.1,
    )

    x = torch.randn(4, 512)

    output = module(x)

    assert output.shape == (4, 5)


def test_target_group_names():
    """Target-group ordering must match the project specification."""
    module = TargetGroupClassificationHead()

    assert module.target_groups == (
        "race",
        "religion",
        "gender",
        "disability",
        "sexuality",
    )


def test_multitask_classification_heads_shapes():
    """All three classification outputs must have correct shapes."""
    module = MultitaskClassificationHeads(
        input_dim=512,
        dropout=0.1,
    )

    x = torch.randn(4, 512)

    outputs = module(x)

    assert set(outputs.keys()) == {
        "hate",
        "sarcasm",
        "target",
    }

    assert outputs["hate"].shape == (4, 2)
    assert outputs["sarcasm"].shape == (4, 2)
    assert outputs["target"].shape == (4, 5)


def test_classification_heads_return_logits():
    """Heads must return raw logits, not probabilities."""
    module = MultitaskClassificationHeads(
        input_dim=512,
        dropout=0.0,
    )

    x = torch.randn(8, 512)

    outputs = module(x)

    for logits in outputs.values():
        assert torch.isfinite(logits).all()


def test_classification_heads_gradient_flow():
    """Gradients must flow through every classification head."""
    module = MultitaskClassificationHeads(
        input_dim=512,
        dropout=0.1,
    )

    x = torch.randn(
        4,
        512,
        requires_grad=True,
    )

    outputs = module(x)

    loss = (
        outputs["hate"].mean()
        + outputs["sarcasm"].mean()
        + outputs["target"].mean()
    )

    loss.backward()

    assert x.grad is not None

    for parameter in module.parameters():
        assert parameter.grad is not None


def test_classification_head_rejects_wrong_rank():
    """Classification input must be [B, D]."""
    module = ClassificationHead(
        input_dim=512,
        output_dim=2,
    )

    x = torch.randn(2, 10, 512)

    with pytest.raises(ValueError):
        module(x)


def test_classification_head_rejects_wrong_dimension():
    """Classification input dimension must match the configured size."""
    module = ClassificationHead(
        input_dim=512,
        output_dim=2,
    )

    x = torch.randn(4, 256)

    with pytest.raises(ValueError):
        module(x)


def test_multitask_heads_reject_wrong_input_dimension():
    """Multitask heads must reject incompatible pooled representations."""
    module = MultitaskClassificationHeads(
        input_dim=512,
    )

    x = torch.randn(4, 256)

    with pytest.raises(ValueError):
        module(x)