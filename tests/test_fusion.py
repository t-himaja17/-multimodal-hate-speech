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


# ============================================================
# PROJECTION TESTS
# ============================================================

def test_image_projection_shape():
    """1024-dimensional image tokens should become 512-dimensional."""
    module = ModalityProjection(
        input_dim=1024,
        fusion_dim=512,
    )

    x = torch.randn(2, 257, 1024)

    output = module(x)

    assert output.shape == (2, 257, 512)


def test_text_projection_shape():
    """768-dimensional text tokens should become 512-dimensional."""
    module = ModalityProjection(
        input_dim=768,
        fusion_dim=512,
    )

    x = torch.randn(2, 128, 768)

    output = module(x)

    assert output.shape == (2, 128, 512)


def test_projection_alignment_shapes():
    """Image and text should both enter the shared 512-D fusion space."""
    module = ProjectionAlignment(
        image_input_dim=1024,
        text_input_dim=768,
        fusion_dim=512,
    )

    image = torch.randn(4, 257, 1024)
    text = torch.randn(4, 128, 768)

    projected_image, projected_text = module(
        image,
        text,
    )

    assert projected_image.shape == (4, 257, 512)
    assert projected_text.shape == (4, 128, 512)


def test_projection_alignment_supports_different_sequence_lengths():
    """N_img and N_txt must remain independently variable."""
    module = ProjectionAlignment(
        image_input_dim=1024,
        text_input_dim=768,
        fusion_dim=512,
    )

    image = torch.randn(2, 10, 1024)
    text = torch.randn(2, 37, 768)

    projected_image, projected_text = module(
        image,
        text,
    )

    assert projected_image.shape == (2, 10, 512)
    assert projected_text.shape == (2, 37, 512)


def test_modality_embeddings_are_learnable():
    """IMG_TOKEN and TXT_TOKEN must be trainable parameters."""
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
    """Gradients must flow through both projection branches."""
    module = ProjectionAlignment(
        image_input_dim=1024,
        text_input_dim=768,
        fusion_dim=512,
    )

    image = torch.randn(
        2,
        10,
        1024,
        requires_grad=True,
    )

    text = torch.randn(
        2,
        12,
        768,
        requires_grad=True,
    )

    projected_image, projected_text = module(
        image,
        text,
    )

    loss = projected_image.mean() + projected_text.mean()

    loss.backward()

    assert image.grad is not None
    assert text.grad is not None

    assert module.img_token.grad is not None
    assert module.txt_token.grad is not None


def test_projection_rejects_wrong_image_dimension():
    """Incorrect image feature dimensions must fail clearly."""
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
    """Incorrect text feature dimensions must fail clearly."""
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
    """Image and text batch sizes must match."""
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
    """Image tokens must have [B, N, D] shape."""
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
    """Sarcasm gate should become one bias value per attention head."""
    module = SarcasmConditionedAttentionBias(
        gate_dim=64,
        num_heads=8,
    )

    sarcasm_gate = torch.randn(4, 64)

    bias = module(sarcasm_gate)

    assert bias.shape == (4, 8, 1, 1)


def test_sarcasm_bias_works_with_non_512_gate_dimension():
    """Gate dimension must remain independent of fusion dimension."""
    module = SarcasmConditionedAttentionBias(
        gate_dim=37,
        num_heads=8,
    )

    sarcasm_gate = torch.randn(2, 37)

    bias = module(sarcasm_gate)

    assert bias.shape == (2, 8, 1, 1)


def test_sarcasm_bias_broadcasts_to_attention_logits():
    """The bias must broadcast across query and key positions."""
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
    """Adding a non-zero learned bias should modify the logits."""
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
    """Gradients must flow from attention bias back into the gate."""
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
    """Gate must have shape [B, d]."""
    module = SarcasmConditionedAttentionBias(
        gate_dim=64,
        num_heads=8,
    )

    sarcasm_gate = torch.randn(
        2,
        5,
        64,
    )

    with pytest.raises(ValueError):
        module(sarcasm_gate)


def test_sarcasm_bias_rejects_wrong_gate_dimension():
    """Incorrect gate dimension must fail clearly."""
    module = SarcasmConditionedAttentionBias(
        gate_dim=64,
        num_heads=8,
    )

    sarcasm_gate = torch.randn(
        2,
        32,
    )

    with pytest.raises(ValueError):
        module(sarcasm_gate)


def test_sarcasm_bias_rejects_wrong_attention_head_count():
    """Attention logits must use the configured number of heads."""
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
    """Attention logits and gate must have the same batch size."""
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
    """One fusion layer must preserve image/text token dimensions."""
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
    """Image and text sequence lengths must remain independent."""
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
    """Transformer must follow the methodology defaults."""
    module = SarcasmAwareFusionTransformer()

    assert module.fusion_dim == 512
    assert module.num_layers == 4
    assert module.num_heads == 8
    assert module.dropout == 0.1

    assert len(module.layers) == 4


def test_fusion_transformer_output_shapes():
    """Complete transformer must produce fused and pooled outputs."""
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
    """Batch dimension must remain unchanged."""
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
    """
    Sarcasm gate dimension must not be hard-coded to 512.

    The interface currently marks this dimension as TBD.
    """
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
    """Gradients must flow through the complete fusion transformer."""
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
    """Image tokens must use the configured fusion dimension."""
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
    """Text tokens must use the configured fusion dimension."""
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
    """All three inputs must have matching batch sizes."""
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
    """Each layer must contain all three required attention mechanisms."""
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
    """Every fusion layer must receive sarcasm-conditioned attention."""
    module = SarcasmAwareFusionTransformer(
        sarcasm_gate_dim=64,
    )

    assert len(module.layers) == 4

    for layer in module.layers:
        assert hasattr(
            layer,
            "sarcasm_bias",
        )

        assert layer.sarcasm_bias.gate_dim == 64
        assert layer.sarcasm_bias.num_heads == 8