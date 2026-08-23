import pytest
import torch

from multimodal_hate.models.sarcasm.gate import SarcasmGate


def test_gate_shape():
    gate = SarcasmGate(gate_dim=16)

    output = gate(
        torch.rand(4, 1),
        torch.rand(4, 1),
        torch.rand(4, 1),
    )

    assert output.shape == (4, 16)


def test_gate_range():
    gate = SarcasmGate(gate_dim=16)

    output = gate(
        torch.rand(4, 1),
        torch.rand(4, 1),
        torch.rand(4, 1),
    )

    assert torch.all(output >= 0)
    assert torch.all(output <= 1)


def test_single_batch():
    gate = SarcasmGate(gate_dim=8)

    output = gate(
        torch.tensor([[0.7]]),
        torch.tensor([[0.8]]),
        torch.tensor([[1.0]]),
    )

    assert output.shape == (1, 8)


def test_batch_support():
    gate = SarcasmGate(gate_dim=32)

    output = gate(
        torch.rand(10, 1),
        torch.rand(10, 1),
        torch.rand(10, 1),
    )

    assert output.shape == (10, 32)


def test_configurable_dimension():
    gate = SarcasmGate(gate_dim=64)

    assert gate.gate_dim == 64

    output = gate(
        torch.rand(2, 1),
        torch.rand(2, 1),
        torch.rand(2, 1),
    )

    assert output.shape == (2, 64)


def test_gradients():
    gate = SarcasmGate(gate_dim=8)

    sarcasm = torch.rand(4, 1, requires_grad=True)
    incongruity = torch.rand(4, 1, requires_grad=True)
    reversal = torch.rand(4, 1, requires_grad=True)

    output = gate(
        sarcasm,
        incongruity,
        reversal,
    )

    loss = output.sum()
    loss.backward()

    assert sarcasm.grad is not None
    assert incongruity.grad is not None
    assert reversal.grad is not None

    for parameter in gate.parameters():
        assert parameter.grad is not None


def test_deterministic_without_dropout():
    gate = SarcasmGate(
        gate_dim=8,
        dropout=0.0,
    )

    gate.eval()

    sarcasm = torch.tensor([[0.5], [0.8]])
    incongruity = torch.tensor([[0.2], [0.9]])
    reversal = torch.tensor([[0.0], [1.0]])

    first = gate(
        sarcasm,
        incongruity,
        reversal,
    )

    second = gate(
        sarcasm,
        incongruity,
        reversal,
    )

    assert torch.equal(first, second)


def test_batch_mismatch_rejected():
    gate = SarcasmGate(gate_dim=8)

    with pytest.raises(ValueError):
        gate(
            torch.rand(4, 1),
            torch.rand(3, 1),
            torch.rand(4, 1),
        )


def test_wrong_shape_rejected():
    gate = SarcasmGate(gate_dim=8)

    with pytest.raises(ValueError):
        gate(
            torch.rand(4, 2),
            torch.rand(4, 1),
            torch.rand(4, 1),
        )


def test_non_tensor_rejected():
    gate = SarcasmGate(gate_dim=8)

    with pytest.raises(TypeError):
        gate(
            [0.5],
            torch.rand(1, 1),
            torch.rand(1, 1),
        )


def test_invalid_gate_dimension():
    with pytest.raises(ValueError):
        SarcasmGate(gate_dim=0)


def test_invalid_hidden_dimension():
    with pytest.raises(ValueError):
        SarcasmGate(
            gate_dim=8,
            hidden_dim=0,
        )


def test_invalid_dropout():
    with pytest.raises(ValueError):
        SarcasmGate(
            gate_dim=8,
            dropout=1.0,
        )


def test_device_mismatch_rejected():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    gate = SarcasmGate(gate_dim=8)

    with pytest.raises(ValueError):
        gate(
            torch.rand(2, 1),
            torch.rand(2, 1, device="cuda"),
            torch.rand(2, 1),
        )