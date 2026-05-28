import torch

from gemmanima.rendering.gemma_hidden import GemmaHiddenProvider, normalize_gemma_hidden


class RuntimeWithIntent:
    def __init__(self) -> None:
        self.calls = []

    def encode_image_intent(self, source_text: str, span_text: str):
        self.calls.append((source_text, span_text))
        return torch.ones(4, 1536)


class RuntimeWithEncodeOnly:
    def __init__(self) -> None:
        self.encoded = []

    def encode(self, text: str):
        self.encoded.append(text)
        return torch.ones(3, 1536)


def test_gemma_hidden_provider_prefers_image_intent_contract() -> None:
    runtime = RuntimeWithIntent()
    provider = GemmaHiddenProvider(runtime)

    hidden = provider.encode_image_intent("full chat", "visible prompt")

    assert runtime.calls == [("full chat", "visible prompt")]
    assert hidden.shape == (1, 4, 1536)
    assert hidden.dtype == torch.float32


def test_gemma_hidden_provider_falls_back_to_span_encode() -> None:
    runtime = RuntimeWithEncodeOnly()
    provider = GemmaHiddenProvider(runtime)

    hidden = provider.encode_image_intent("full chat", "visible prompt")

    assert runtime.encoded == ["visible prompt"]
    assert hidden.shape == (1, 3, 1536)


def test_normalize_gemma_hidden_rejects_wrong_width() -> None:
    try:
        normalize_gemma_hidden(torch.ones(1, 4, 1024))
    except ValueError as exc:
        assert "1536" in str(exc)
    else:
        raise AssertionError("expected ValueError")
