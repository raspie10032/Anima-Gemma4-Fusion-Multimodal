import torch

from gemmanima.rendering.t5_tokenizer import T5TokenizerProvider, t5_tokenizer_environment


class FakeT5Tokenizer:
    def __init__(self) -> None:
        self.calls = []

    def tokenize_with_weights(self, prompt: str, return_word_ids: bool = False):
        self.calls.append((prompt, return_word_ids))
        return [[(11, 1.0), (22, 0.5), (33, 1.25)]]


def test_t5_tokenizer_provider_returns_ids_and_weights() -> None:
    tokenizer = FakeT5Tokenizer()
    provider = T5TokenizerProvider(tokenizer)

    ids, weights = provider.encode_ids_weights("bright forest")

    assert tokenizer.calls == [("bright forest", False)]
    assert ids.tolist() == [11, 22, 33]
    assert ids.dtype == torch.int32
    assert weights.tolist() == [1.0, 0.5, 1.25]
    assert weights.dtype == torch.float32


def test_t5_tokenizer_environment_reports_module_without_loading_tokenizer() -> None:
    env = t5_tokenizer_environment(load_tokenizer=False)

    assert env["provider_module"] is True
    assert env["tokenizer_load_attempted"] is False
