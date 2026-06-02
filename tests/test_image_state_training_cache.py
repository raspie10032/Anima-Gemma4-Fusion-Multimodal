from pathlib import Path

import torch

from scripts.train_image_state_translator import TensorFileCache, expand_guard_replay_examples, load_guard_replay_indices


def test_tensor_file_cache_reuses_loaded_tensor(tmp_path: Path) -> None:
    path = tmp_path / "embed.pt"
    torch.save(torch.ones(2, 3), path)
    cache = TensorFileCache(max_bytes=1024 * 1024)

    first = cache.load(path)
    second = cache.load(path)

    assert torch.equal(first, second)
    assert cache.stats["misses"] == 1
    assert cache.stats["hits"] == 1
    assert cache.stats["items"] == 1


def test_tensor_file_cache_evicts_oldest_when_capacity_is_exceeded(tmp_path: Path) -> None:
    first_path = tmp_path / "first.pt"
    second_path = tmp_path / "second.pt"
    torch.save(torch.ones(16, dtype=torch.float32), first_path)
    torch.save(torch.zeros(16, dtype=torch.float32), second_path)
    cache = TensorFileCache(max_bytes=80)

    cache.load(first_path)
    cache.load(second_path)
    cache.load(first_path)

    assert cache.stats["evictions"] >= 1
    assert cache.stats["misses"] == 3


def test_load_guard_replay_indices_reads_unique_indices(tmp_path: Path) -> None:
    replay = tmp_path / "replay.jsonl"
    replay.write_text('{"idx": 7}\n{"idx": 7}\n{"idx": 9}\n', encoding="utf-8")

    assert load_guard_replay_indices(replay) == {7, 9}


def test_expand_guard_replay_examples_oversamples_matching_examples() -> None:
    examples = [{"idx": 0}, {"idx": 1}, {"idx": 2}]

    expanded = expand_guard_replay_examples(examples, guard_indices={1}, replay_weight=4)

    assert [item["idx"] for item in expanded] == [0, 1, 2, 1, 1, 1]
    assert expanded[-1]["guard_replay"] is True
