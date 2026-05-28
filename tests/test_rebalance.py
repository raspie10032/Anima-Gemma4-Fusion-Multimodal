from pathlib import Path

from gemmanima.training.rebalance import build_rebalance_subsets


def test_build_rebalance_subsets_skips_completed_4070_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(f'{{"idx": {i}, "text": "x"}}' for i in range(10)) + "\n", encoding="utf-8")

    plan = build_rebalance_subsets(
        completed_4070_shards=0,
        source_subset=source,
        output_dir=tmp_path / "out",
        share_4070=0.5,
    )

    assert plan.remaining_rows == 10
    assert plan.rows_for_4070 == 5
    assert plan.rows_for_5060 == 5
    assert plan.subset_4070.exists()
    assert plan.subset_5060.exists()
