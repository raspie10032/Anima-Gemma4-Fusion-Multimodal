from gemmanima.cli import main


def test_cli_rebalance_requires_completed_shards(capsys) -> None:
    code = main(["rebalance-targets", "--completed-4070-shards", "9999", "--json"])

    assert code == 0
    assert "remaining_rows" in capsys.readouterr().out
