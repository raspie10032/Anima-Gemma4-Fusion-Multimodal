from gemmanima.training.bridge_training import BridgeTrainingPlan


def test_bridge_training_plan_targets_4070_ti_super() -> None:
    plan = BridgeTrainingPlan()
    payload = plan.to_json_dict()

    assert payload["gpu_name"] == "RTX 4070 Ti SUPER"
    assert payload["gpu_index"] == 0
    assert "$env:CUDA_VISIBLE_DEVICES='0'; & " in payload["command"]
    assert "08_train_stream_batched.py" in payload["command"]
    assert "--save-every-shards 25" in payload["command"]


def test_bridge_training_plan_supports_poc1_limited_shard_run(tmp_path) -> None:
    plan = BridgeTrainingPlan(
        target_dir=tmp_path / "targets",
        gemma_dir=tmp_path / "gemma",
        output=tmp_path / "poc1_bridge.pt",
        epochs=1,
        val=100,
        prefetch_gb=4.0,
        limit_shards=1,
    )
    payload = plan.to_json_dict()

    assert payload["limit_shards"] == 1
    assert "--limit-shards 1" in payload["command"]
    assert str(tmp_path / "poc1_bridge.pt") in payload["command"]
