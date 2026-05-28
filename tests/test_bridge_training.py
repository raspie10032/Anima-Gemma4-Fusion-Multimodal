from gemmanima.training.bridge_training import BridgeTrainingPlan


def test_bridge_training_plan_targets_4070_ti_super() -> None:
    plan = BridgeTrainingPlan()
    payload = plan.to_json_dict()

    assert payload["gpu_name"] == "RTX 4070 Ti SUPER"
    assert payload["gpu_index"] == 0
    assert "08_train_stream_batched.py" in payload["command"]
    assert "--save-every-shards 25" in payload["command"]
