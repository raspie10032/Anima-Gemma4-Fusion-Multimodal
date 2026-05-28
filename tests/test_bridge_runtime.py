from gemmanima.modules.bridge_runtime import BridgeForwardSummary


def test_bridge_forward_summary_json() -> None:
    summary = BridgeForwardSummary(
        checkpoint="E:/x.pt",
        input_shape=(1, 16, 1536),
        t5_ids_shape=(1, 32),
        output_shape=(1, 32, 1024),
        finite=True,
        val_mse=0.001,
        epoch=2,
    )

    data = summary.to_json_dict()
    assert data["output_shape"] == [1, 32, 1024]
    assert data["finite"] is True
