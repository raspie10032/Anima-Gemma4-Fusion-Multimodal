import json
from pathlib import Path

from gemmanima.cli import main
from gemmanima.training.candidate_workflow import (
    build_candidate_objective_manifest,
    build_candidate_promotion_bundle,
    build_candidate_workflow_status,
)


def test_candidate_workflow_marks_rejected_fixed6_as_concept_proof(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    fixed6 = tmp_path / "fixed6.json"
    fixed6.write_text(
        json.dumps(
            {
                "status": "rejected",
                "protected_baseline": "v5",
                "promotion_allowed": False,
                "mean_v5_mse": 0.01,
                "mean_candidate_mse": 0.09,
                "mean_delta_vs_v5": 0.08,
                "regression_count": 1,
                "cases": [
                    {
                        "case_id": "text_eval_001_sign_luna_gate",
                        "v5_mse": 0.01,
                        "candidate_mse": 0.09,
                        "delta_vs_v5": 0.08,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    status = build_candidate_workflow_status(
        candidate_name="gemma_text_delta_300k_epoch1_a0p35",
        checkpoint=checkpoint,
        fixed6_summary=fixed6,
    )

    assert status["candidate"]["checkpoint_exists"] is True
    assert status["safety"]["v5_default_changed"] is False
    assert status["safety"]["cuda_visible_devices"] == "0"
    assert status["workflow_position"]["mode"] == "concept_proof_only"
    assert status["workflow_position"]["promotion_allowed"] is False
    assert status["gates"]["fixed6"]["regression_count"] == 1
    assert status["gates"]["fixed6"]["worst_regression"]["case_id"] == "text_eval_001_sign_luna_gate"


def test_candidate_workflow_reports_missing_fixed6_as_next_required_gate(tmp_path: Path) -> None:
    status = build_candidate_workflow_status(
        candidate_name="fresh_candidate",
        checkpoint=tmp_path / "fresh.pt",
    )

    assert status["workflow_position"]["current"] == "adapter_registered_needs_fixed6"
    assert status["workflow_position"]["next"] == "run_fixed6_gate"
    assert status["gates"]["fixed6"]["status"] == "missing"


def test_candidate_workflow_reads_general_quality_contact_sheet(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    contact = tmp_path / "contact.png"
    report = tmp_path / "general.json"
    contact.write_bytes(b"png")
    report.write_text(
        json.dumps(
            {
                "count": 2,
                "contact_sheet": str(contact),
                "purpose": "visual quality comparison",
                "cases": [
                    {"pair_mse_not_quality_score": 0.1},
                    {"pair_mse_not_quality_score": 0.3},
                ],
            }
        ),
        encoding="utf-8",
    )

    status = build_candidate_workflow_status(
        candidate_name="candidate",
        checkpoint=checkpoint,
        general_quality_report=report,
    )

    assert status["gates"]["general_quality"]["status"] == "review_ready"
    assert status["gates"]["general_quality"]["contact_sheet_exists"] is True
    assert status["gates"]["general_quality"]["pair_mse_mean_not_quality_score"] == 0.2


def test_candidate_workflow_reads_utf8_bom_json_artifacts(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    smoke = tmp_path / "smoke.json"
    output = tmp_path / "smoke.png"
    output.write_bytes(b"png")
    smoke.write_text(
        "\ufeff" + json.dumps({"output": str(output)}),
        encoding="utf-8",
    )

    status = build_candidate_workflow_status(
        candidate_name="candidate",
        checkpoint=checkpoint,
        smoke_report=smoke,
    )

    assert status["gates"]["smoke"]["status"] == "completed"
    assert status["gates"]["smoke"]["output_exists"] is True


def test_cli_candidate_workflow_status_writes_output(tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "candidate.pt"
    output = tmp_path / "status.json"
    checkpoint.write_bytes(b"checkpoint")

    code = main(
        [
            "candidate-workflow-status",
            "--candidate-name",
            "candidate",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status_path"] == str(output)
    assert output.exists()
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["candidate"]["name"] == "candidate"


def test_candidate_objective_manifest_turns_regressions_into_replay_cases(tmp_path: Path) -> None:
    donor = tmp_path / "donor.pt"
    fixed6 = tmp_path / "fixed6.json"
    donor.write_bytes(b"checkpoint")
    fixed6.write_text(
        json.dumps(
            {
                "status": "rejected",
                "regressions": [
                    {"case_id": "text_eval_005_handwritten_note_meet_at_dawn", "delta_vs_v5": 0.1888},
                    {"case_id": "text_eval_006_label_tea", "delta_vs_v5": 0.012},
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = build_candidate_objective_manifest(
        candidate_name="candidate",
        donor_checkpoint=donor,
        fixed6_summary=fixed6,
        target_sample_count=10000,
    )

    assert manifest["objective"]["mode"] == "hybrid_repair"
    assert manifest["lineage"]["style_donor_exists"] is True
    assert manifest["fixed6_replay"]["regression_count"] == 2
    assert manifest["fixed6_replay"]["cases"][0]["repair_priority"] == "hard"
    assert manifest["fixed6_replay"]["cases"][0]["replay_weight"] > manifest["fixed6_replay"]["cases"][1]["replay_weight"]
    assert manifest["next_gate"]["first"] == "fixed6"
    assert manifest["next_gate"]["promotion_allowed"] is False


def test_cli_candidate_objective_manifest_writes_output(tmp_path: Path, capsys) -> None:
    donor = tmp_path / "donor.pt"
    fixed6 = tmp_path / "fixed6.json"
    output = tmp_path / "objective.json"
    fixed6.write_text(json.dumps({"status": "rejected", "cases": []}), encoding="utf-8")

    code = main(
        [
            "candidate-objective-manifest",
            "--candidate-name",
            "candidate",
            "--donor-checkpoint",
            str(donor),
            "--fixed6-summary",
            str(fixed6),
            "--output",
            str(output),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest_path"] == str(output)
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["candidate"] == "candidate"


def test_candidate_promotion_bundle_blocks_rejected_fixed6(tmp_path: Path) -> None:
    fixed6 = tmp_path / "fixed6.json"
    smoke = tmp_path / "smoke.json"
    workflow = tmp_path / "workflow.json"
    fixed6.write_text("{}", encoding="utf-8")
    smoke.write_text("{}", encoding="utf-8")
    workflow.write_text(
        json.dumps(
            {
                "candidate": {"name": "candidate"},
                "gates": {
                    "fixed6": {"status": "rejected", "regression_count": 6, "source": str(fixed6)},
                    "smoke": {"status": "completed", "source": str(smoke)},
                    "general_quality": {"status": "review_ready"},
                },
            }
        ),
        encoding="utf-8",
    )

    bundle = build_candidate_promotion_bundle(workflow_status=workflow)

    assert bundle["candidate"] == "candidate"
    assert bundle["decision"]["status"] == "blocked"
    assert bundle["decision"]["default_update_allowed"] is False
    assert bundle["decision"]["reason"] == "fixed6_regression"
    assert bundle["default_change_package"]["write_defaults"] is False


def test_cli_candidate_promotion_bundle_writes_output(tmp_path: Path, capsys) -> None:
    workflow = tmp_path / "workflow.json"
    output = tmp_path / "promotion.json"
    workflow.write_text(
        json.dumps(
            {
                "candidate": {"name": "candidate"},
                "gates": {
                    "fixed6": {"status": "rejected", "regression_count": 1},
                    "smoke": {"status": "completed"},
                },
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "candidate-promotion-bundle",
            "--workflow-status",
            str(workflow),
            "--output",
            str(output),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["bundle_path"] == str(output)
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["decision"]["status"] == "blocked"
