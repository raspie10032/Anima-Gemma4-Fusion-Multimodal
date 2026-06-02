import json
from pathlib import Path

from PIL import Image

from gemmanima.cli import main
from gemmanima.training.comparison import (
    GenerationCompareReport,
    compare_images_if_available,
    compare_text_roi_if_available,
    write_compare_report,
)


def test_compare_report_records_prompt_seed_and_conditioning_distance(tmp_path: Path) -> None:
    report = GenerationCompareReport(
        prompt="moonlit forest",
        seed=424242,
        teacher_image=tmp_path / "teacher.png",
        student_image=tmp_path / "student.png",
        student_checkpoint=tmp_path / "bridge.pt",
        conditioning_mse=0.0031,
    )

    payload = report.to_json_dict()

    assert payload["prompt"] == "moonlit forest"
    assert payload["seed"] == 424242
    assert payload["conditioning"]["mse"] == 0.0031
    assert payload["images"]["teacher"] == str(tmp_path / "teacher.png")


def test_compare_images_if_available_returns_pixel_metrics(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher.png"
    student = tmp_path / "student.png"
    Image.new("RGB", (2, 2), (0, 0, 0)).save(teacher)
    Image.new("RGB", (2, 2), (255, 255, 255)).save(student)

    metrics = compare_images_if_available(teacher, student)

    assert metrics["shape"] == [2, 2, 3]
    assert metrics["mse"] == 1.0
    assert metrics["mae"] == 1.0


def test_compare_text_roi_if_available_measures_named_crop(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher.png"
    student = tmp_path / "student.png"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(teacher)
    Image.new("RGB", (4, 4), (0, 0, 0)).save(student)
    edited = Image.open(student)
    edited.putpixel((2, 2), (255, 255, 255))
    edited.save(student)

    metrics = compare_text_roi_if_available(
        teacher,
        student,
        roi={"name": "center_2x2", "box": [1, 1, 3, 3]},
    )

    assert metrics["roi"]["name"] == "center_2x2"
    assert metrics["roi"]["box"] == [1, 1, 3, 3]
    assert metrics["shape"] == [2, 2, 3]
    assert metrics["mse"] == 0.25
    assert metrics["mae"] == 0.25


def test_compare_report_includes_optional_text_roi_metrics(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher.png"
    student = tmp_path / "student.png"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(teacher)
    Image.new("RGB", (4, 4), (0, 0, 0)).save(student)
    edited = Image.open(student)
    edited.putpixel((2, 2), (255, 255, 255))
    edited.save(student)
    report = GenerationCompareReport(
        prompt="jar label that reads TEA",
        seed=424242,
        teacher_image=teacher,
        student_image=student,
        student_checkpoint=tmp_path / "bridge.pt",
        text_roi={"name": "center_2x2", "box": [1, 1, 3, 3]},
    )

    payload = report.to_json_dict()

    assert payload["text_roi_metrics"]["roi"]["name"] == "center_2x2"
    assert payload["text_roi_metrics"]["mse"] == 0.25


def test_write_compare_report_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "compare.json"
    report = GenerationCompareReport(
        prompt="moonlit forest",
        seed=424242,
        teacher_image=tmp_path / "teacher.png",
        student_image=tmp_path / "student.png",
        student_checkpoint=tmp_path / "bridge.pt",
        conditioning_mse=0.0031,
    )

    written = write_compare_report(report, out)

    assert written == out
    assert json.loads(out.read_text(encoding="utf-8"))["conditioning"]["mse"] == 0.0031


def test_cli_write_compare_report(tmp_path: Path, capsys) -> None:
    out = tmp_path / "compare.json"
    code = main(
        [
            "write-compare-report",
            "--prompt",
            "moonlit forest",
            "--seed",
            "424242",
            "--teacher-image",
            str(tmp_path / "teacher.png"),
            "--student-image",
            str(tmp_path / "student.png"),
            "--student-checkpoint",
            str(tmp_path / "bridge.pt"),
            "--conditioning-mse",
            "0.0031",
            "--output",
            str(out),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["report_path"] == str(out)
