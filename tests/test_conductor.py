from pathlib import Path

from gemmanima.core.conductor import GemmAnimaConductor
from gemmanima.core.schemas import JobStatus, Mode


def make_conductor(tmp_path: Path) -> GemmAnimaConductor:
    return GemmAnimaConductor(
        session_id="test-session",
        manifest_root=tmp_path / "manifests",
        image_root=tmp_path / "images",
    )


def test_normal_chat_does_not_run_image_path(tmp_path: Path) -> None:
    conductor = make_conductor(tmp_path)

    response = conductor.handle_user_message("오늘 구조 설계 계속하자")

    assert response.mode == Mode.CHAT
    assert response.status == JobStatus.COMPLETED
    assert response.output_path is None
    assert list((tmp_path / "images").glob("*.dryrun.txt")) == []


def test_explicit_image_request_generates_manifest_and_dryrun_output(tmp_path: Path) -> None:
    conductor = make_conductor(tmp_path)

    response = conductor.handle_user_message("원신의 나히다를 밝은 숲 배경 애니 일러스트로 그려줘")

    assert response.mode == Mode.GENERATE_IMAGE
    assert response.status == JobStatus.COMPLETED
    assert response.prompt is not None
    assert "나히다" in response.prompt
    assert response.manifest_path and response.manifest_path.exists()
    assert response.output_path and response.output_path.exists()
    manifest_text = response.manifest_path.read_text(encoding="utf-8")
    assert "RTX 4070 Ti SUPER" in manifest_text
    assert "adapter_model.safetensors" in manifest_text


def test_dryrun_generation_message_does_not_claim_image_was_created(tmp_path: Path) -> None:
    conductor = make_conductor(tmp_path)

    response = conductor.handle_user_message("draw a moonlit garden")

    assert response.output_path and response.output_path.suffix == ".txt"
    assert "dry-run" in response.message
    assert "실제 이미지" in response.message
    assert "생성하지 않았습니다" in response.message
    assert "prompt:" not in response.message


def test_vague_image_request_asks_one_clarifying_question(tmp_path: Path) -> None:
    conductor = make_conductor(tmp_path)

    response = conductor.handle_user_message("그려줘")

    assert response.mode == Mode.GENERATE_IMAGE
    assert response.status == JobStatus.ASK_CLARIFY
    assert response.output_path is None
    assert response.message.endswith("?") or response.message.endswith("까?")


def test_context_capsule_omits_technical_context_from_prompt(tmp_path: Path) -> None:
    conductor = make_conductor(tmp_path)
    conductor.handle_user_message("RTX 4070 Ti SUPER 전력 문제랑 체크포인트를 계속 봐줘")
    conductor.handle_user_message("푸른 숲에서 작은 신 같은 캐릭터 그림이 좋아")

    response = conductor.handle_user_message("그걸 이미지로 만들어줘")

    assert response.status == JobStatus.COMPLETED
    assert response.prompt is not None
    assert "푸른 숲" in response.prompt
    assert "4070" not in response.prompt
    assert "체크포인트" not in response.prompt


def test_context_capsule_omits_chat_only_history_from_image_prompt(tmp_path: Path) -> None:
    conductor = make_conductor(tmp_path)
    chat = conductor.handle_user_message("hi")

    response = conductor.handle_user_message("draw tiny forest fairy anime")

    assert chat.mode == Mode.CHAT
    assert response.status == JobStatus.COMPLETED
    assert response.prompt is not None
    assert "draw tiny forest fairy anime" in response.prompt
    assert "hi" not in response.prompt.lower()
