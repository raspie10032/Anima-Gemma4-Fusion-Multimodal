import pytest

from gemmanima.core.schemas import GenerationPlan


def test_generation_plan_from_dict_validates_known_shape() -> None:
    plan = GenerationPlan.from_dict(
        {
            "prompt": "nahida in a luminous forest",
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "cfg": 3.5,
            "lora_stack": ["style-a"],
        }
    )

    assert plan.prompt == "nahida in a luminous forest"
    assert plan.lora_stack == ("style-a",)


def test_generation_plan_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown generation plan keys"):
        GenerationPlan.from_dict({"prompt": "x", "root_access": True})


def test_generation_plan_rejects_bad_dimensions() -> None:
    with pytest.raises(ValueError, match="multiples of 8"):
        GenerationPlan.from_dict({"prompt": "x", "width": 1025, "height": 1024})
