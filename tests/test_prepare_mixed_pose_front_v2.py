from scripts.prepare_mixed_pose_front_v2 import (
    explicit_tags_in_safe_rows,
    first_position_stats,
    prepare_row,
)


def test_prepare_row_moves_pose_action_after_subject_and_sanitizes_safe_explicit():
    row = {
        "id": 1,
        "rating": "s",
        "target_tags": (
            "1girl, solo, long hair, pink hair, large breasts, standing, "
            "holding sword, looking at viewer, sex, forest"
        ),
        "flat_tags": [
            "1girl",
            "solo",
            "long hair",
            "pink hair",
            "large breasts",
            "standing",
            "holding sword",
            "looking at viewer",
            "sex",
            "forest",
        ],
        "taxonomy_groups": {
            "subject": ["1girl", "solo"],
            "appearance": ["long hair", "pink hair"],
            "body_focus": ["large breasts"],
            "pose_action": ["standing", "holding sword", "looking at viewer", "sex"],
            "setting": ["forest"],
        },
        "core_tags": [
            "1girl",
            "solo",
            "long hair",
            "pink hair",
            "large breasts",
            "standing",
            "holding sword",
            "looking at viewer",
            "sex",
            "forest",
        ],
    }

    prepared, removed = prepare_row(row)

    assert removed == ["sex"]
    assert prepared["flat_tags"][:5] == [
        "1girl",
        "solo",
        "standing",
        "holding sword",
        "looking at viewer",
    ]
    assert "sex" not in prepared["flat_tags"]
    assert explicit_tags_in_safe_rows([prepared]) == {}
    assert first_position_stats([prepared], "pose_action")["median"] == 3


def test_prepare_row_preserves_explicit_pose_action_for_explicit_rating():
    row = {
        "id": 2,
        "rating": "e",
        "flat_tags": ["1girl", "solo", "blue hair", "sex", "vaginal", "lying"],
        "target_tags": "1girl, solo, blue hair, sex, vaginal, lying",
        "taxonomy_groups": {
            "subject": ["1girl", "solo"],
            "appearance": ["blue hair"],
            "pose_action": ["sex", "vaginal", "lying"],
        },
        "core_tags": ["1girl", "solo", "blue hair", "sex", "vaginal", "lying"],
    }

    prepared, removed = prepare_row(row)

    assert removed == []
    assert prepared["flat_tags"][:5] == ["1girl", "solo", "sex", "vaginal", "lying"]
