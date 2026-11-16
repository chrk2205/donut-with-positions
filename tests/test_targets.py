import json
from pathlib import Path

import pytest

from donut_positions.targets import (
    aligned_mean_iou,
    boxes_from_ids,
    box_iou,
    cord_target,
    is_box,
    sequence_accuracy,
    structure_from_valid_line,
    words_to_target,
)


def _quad(x1, y1, x2, y2):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y1, "x3": x2, "y3": y2, "x4": x1, "y4": y2}


def _line(category, group_id, text, box):
    return {
        "category": category,
        "group_id": group_id,
        "words": [{"text": text, "quad": _quad(*box)}],
    }


def test_words_get_a_box_immediately_after_the_word():
    text = words_to_target(
        [
            {"text": "Tebu", "bins": [12, 40, 80, 55]},
            {"text": "Lemon", "bins": [90, 40, 140, 55]},
        ]
    )
    assert text == (
        "<s_synthdog>Tebu<pos_12><pos_40><pos_80><pos_55> "
        "Lemon<pos_90><pos_40><pos_140><pos_55></s>"
    )


def test_cord_box_sits_before_the_leaf_key_closes():
    gt = {"menu": {"nm": "Tebu Lemon", "price": "20.000"}}
    boxes = {"menu": {"nm": [12, 40, 140, 55], "price": [1, 2, 3, 4]}}
    text = cord_target(gt, boxes, task_token="<s_cord-v2>", eos_token="</s>")
    assert text == (
        "<s_cord-v2><s_menu>"
        "<s_price>20.000<pos_1><pos_2><pos_3><pos_4></s_price>"
        "<s_nm>Tebu Lemon<pos_12><pos_40><pos_140><pos_55></s_nm>"
        "</s_menu></s>"
    )
    assert text.endswith("</s_nm></s_menu></s>")


def test_menu_list_uses_sep_and_sub_is_nested():
    lines = [
        _line("menu.nm", 3, "Tea", (0, 0, 10, 4)),
        _line("menu.sub.nm", 3, "70%", (0, 5, 8, 9)),
        _line("menu.sub.nm", 3, "Less Ice", (0, 10, 20, 14)),
        _line("menu.nm", 4, "Toast", (0, 20, 12, 24)),
        _line("menu.sub_nm", 4, "Cheese", (0, 25, 16, 29)),
    ]
    tree = structure_from_valid_line(lines)
    assert isinstance(tree["menu"], list)
    assert tree["menu"][0]["sub"] == [{"nm": [0, 5, 8, 9]}, {"nm": [0, 10, 20, 14]}]
    assert tree["menu"][1]["sub"] == {"nm": [0, 25, 16, 29]}


def test_iou_and_edit_distance():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert box_iou([0, 0, 2, 2], [5, 5, 8, 8]) == 0
    assert aligned_mean_iou([[0, 0, 4, 4]], [[0, 0, 4, 4], [1, 1, 2, 2]]) == 0.5
    assert sequence_accuracy([1, 2, 3], [1, 2, 3]) == 1
    assert sequence_accuracy([1, 2], [1, 2, 3]) < 1


def _shape(node):
    if is_box(node):
        return "leaf"
    if isinstance(node, dict):
        return {key: _shape(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_shape(value) for value in node]
    return "leaf"


def test_validation_parses_match_the_box_tree():
    path = Path("/tmp/cord_val.parquet")
    if not path.is_file():
        pytest.skip("CORD validation parquet is not downloaded")
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["ground_truth"])
    for index in range(table.num_rows):
        ground_truth = json.loads(table.column("ground_truth")[index].as_py())
        built = _shape(structure_from_valid_line(ground_truth["valid_line"]))
        gold = _shape(ground_truth["gt_parse"])
        assert built == gold, index


def test_boxes_from_ids_reset_on_other_tokens():
    pos = {10: 1, 11: 2, 12: 3, 13: 4}
    assert boxes_from_ids([10, 11, 12, 13, 7, 10, 11], pos) == [[1, 2, 3, 4]]
