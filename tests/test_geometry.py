import random

from donut_positions.geometry import Canvas, bin_box, bin_coord, fit_canvas


def test_bin_edges():
    assert bin_coord(0, 100) == 0
    assert bin_coord(100, 100) == 511
    assert bin_coord(50, 100) == 256


def test_letterbox_pad():
    canvas = fit_canvas(100, 200, 1280, 960, random_pad=False, rng=random.Random(0))
    assert canvas.resized_width == 640
    assert canvas.resized_height == 1280
    assert canvas.pad_left == 160
    assert canvas.pad_top == 0


def test_pad_moves_the_bin():
    left = Canvas(100, 100, 1.0, 0, 0, 100, 100)
    right = Canvas(100, 100, 1.0, 40, 0, 60, 100)
    assert bin_box((0, 0, 10, 10), right)[0] > bin_box((0, 0, 10, 10), left)[0]
