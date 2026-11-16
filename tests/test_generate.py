import random

from donut_positions.generate import render_page
from donut_positions.targets import quad_to_xyxy


def test_rendered_words_stay_inside_the_page():
    image, words = render_page(random.Random(0), min_words=6, max_words=12)
    assert len(words) >= 6
    for word in words:
        x1, y1, x2, y2 = quad_to_xyxy(word["quad"])
        assert word["text"]
        assert 0 <= x1 < x2 <= image.width
        assert 0 <= y1 < y2 <= image.height
