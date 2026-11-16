import torch
from transformers.models.mbart.modeling_mbart import MBartLearnedPositionalEmbedding

from donut_positions.train import extend_decoder_positions


class _Config:
    def __init__(self, value):
        self.max_position_embeddings = value


def test_decoder_positions_grow_without_dropping_old_rows():
    positions = MBartLearnedPositionalEmbedding(4, 3)
    with torch.no_grad():
        positions.weight.zero_()
        positions.weight[0].fill_(1)
    decoder = type("Decoder", (), {})()
    decoder.embed_positions = positions
    decoder.max_target_positions = 4
    model = type("Model", (), {})()
    model.decoder = type("Wrapper", (), {})()
    model.decoder.model = type("Inner", (), {})()
    model.decoder.model.decoder = decoder
    model.decoder.config = _Config(4)
    model.config = type("Top", (), {})()
    model.config.decoder = _Config(4)

    extend_decoder_positions(model, 6)

    grown = model.decoder.model.decoder.embed_positions
    assert grown.num_embeddings == 8
    assert torch.all(grown.weight[0] == 1)
    assert model.decoder.config.max_position_embeddings == 6
    assert model.decoder.model.decoder.max_target_positions == 6
