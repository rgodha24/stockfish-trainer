import pytest
import stockfish_trainer


def test_sum_as_string():
    assert stockfish_trainer.sum_as_string(1, 1) == "2"
