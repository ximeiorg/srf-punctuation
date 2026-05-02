import pytest
from srf_punctuation.config import Config
from srf_punctuation.data import DataProcessor


def test_extract_punctuation_labels():
    config = Config()
    processor = DataProcessor(config)

    text = "今天天气很好，我们出去散步吧。"
    clean_text, labels = processor.extract_punctuation_labels(text)

    assert clean_text == "今天天气很好我们出去散步吧"
    assert len(labels) == len(clean_text)
    assert labels[5] == 1  # "好" 后面是逗号
    assert labels[-1] == 2  # "吧" 后面是句号


def test_extract_question():
    config = Config()
    processor = DataProcessor(config)

    text = "你喜欢吃苹果吗？"
    clean_text, labels = processor.extract_punctuation_labels(text)

    assert clean_text == "你喜欢吃苹果吗"
    assert labels[-1] == 3  # 问号


def test_extract_exclamation():
    config = Config()
    processor = DataProcessor(config)

    text = "太棒了！"
    clean_text, labels = processor.extract_punctuation_labels(text)

    assert clean_text == "太棒了"
    assert labels[-1] == 4  # 感叹号