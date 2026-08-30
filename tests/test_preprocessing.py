import numpy as np
import pytest
from src.preprocessing import ImagePreprocessor


def test_preprocessor_output_dimensions():
    preprocessor = ImagePreprocessor(target_height=32, target_width=128)
    dummy_bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    processed = preprocessor.preprocess_image(dummy_bgr)
    
    assert processed.shape == (32, 128)
    assert processed.dtype == np.uint8


def test_preprocessor_wide_image():
    preprocessor = ImagePreprocessor(target_height=32, target_width=128)
    dummy_wide = np.ones((30, 300), dtype=np.uint8) * 200
    processed = preprocessor.preprocess_image(dummy_wide)
    
    assert processed.shape == (32, 128)


def test_preprocessor_blank_padding():
    preprocessor = ImagePreprocessor(target_height=32, target_width=128)
    dummy_small = np.zeros((32, 32), dtype=np.uint8)
    processed = preprocessor.preprocess_image(dummy_small)
    
    assert processed.shape == (32, 128)
    assert processed[:, 40:].mean() == 255.0
