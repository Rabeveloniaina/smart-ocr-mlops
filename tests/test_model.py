import torch
import pytest
from src.models.crnn import CRNN
from src.models.ctc_decoder import CTCDecoder


def test_crnn_forward_shape():
    charset = "abcdefghijklmnopqrstuvwxyz"
    num_classes = len(charset) + 1
    
    model = CRNN(num_classes=num_classes, img_height=32, hidden_size=64, num_rnn_layers=1, pretrained=False)
    batch = torch.randn(2, 1, 32, 128)
    output = model(batch)
    
    T, B, C = output.shape
    assert B == 2
    assert C == num_classes
    assert T > 0


def test_ctc_decoder_greedy():
    charset = "abc"
    decoder = CTCDecoder(charset=charset)
    
    T = 6
    B = 1
    C = 4
    
    logits = torch.zeros(T, B, C)
    logits[0, 0, 0] = 10.0
    logits[1, 0, 1] = 10.0
    logits[2, 0, 1] = 10.0
    logits[3, 0, 0] = 10.0
    logits[4, 0, 2] = 10.0
    logits[5, 0, 2] = 10.0
    
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
    results = decoder.greedy_decode(log_probs)
    
    assert len(results) == 1
    assert results[0].text == "ab"
