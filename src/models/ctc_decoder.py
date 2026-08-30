import torch
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DecodingResult:
    text: str
    confidence: float
    log_probability: float


class CTCDecoder:
    def __init__(self, charset: str, blank_idx: int = 0):
        self.charset = charset
        self.blank_idx = blank_idx

        self.idx_to_char = {0: "[BLANK]"}
        for i, char in enumerate(charset):
            self.idx_to_char[i + 1] = char

        self.char_to_idx = {v: k for k, v in self.idx_to_char.items()}
        self.num_classes = len(charset) + 1

    def greedy_decode(self, log_probs: torch.Tensor) -> List[DecodingResult]:
        best_indices = torch.argmax(log_probs, dim=2)
        best_indices = best_indices.permute(1, 0)

        results = []
        for b in range(best_indices.size(0)):
            indices = best_indices[b].cpu().numpy()

            probs = torch.exp(log_probs[:, b, :])
            max_probs = probs.max(dim=1).values
            confidence = float(max_probs.mean().item()) * 100

            text = self._collapse_sequence(indices)
            log_prob = float(log_probs[:, b, :].max(dim=1).values.sum().item())

            results.append(DecodingResult(
                text=text,
                confidence=round(confidence, 2),
                log_probability=log_prob
            ))

        return results

    def beam_search_decode(
        self,
        log_probs: torch.Tensor,
        beam_width: int = 5,
    ) -> List[DecodingResult]:
        T, B, C = log_probs.shape
        results = []

        for b in range(B):
            seq_log_probs = log_probs[:, b, :].cpu().numpy()
            beams = [("", 0.0)]

            for t in range(T):
                new_beams = {}

                for prefix, score in beams:
                    for c in range(C):
                        char_log_prob = seq_log_probs[t, c]
                        char = self.idx_to_char.get(c, "[BLANK]")

                        if char == "[BLANK]":
                            new_prefix = prefix
                        elif prefix and prefix[-1] == char:
                            new_prefix = prefix
                        else:
                            new_prefix = prefix + char

                        new_score = score + char_log_prob

                        if new_prefix in new_beams:
                            old_score = new_beams[new_prefix]
                            new_beams[new_prefix] = np.logaddexp(old_score, new_score)
                        else:
                            new_beams[new_prefix] = new_score

                beams = sorted(
                    new_beams.items(), key=lambda x: x[1], reverse=True
                )[:beam_width]

            best_text, best_log_prob = beams[0]
            confidence = min(100.0, float(np.exp(best_log_prob / max(len(best_text), 1)) * 100))

            results.append(DecodingResult(
                text=best_text,
                confidence=round(confidence, 2),
                log_probability=float(best_log_prob)
            ))

        return results

    def _collapse_sequence(self, indices: np.ndarray) -> str:
        chars = []
        prev_idx = None

        for idx in indices:
            if idx == self.blank_idx:
                prev_idx = idx
                continue
            if idx != prev_idx:
                char = self.idx_to_char.get(int(idx), "?")
                if char != "[BLANK]":
                    chars.append(char)
            prev_idx = idx

        return "".join(chars)

    def encode(self, text: str) -> Tuple[List[int], int]:
        indices = []
        for char in text:
            idx = self.char_to_idx.get(char)
            if idx is not None and idx != self.blank_idx:
                indices.append(idx)

        return indices, len(indices)

    def decode_batch(
        self,
        log_probs: torch.Tensor,
        use_beam_search: bool = False,
        beam_width: int = 5,
    ) -> List[DecodingResult]:
        if use_beam_search:
            return self.beam_search_decode(log_probs, beam_width=beam_width)
        return self.greedy_decode(log_probs)
