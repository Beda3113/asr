 
import re
from collections import defaultdict
from enum import Enum
from pathlib import Path
from string import ascii_lowercase

import torch
import numpy as np


class DecoderType(Enum):
    ARGMAX = "argmax"
    BS = "bs"           # Your own beam search
    BS_TORCH = "bs_torch"
    BS_LM = "bs_lm"


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(
        self,
        alphabet=None,
        decoder_type=DecoderType.ARGMAX,
        beam_size=10,
        **kwargs,
    ):
        if isinstance(decoder_type, str):
            decoder_type = DecoderType(decoder_type)

        self.decoder_type = decoder_type
        self.beam_size = beam_size

        self.alphabet = alphabet or list(ascii_lowercase + " ")
        self.vocab = [self.EMPTY_TOK] + self.alphabet

        self.ind2char = {i: c for i, c in enumerate(self.vocab)}
        self.char2ind = {c: i for i, c in self.ind2char.items()}

        self.blank_idx = self.char2ind[self.EMPTY_TOK]

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def __call__(
        self, log_probs: torch.FloatTensor, log_probs_length: torch.LongTensor
    ) -> list[str] | list[list[str]]:
        if self.decoder_type == DecoderType.ARGMAX:
            preds = log_probs.argmax(-1).cpu().numpy()
            return [[self.ctc_decode(p[:l])] for p, l in zip(preds, log_probs_length)]

        if self.decoder_type == DecoderType.BS:
            results = []
            for lp, l in zip(log_probs, log_probs_length):
                hyps = self.ctc_beam_search(lp.exp(), int(l), self.beam_size)
                results.append([text for text, prob in hyps])
            return results

        # For other decoders, fallback to argmax
        preds = log_probs.argmax(-1).cpu().numpy()
        return [[self.ctc_decode(p[:l])] for p, l in zip(preds, log_probs_length)]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        if inds is None or len(inds) == 0:
            return ""
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        """
        CTC decoding: remove blanks and duplicates.

        Args:
            inds (list): list of token indices from argmax
        Returns:
            text (str): decoded text
        """
        if inds is None or len(inds) == 0:
            return ""

        blank_idx = self.blank_idx
        decoded = []
        prev = blank_idx

        for idx in inds:
            idx = int(idx)
            if idx != blank_idx and idx != prev:
                decoded.append(self.ind2char[idx])
            prev = idx

        return "".join(decoded)

    def ctc_beam_search(
        self, probs: torch.FloatTensor, length: int, beam_size: int = 10
    ) -> list[tuple[str, float]]:
        """
        Vanilla CTC beam search decoder (hand-crafted, no external libraries).

        Args:
            probs: probability matrix of shape (T, vocab_size)
            length: actual length of the sequence
            beam_size: number of beams to keep
        Returns:
            list of (text, probability) tuples sorted by probability (descending)
        """
        blank_idx = self.blank_idx

        # Initialize: (prefix, last_char) -> probability
        beams = {("", blank_idx): 1.0}

        for t in range(length):
            next_beams = defaultdict(float)
            current_probs = probs[t].cpu().numpy()

            for (prefix, last_char), prefix_prob in beams.items():
                for char_idx, char_prob in enumerate(current_probs):
                    if char_prob < 1e-10:
                        continue

                    cur_char = self.ind2char[char_idx]
                    new_prob = prefix_prob * char_prob

                    # CTC merging rule
                    if char_idx == blank_idx or char_idx == last_char:
                        new_prefix = prefix
                    else:
                        new_prefix = prefix + cur_char

                    key = (new_prefix, char_idx)
                    next_beams[key] = max(next_beams[key], new_prob)

            # Keep only top beam_size beams
            beams = dict(sorted(next_beams.items(), key=lambda x: -x[1])[:beam_size])

        # Aggregate probabilities for identical prefixes
        final_beams = defaultdict(float)
        for (prefix, _), prob in beams.items():
            final_beams[prefix] += prob

        # Sort by probability descending
        sorted_beams = sorted(final_beams.items(), key=lambda x: -x[1])

        return sorted_beams

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
