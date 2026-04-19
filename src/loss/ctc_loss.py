import torch
from torch import Tensor
from torch.nn import CTCLoss


class CTCLossWrapper(CTCLoss):
    def forward(
        self, log_probs, log_probs_length, text_encoded, text_encoded_length, **batch
    ) -> Tensor:
        # Ensure tensors are on CPU for CTC loss computation
        log_probs = log_probs.cpu() if log_probs.is_cuda else log_probs
        text_encoded = text_encoded.cpu() if text_encoded.is_cuda else text_encoded
        log_probs_length = log_probs_length.cpu() if log_probs_length.is_cuda else log_probs_length
        text_encoded_length = text_encoded_length.cpu() if text_encoded_length.is_cuda else text_encoded_length
        
        # Transpose log_probs: (T, B, C) format for CTC loss
        log_probs_t = torch.transpose(log_probs, 0, 1)
        
        # Compute loss
        loss = super().forward(
            log_probs=log_probs_t,
            targets=text_encoded,
            input_lengths=log_probs_length,
            target_lengths=text_encoded_length,
        )
        
        # Return loss as a tensor that requires grad
        # The loss is already connected to the graph via log_probs
        return {"loss": loss}