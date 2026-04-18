import torch
import torch.nn as nn


class DeepSpeech2(nn.Module):
    """
    DeepSpeech2-like model for ASR with CTC loss.
    Architecture: CNN + BiGRU + Linear
    
    Based on: https://proceedings.mlr.press/v48/amodei16.pdf
    """

    def __init__(
        self,
        n_tokens: int = 29,
        n_feats: int = 80,
        dim: int = 512,
        n_channels: int = 48,
        gru_layers: int = 5,
        dropout: float = 0.3,
    ):
        """
        Args:
            n_tokens (int): number of tokens in vocabulary (including blank)
            n_feats (int): number of mel spectrogram features (n_mels)
            dim (int): hidden dimension of GRU layers
            n_channels (int): number of channels in CNN layers
            gru_layers (int): number of GRU layers
            dropout (float): dropout probability
        """
        super().__init__()

        self.dim = dim
        self.time_reduction = 2  # Total time reduction factor from CNN

        # CNN feature extractor
        self.extractor = nn.Sequential(
            # First conv layer
            nn.Conv2d(
                in_channels=1,
                out_channels=n_channels,
                kernel_size=(41, 11),
                stride=(2, 2),
                padding=(20, 5),
                bias=False,
            ),
            nn.BatchNorm2d(n_channels),
            nn.Hardtanh(0, 20, inplace=True),
            
            # Second conv layer
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=(21, 11),
                stride=(2, 1),
                padding=(10, 5),
                bias=False,
            ),
            nn.BatchNorm2d(n_channels),
            nn.Hardtanh(0, 20, inplace=True),
        )

        # Frequency reduction factor
        self.freq_reduction = 4
        
        # Input size for GRU after CNN
        rnn_input_size = (n_feats // self.freq_reduction) * n_channels

        # Bidirectional GRU layers
        self.rnn_layers = nn.ModuleList()

        for i in range(gru_layers):
            gru = nn.GRU(
                input_size=rnn_input_size if i == 0 else dim,
                hidden_size=dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0.0,
            )
            self.rnn_layers.append(gru)

            # Add batch norm and dropout between GRU layers
            if i < gru_layers - 1:
                self.rnn_layers.append(nn.BatchNorm1d(dim))
                if dropout > 0:
                    self.rnn_layers.append(nn.Dropout(dropout))

        # Final linear layer
        self.fc = nn.Linear(dim, n_tokens, bias=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(
        self,
        spectrogram: torch.FloatTensor,
        spectrogram_length: torch.LongTensor,
        **batch,
    ) -> dict:
        """
        Forward pass.

        Args:
            spectrogram: (B, n_feats, T) - input mel spectrogram
            spectrogram_length: (B,) - sequence lengths

        Returns:
            dict with:
                - log_probs: (B, T', n_tokens) - log probabilities
                - log_probs_length: (B,) - new sequence lengths
        """
        # Add channel dimension: (B, n_feats, T) -> (B, 1, n_feats, T)
        x = spectrogram.unsqueeze(1)

        # CNN forward
        x = self.extractor(x)

        # Reshape: (B, C, F, T) -> (B, T, C * F)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, T, C * F)

        # GRU forward
        for layer in self.rnn_layers:
            if isinstance(layer, nn.GRU):
                x, _ = layer(x)
                # Combine bidirectional outputs
                x = x[..., :self.dim] + x[..., self.dim:]
            elif isinstance(layer, nn.BatchNorm1d):
                x = layer(x.transpose(1, 2)).transpose(1, 2)
            elif isinstance(layer, nn.Dropout):
                x = layer(x)

        # Final linear layer
        logits = self.fc(x)
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        # Calculate output lengths (time reduction from CNN)
        log_probs_length = spectrogram_length // self.time_reduction

        return {
            "log_probs": log_probs,
            "log_probs_length": log_probs_length,
        }

    def __str__(self):
        """Print model with number of parameters."""
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"

        return result_info
