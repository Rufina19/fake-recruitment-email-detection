import torch
import torch.nn as nn


class Attention(nn.Module):
    """
    Simple additive attention over LSTM outputs.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, encoder_outputs, mask=None):
        # encoder_outputs: [batch, seq_len, hidden_dim]
        energy = torch.tanh(self.proj(encoder_outputs))  # [batch, seq_len, hidden_dim]
        energy = energy @ self.v  # [batch, seq_len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(energy, dim=1)  # [batch, seq_len]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(
            1
        )  # [batch, hidden_dim]
        return context, attn_weights

