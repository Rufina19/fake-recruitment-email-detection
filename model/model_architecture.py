import torch
import torch.nn as nn
from transformers import BertModel

try:
    # When imported as part of the `model` package (e.g. from backend)
    from .attention_layer import Attention  # type: ignore
except ImportError:
    # When run as a standalone script from within the `model` directory
    from attention_layer import Attention


class HybridBertCnnLstmAttention(nn.Module):
    """
    BERT -> CNN -> LSTM -> Attention -> Dense classifier.
    Outputs two probabilities: fraud and genuine.
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        cnn_out_channels: int = 128,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 1,
        num_classes: int = 2,
        freeze_bert: bool = False,
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # CNN over token representations (sequence dimension)
        self.cnn = nn.Conv1d(
            in_channels=bert_hidden_size,
            out_channels=cnn_out_channels,
            kernel_size=3,
            padding=1,
        )

        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Name must match the attribute used during training ("attn")
        self.attn = Attention(hidden_dim=lstm_hidden_dim * 2)

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(lstm_hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state  # [batch, seq_len, hidden]

        x = sequence_output.transpose(1, 2)  # [batch, hidden, seq_len]
        x = self.cnn(x)  # [batch, cnn_out_channels, seq_len]
        x = torch.relu(x)
        x = x.transpose(1, 2)  # [batch, seq_len, cnn_out_channels]

        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]

        # Use the same attention module that was trained under the "attn" name
        context, attn_weights = self.attn(lstm_out, mask=attention_mask)

        x = self.dropout(context)
        logits = self.fc(x)  # [batch, num_classes]
        probs = torch.softmax(logits, dim=-1)  # fraud, genuine

        return {
            "logits": logits,
            "probs": probs,
            "attention_weights": attn_weights,
        }

