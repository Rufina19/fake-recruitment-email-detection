from typing import List, Dict

import torch
from transformers import BertTokenizer


class BertEmbeddingGenerator:
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 128):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def encode_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encoding

