import torch
import torch.nn as nn
from transformers import AutoModel

class TextModel(nn.Module):
    def __init__(self, num_classes=4):
        super(TextModel, self).__init__()

        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # DistilBERT has no pooler → use CLS token
        cls_output = outputs.last_hidden_state[:, 0]

        x = self.dropout(cls_output)
        logits = self.classifier(x)

        return logits
