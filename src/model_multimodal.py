import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision import models

class MultimodalModel(nn.Module):
    def __init__(self, num_classes=4):
        super(MultimodalModel, self).__init__()

        # ---- TEXT ENCODER ----
        self.text_model = AutoModel.from_pretrained("distilbert-base-uncased")

        # ---- IMAGE ENCODER ----
        self.image_model = models.resnet18(pretrained=True)

        # Freeze image backbone (same fix as before)
        for param in self.image_model.parameters():
            param.requires_grad = False

        # Remove final FC layer
        self.image_model.fc = nn.Identity()

        # ---- FUSION ----
        text_dim = self.text_model.config.hidden_size   # 768
        image_dim = 512

        self.fc = nn.Sequential(
            nn.Linear(text_dim + image_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, images):

        # ---- TEXT ----
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feat = text_outputs.last_hidden_state[:, 0]  # CLS token

        # ---- IMAGE ----
        image_feat = self.image_model(images)

        # ---- FUSION ----
        combined = torch.cat((text_feat, image_feat), dim=1)

        logits = self.fc(combined)

        return logits
