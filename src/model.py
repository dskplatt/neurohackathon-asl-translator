"""
CNN + Bidirectional LSTM classifier for ASL fingerspelling.

Architecture designed for two-phase use:
  1. Standard training: all layers trained end-to-end
  2. Calibration fine-tuning: feature_extractor + temporal_encoder + projection
     are frozen, only classifier_head is fine-tuned on ~26 user samples

Input:  (batch, n_channels, n_timesteps) — default (batch, 11, 40)
Output: (batch, n_classes) logits — default (batch, 26)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASLClassifier(nn.Module):

    def __init__(self, n_channels=11, n_timesteps=40, n_classes=26):
        super().__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.n_classes = n_classes

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.temporal_encoder = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        self.projection = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.classifier_head = nn.Linear(64, n_classes)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 64-dim embedding. Used during calibration to precompute
        feature vectors without repeating the full forward pass."""
        x = self.feature_extractor(x)          # (B, 64, T)
        x = x.permute(0, 2, 1)                 # (B, T, 64)
        x, _ = self.temporal_encoder(x)         # (B, T, 128)
        x = x[:, -1, :]                         # (B, 128)
        x = self.projection(x)                  # (B, 64)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits — no softmax."""
        return self.classifier_head(self.get_features(x))

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities summing to 1.0."""
        return F.softmax(self.forward(x), dim=1)

    def freeze_feature_extractor(self) -> None:
        """Freeze everything except classifier_head for calibration.
        Also locks BatchNorm to eval mode so tiny calibration batches
        don't corrupt running statistics."""
        for module in [self.feature_extractor, self.temporal_encoder, self.projection]:
            for param in module.parameters():
                param.requires_grad = False

        for m in self.feature_extractor.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()

        for param in self.classifier_head.parameters():
            param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Restore all parameters to trainable."""
        for param in self.parameters():
            param.requires_grad = True
        self.train()

    def get_calibration_param_count(self) -> int:
        return sum(p.numel() for p in self.classifier_head.parameters())

    def get_total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = ASLClassifier()
    x = torch.randn(4, 11, 40)

    logits = model(x)
    probs = model.get_probabilities(x)
    features = model.get_features(x)

    print(f"Input shape:    {x.shape}")
    print(f"Logits shape:   {logits.shape}")
    print(f"Probs shape:    {probs.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Probs sum:      {probs.sum(dim=1)}")

    total = model.get_total_param_count()
    cal = model.get_calibration_param_count()
    print(f"\nTotal parameters:       {total:,}")
    print(f"Calibration parameters: {cal:,}")

    model.freeze_feature_extractor()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable after freeze: {trainable:,}")
    assert trainable == cal, f"Expected {cal}, got {trainable}"

    model.unfreeze_all()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable after unfreeze: {trainable:,}")
    assert trainable == total, f"Expected {total}, got {trainable}"

    assert logits.shape == (4, 26)
    assert probs.shape == (4, 26)
    assert features.shape == (4, 64)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-4)

    print("\nAll assertions passed.")
