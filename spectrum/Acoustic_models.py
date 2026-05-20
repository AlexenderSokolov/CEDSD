import torch
import torch.nn as nn

# ===================== acoustic feature models =====================

class CNN_MFCC(nn.Module):
    """CNN encoder for MFCC plus delta features."""
    def __init__(self, in_dim=120, hidden_dim=128, max_len=512): # Keep max_len aligned with Dataset_all.
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        # Two MaxPool1d(2) layers reduce length to max_len // 4.
        self.fc = nn.Linear(hidden_dim*2 * (max_len//4), 256)
    
    def forward(self, x):
        # x shape: [B, max_len, 120]
        x = x.transpose(1, 2) # Convert to [B, 120, max_len] for Conv1d.
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(1)
        x = torch.relu(self.fc(x))
        return x

class Transformer_F0(nn.Module):
    """Transformer encoder for F0 features."""
    def __init__(self, in_dim=1, embed_dim=64, num_heads=4, max_len=512): # Keep max_len aligned with Dataset_all.
        super().__init__()
        self.embedding = nn.Linear(in_dim, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, embed_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True),
            num_layers=2
        )
        self.fc = nn.Linear(embed_dim * max_len, 256)
    
    def forward(self, x):
        # x shape: [B, max_len, 1]
        x = self.embedding(x) + self.pos_encoding # [B, max_len, embed_dim]
        x = self.transformer(x)
        x = x.flatten(1)
        x = torch.relu(self.fc(x))
        return x

class Fusion_MLP(nn.Module):
    """Top-level fusion module for CNN and Transformer acoustic features."""
    def __init__(self, cnn_module, transformer_module):
        super().__init__()
        self.cnn = cnn_module
        self.transformer = transformer_module
        self.mlp = nn.Sequential(
            nn.Linear(256+256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
    
    def forward(self, mfcc, f0):
        cnn_feat = self.cnn(mfcc)            # [B, 256]
        transformer_feat = self.transformer(f0) # [B, 256]
        fusion_feat = torch.cat([cnn_feat, transformer_feat], dim=1) # [B, 512]
        return fusion_feat # Return features; classification happens downstream.

# Fusion_MLP returns a feature matrix for downstream emotion or spoofing heads, without local softmax classification.
