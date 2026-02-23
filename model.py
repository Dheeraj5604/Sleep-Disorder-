import torch
import torch.nn as nn

class SleepTransformer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SleepTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_dim, 256)
        
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1) 
        x = self.transformer(x).squeeze(1)
        return self.classifier(x)