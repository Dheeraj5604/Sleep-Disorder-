import torch
import torch.nn as nn

class SleepTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4):
        super(SleepTransformer, self).__init__()
        
        # Project EACH feature individually into d_model dimensions
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(input_dim)
        ])
        
        # A learnable CLS token to aggregate information 
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # The Transformer block
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Classifier that only looks at the CLS token output
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Embed each feature: shape becomes (Batch, Sequence_Length=10, d_model)
        embedded_features = []
        for i in range(x.shape[1]):
            feature = x[:, i].unsqueeze(1) 
            embedded_features.append(self.feature_embeddings[i](feature).unsqueeze(1))
            
        x_seq = torch.cat(embedded_features, dim=1) 
        
        # Add the CLS token to the start of the sequence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat((cls_tokens, x_seq), dim=1)
        
        # Pass through transformer
        transformed_seq = self.transformer(x_seq)
        
        # Extract the CLS token's output to make the final prediction
        cls_output = transformed_seq[:, 0, :]
        return self.classifier(cls_output)