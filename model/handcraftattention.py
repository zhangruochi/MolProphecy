import torch.nn as nn

class HandcraftedAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.3):
        super().__init__()
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_layers)
        ])
        
    def forward(self, fusion_embedding, handcrafted_features):
        fusion_embedding = fusion_embedding.unsqueeze(1)  # [batch_size, 1, embed_dim]
        handcrafted_features = handcrafted_features.unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        for attn_layer, layer_norm in zip(self.attn_layers, self.layer_norms):
            residual = fusion_embedding
            
            attn_output, _ = attn_layer(
                query=fusion_embedding,
                key=handcrafted_features,
                value=handcrafted_features
            )
            
            fusion_embedding = layer_norm(attn_output + residual)
        
        return fusion_embedding.squeeze(1) 