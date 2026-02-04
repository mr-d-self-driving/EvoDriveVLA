class CrossAttentionCompressor(nn.Module):
    def __init__(self, input_dim: int, target_length: int, num_heads: int = 8):
        super().__init__()
        self.learned_queries = nn.Parameter(torch.randn(target_length, input_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, teacher_features: torch.Tensor) -> torch.Tensor:
        # teacher_features 形状: [B, L_T, D], 例如 [B, 924, 2048]
        B = teacher_features.shape[0]
        queries = self.learned_queries.unsqueeze(0).expand(B, -1, -1) # 形状: [B, 231, D]
        
        compressed_feats, _ = self.cross_attn(
            query=queries,
            key=teacher_features,
            value=teacher_features,
            need_weights=False
        )
        compressed_feats = self.norm(compressed_feats + queries) 
        
        return compressed_feats