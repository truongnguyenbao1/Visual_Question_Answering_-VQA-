import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

from .encoders import QuestionEncoder, ImageEncoder

class VQAModel(nn.Module):
    def __init__(self, question_vocab_size, answer_vocab_size, 
                 image_feature_dim=2048, question_hidden_dim=512, 
                 fusion_hidden_dim=1024, dropout=0.5,
                 use_attention=False, num_heads=8):
        """
        Unified VQAModel that can toggle between Simple MLP fusion and Attention-based fusion.
        """
        super().__init__()
        self.use_attention = use_attention
        self.question_encoder = QuestionEncoder(question_vocab_size, hidden_dim=question_hidden_dim)
        self.image_encoder = ImageEncoder(image_feature_dim, pretrained=True)
        
        if self.use_attention:
            # Projection to match fusion_hidden_dim for attention
            self.img_proj = nn.Linear(image_feature_dim, fusion_hidden_dim)  # 2048 → 1024
            self.q_proj = nn.Linear(question_hidden_dim * 2, fusion_hidden_dim)  # 1024 → 1024

            # Attention fusion: Query from question, key/value from image
            self.attention = MultiheadAttention(embed_dim=fusion_hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

            # Fusion dim after concat (3 * fusion_hidden_dim = 3072)
            self.fusion_dim = fusion_hidden_dim * 3
        else:
            self.fusion_dim = image_feature_dim + question_hidden_dim * 2 # 2048 + 1024 = 3072

        self.fusion = nn.Sequential(
            nn.Linear(self.fusion_dim, fusion_hidden_dim), # 3072 --> 1024
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2), # 1024 --> 512
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, answer_vocab_size) # 512 --> answer_vocab_size
        )

    def forward(self, image, question):
        f_img = self.image_encoder(image)  # [batch, 2048]
        h_q = self.question_encoder(question)  # [batch, 1024]

        if self.use_attention:
            # Project to fusion_hidden_dim
            f_img_proj = self.img_proj(f_img)  # [batch, 1024]
            h_q_proj = self.q_proj(h_q)  # [batch, 1024]

            # Reshape for attention (seq_len=1 for vectors)
            f_img_seq = f_img_proj.unsqueeze(1)  # [batch, 1, 1024] (key/value)
            h_q_seq = h_q_proj.unsqueeze(1)  # [batch, 1, 1024] (query)

            # Cross-attention
            attn_output, _ = self.attention(h_q_seq, f_img_seq, f_img_seq)  # [batch, 1, 1024]
            attn_output = attn_output.squeeze(1)  # [batch, 1024]

            # Concat (now 3072 dim)
            fused = torch.cat((h_q_proj, f_img_proj, attn_output), dim=1)  # [batch, 3072]
        else:
            # Simple Concat
            fused = torch.cat((f_img, h_q), dim=1)
            
        output = self.fusion(fused)  # [batch, answer_vocab_size]
        return output
