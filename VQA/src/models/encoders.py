import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.models import resnet50, ResNet50_Weights

class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=512, num_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        with torch.no_grad():
            init.xavier_uniform_(self.embedding.weight) # Initialize embeddings using xavier_uniform
            self.embedding.weight[0] = torch.zeros(embed_dim)
            self.embedding.weight[0].requires_grad = False
            
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 2 # LSTM encodes question bidirectionally

    def forward(self, question):
        embedded = self.embedding(question)
        embedded = self.dropout(embedded)
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        h_q = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # (batch_size, hidden_dim*2)
        return h_q  # Vector representation of question

class ImageEncoder(nn.Module):
    def __init__(self, feature_dim=2048, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else ResNet50_Weights.IMAGENET1K_V1
        self.resnet = resnet50(weights=weights)

        for param in self.resnet.parameters(): # Freeze parameters to extract features quickly
            param.requires_grad = False
        for param in self.resnet.layer4.parameters(): # Unfreeze last layer to allow fine-tuning
            param.requires_grad = True
            
        self.resnet.fc = nn.Identity()

    def forward(self, image):
        f_img = self.resnet(image)  # [batch, 2048]
        return f_img
