import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

    def forward(self, seq_len, device):
        """
        :param seq_len: 입력 데이터의 시퀀스 길이
        :param device: 텐서가 저장될 장치 (CPU/GPU)
        :return: Positional Encoding 텐서 (1, seq_len, embed_dim)
        """
        position = torch.arange(0, seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, device=device) *
                             -(torch.log(torch.tensor(10000.0)) / self.embed_dim))
        pe = torch.zeros(seq_len, self.embed_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, seq_len, embed_dim)

class TSTModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=6, num_layers=2, dim_feedforward=512):
        super(TSTModel, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim=input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)

        self.class_mapping = {
            0: "normal",
            1: "violence",
            2: "kidnap"
        }
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Positional Encoding 계산
        pos_enc = self.positional_encoding(seq_len, device)

        x = x + pos_enc  # Positional Encoding 추가

        # src_key_padding_mask가 None일 경우 처리
        if mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=~mask)  # True를 패딩 위치로 처리
        else:
            x = self.transformer_encoder(x)  # 마스킹 없이 처리

        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)


    def inference(self, x):
        self.eval()  # 모델을 평가 모드로 설정
        with torch.no_grad():
            logits = self.forward(x)  # 모델 출력
            probabilities = torch.softmax(logits, dim=-1)  # softmax로 확률 계산
            predicted_class = torch.argmax(probabilities, dim=-1).item()  # 최대 값의 인덱스 추출
        return self.class_mapping[predicted_class]
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)  # 가중치만 저장

    def load_model(self, path, device='cpu'):
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))

