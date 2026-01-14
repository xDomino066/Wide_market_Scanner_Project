import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import math


# --- 1. MECHANIZM TRANSFORMERA ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Matematyka falowa - pozwala modelowi rozumieć upływ czasu
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Dodajemy informację o pozycji do danych wejściowych
        x = x + self.pe[:, :x.size(1), :]
        return x


class TitanFeatureExtractor(BaseFeaturesExtractor):
    """
    Adapter: Zamienia wykres giełdowy (60 dni) na wektor decyzji dla PPO.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(TitanFeatureExtractor, self).__init__(observation_space, features_dim)

        # Wymiary wejścia [60, Liczba_Cech]
        self.seq_len = observation_space.shape[0]
        self.input_dim = observation_space.shape[1]

        # Parametry sieci (Dostosowane pod RTX 5070 Ti)
        d_model = 128
        nhead = 4
        num_layers = 2

        # A. Wejście (Embedding)
        self.embedding = nn.Linear(self.input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.seq_len)

        # B. Transformer Encoder (Serce modelu)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.0,  # W Reinforcement Learningu dropout często destabilizuje
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # C. Wyjście (Spłaszczenie do wektora)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(d_model * self.seq_len, features_dim)
        self.relu = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Przepływ danych: Dane -> Embedding -> Czas -> Transformer -> Decyzja
        x = self.embedding(observations)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.flatten(x)
        return self.relu(self.linear(x))


# --- KONFIGURACJA DLA PPO ---
# Tę zmienną zaimportujemy w pliku train.py
policy_kwargs = dict(
    features_extractor_class=TitanFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=256),
    # Sieć decyzyjna (Actor) i Sieć wartości (Critic) po Transformerze
    net_arch=dict(pi=[128, 64], vf=[128, 64])
)