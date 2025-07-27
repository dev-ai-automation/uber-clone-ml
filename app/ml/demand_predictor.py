"""
Demand Prediction Model using Transformer architecture.
Based on recent research in spatio-temporal forecasting and attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SpatialAttention(nn.Module):
    """Spatial attention mechanism for geographic features."""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        self.query = nn.Linear(feature_dim, hidden_dim)
        self.key = nn.Linear(feature_dim, hidden_dim)
        self.value = nn.Linear(feature_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, feature_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Compute attention weights
        attention_weights = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_dim), dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        output = self.output(attended)
        
        return output + x  # Residual connection

class DemandPredictor(nn.Module):
    """
    Advanced demand prediction model using Transformer architecture.
    Incorporates spatio-temporal features, weather, events, and historical patterns.
    """
    
    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_sequence_length: int = 168  # 1 week of hourly data
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length
        
        # Feature embedding layers
        self.time_embedding = nn.Embedding(24, 16)  # Hour of day
        self.day_embedding = nn.Embedding(7, 8)    # Day of week
        self.month_embedding = nn.Embedding(12, 8)  # Month
        
        # Geographic feature processing
        self.geo_encoder = nn.Sequential(
            nn.Linear(2, 32),  # lat, lon
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Weather feature processing
        self.weather_encoder = nn.Sequential(
            nn.Linear(5, 32),  # temp, humidity, precipitation, wind_speed, visibility
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Historical demand encoder
        self.demand_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_sequence_length)
        
        # Spatial attention
        self.spatial_attention = SpatialAttention(hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.ReLU()  # Ensure positive demand
        )
        
        # Uncertainty estimation (for confidence intervals)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
    def create_features(self, batch_data):
        """Create feature embeddings from input data."""
        batch_size, seq_len = batch_data['hour'].shape
        
        # Time embeddings
        hour_emb = self.time_embedding(batch_data['hour'])
        day_emb = self.day_embedding(batch_data['day_of_week'])
        month_emb = self.month_embedding(batch_data['month'])
        
        # Geographic features
        geo_features = torch.stack([batch_data['latitude'], batch_data['longitude']], dim=-1)
        geo_emb = self.geo_encoder(geo_features)
        
        # Weather features
        weather_features = torch.stack([
            batch_data['temperature'],
            batch_data['humidity'],
            batch_data['precipitation'],
            batch_data['wind_speed'],
            batch_data['visibility']
        ], dim=-1)
        weather_emb = self.weather_encoder(weather_features)
        
        # Historical demand
        demand_emb = self.demand_encoder(batch_data['historical_demand'].unsqueeze(-1))
        
        # Concatenate all features
        features = torch.cat([
            hour_emb, day_emb, month_emb, geo_emb, weather_emb, demand_emb
        ], dim=-1)
        
        return features
    
    def forward(self, batch_data, return_uncertainty=False):
        """Forward pass of the demand prediction model."""
        
        # Create feature embeddings
        features = self.create_features(batch_data)
        
        # Project to hidden dimension
        x = self.input_projection(features)
        
        # Add positional encoding
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Apply spatial attention
        x = self.spatial_attention(x)
        
        # Transformer encoding
        encoded = self.transformer(x)
        
        # Use the last timestep for prediction
        last_hidden = encoded[:, -1, :]
        
        # Predict demand
        demand_pred = self.output_layers(last_hidden)
        
        if return_uncertainty:
            uncertainty = self.uncertainty_head(last_hidden)
            return demand_pred, uncertainty
        
        return demand_pred
    
    def predict_with_confidence(self, batch_data, num_samples=100):
        """Predict demand with confidence intervals using Monte Carlo dropout."""
        self.train()  # Enable dropout for uncertainty estimation
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(batch_data)
                predictions.append(pred)
        
        self.eval()  # Return to eval mode
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # 95% confidence interval
        lower_bound = mean_pred - 1.96 * std_pred
        upper_bound = mean_pred + 1.96 * std_pred
        
        return {
            'prediction': mean_pred,
            'lower_bound': torch.clamp(lower_bound, min=0),
            'upper_bound': upper_bound,
            'uncertainty': std_pred
        }
