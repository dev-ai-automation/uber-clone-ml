"""
Supply Prediction Model using Graph Neural Networks.
Based on recent research in spatial-temporal graph learning for driver availability prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

class GraphConvolution(nn.Module):
    """Graph convolution layer for spatial relationships."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class TemporalConvolution(nn.Module):
    """Temporal convolution for time series patterns."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        x = x.transpose(1, 2)  # (batch_size, features, seq_len)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x.transpose(1, 2)  # (batch_size, seq_len, features)

class SpatialTemporalBlock(nn.Module):
    """Spatial-temporal block combining graph convolution and temporal convolution."""
    
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        
        # Spatial component (Graph Convolution)
        self.spatial_conv = GraphConvolution(in_features, hidden_features)
        
        # Temporal component
        self.temporal_conv = TemporalConvolution(hidden_features, out_features)
        
        # Residual connection
        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, adj):
        # x shape: (batch_size, seq_len, num_nodes, features)
        batch_size, seq_len, num_nodes, features = x.shape
        
        # Reshape for graph convolution
        x_reshaped = x.view(-1, features)  # (batch_size * seq_len * num_nodes, features)
        
        # Apply spatial convolution
        spatial_out = self.spatial_conv(x_reshaped, adj)
        spatial_out = spatial_out.view(batch_size, seq_len, num_nodes, -1)
        
        # Apply temporal convolution (average over nodes for simplicity)
        temporal_input = spatial_out.mean(dim=2)  # (batch_size, seq_len, features)
        temporal_out = self.temporal_conv(temporal_input)
        
        # Expand back to node dimension
        temporal_out = temporal_out.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        
        # Residual connection
        residual = self.residual(x)
        
        # Combine and normalize
        output = self.norm(temporal_out + residual)
        output = self.dropout(output)
        
        return output

class SupplyPredictor(nn.Module):
    """
    Advanced supply prediction model using Graph Neural Networks.
    Predicts driver availability in different geographic regions.
    """
    
    def __init__(
        self,
        num_nodes: int = 100,  # Number of geographic regions
        input_features: int = 16,
        hidden_features: int = 64,
        num_layers: int = 4,
        sequence_length: int = 24,  # 24 hours
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.sequence_length = sequence_length
        
        # Feature encoders
        self.time_encoder = nn.Embedding(24, 8)  # Hour of day
        self.day_encoder = nn.Embedding(7, 4)    # Day of week
        
        # Driver state encoder
        self.driver_state_encoder = nn.Sequential(
            nn.Linear(4, 16),  # num_active, num_busy, avg_rating, avg_distance
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # Regional features encoder
        self.region_encoder = nn.Sequential(
            nn.Linear(6, 16),  # population_density, poi_count, avg_income, traffic_level, event_indicator, weather_score
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # Input projection
        self.input_projection = nn.Linear(input_features, hidden_features)
        
        # Spatial-temporal blocks
        self.st_blocks = nn.ModuleList([
            SpatialTemporalBlock(
                hidden_features if i > 0 else hidden_features,
                hidden_features,
                hidden_features
            ) for i in range(num_layers)
        ])
        
        # Attention mechanism for multi-scale temporal patterns
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_features,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_features, hidden_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features // 2, hidden_features // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features // 4, 1),
            nn.ReLU()  # Ensure positive supply
        )
        
        # Supply category classifier (low, medium, high supply)
        self.supply_classifier = nn.Sequential(
            nn.Linear(hidden_features, hidden_features // 2),
            nn.ReLU(),
            nn.Linear(hidden_features // 2, 3),  # 3 supply categories
            nn.Softmax(dim=-1)
        )
        
    def create_adjacency_matrix(self, locations: torch.Tensor, threshold: float = 5.0):
        """
        Create adjacency matrix based on geographic proximity.
        locations: (num_nodes, 2) tensor of [lat, lon]
        threshold: distance threshold in km
        """
        num_nodes = locations.shape[0]
        adj = torch.zeros(num_nodes, num_nodes)
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # Haversine distance approximation
                    lat1, lon1 = locations[i]
                    lat2, lon2 = locations[j]
                    
                    dlat = torch.abs(lat1 - lat2)
                    dlon = torch.abs(lon1 - lon2)
                    
                    # Simplified distance calculation
                    distance = torch.sqrt(dlat**2 + dlon**2) * 111  # Rough km conversion
                    
                    if distance < threshold:
                        adj[i, j] = 1.0 / (1.0 + distance)  # Inverse distance weighting
        
        # Normalize adjacency matrix
        row_sums = adj.sum(dim=1, keepdim=True)
        adj = adj / (row_sums + 1e-8)
        
        return adj
    
    def create_features(self, batch_data: Dict[str, torch.Tensor]):
        """Create feature embeddings from input data."""
        batch_size, seq_len = batch_data['hour'].shape
        
        # Time embeddings
        hour_emb = self.time_encoder(batch_data['hour'])  # (batch_size, seq_len, 8)
        day_emb = self.day_encoder(batch_data['day_of_week'])  # (batch_size, seq_len, 4)
        
        # Driver state features
        driver_features = torch.stack([
            batch_data['num_active_drivers'],
            batch_data['num_busy_drivers'],
            batch_data['avg_driver_rating'],
            batch_data['avg_driver_distance']
        ], dim=-1)
        driver_emb = self.driver_state_encoder(driver_features)  # (batch_size, seq_len, 8)
        
        # Regional features
        regional_features = torch.stack([
            batch_data['population_density'],
            batch_data['poi_count'],
            batch_data['avg_income'],
            batch_data['traffic_level'],
            batch_data['event_indicator'],
            batch_data['weather_score']
        ], dim=-1)
        region_emb = self.region_encoder(regional_features)  # (batch_size, seq_len, 8)
        
        # Concatenate all features
        features = torch.cat([hour_emb, day_emb, driver_emb, region_emb], dim=-1)
        
        return features
    
    def forward(self, batch_data: Dict[str, torch.Tensor], locations: torch.Tensor):
        """Forward pass of the supply prediction model."""
        
        # Create adjacency matrix
        adj = self.create_adjacency_matrix(locations)
        
        # Create feature embeddings
        features = self.create_features(batch_data)
        batch_size, seq_len, feature_dim = features.shape
        
        # Expand features to include node dimension
        features = features.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)
        
        # Project to hidden dimension
        x = self.input_projection(features)
        
        # Apply spatial-temporal blocks
        for st_block in self.st_blocks:
            x = st_block(x, adj)
        
        # Apply temporal attention
        # Reshape for attention: (batch_size * num_nodes, seq_len, hidden_features)
        x_att = x.view(batch_size * self.num_nodes, seq_len, -1)
        attended, _ = self.temporal_attention(x_att, x_att, x_att)
        attended = attended.view(batch_size, seq_len, self.num_nodes, -1)
        
        # Use the last timestep for prediction
        last_hidden = attended[:, -1, :, :]  # (batch_size, num_nodes, hidden_features)
        
        # Predict supply for each node
        supply_pred = self.output_layers(last_hidden)  # (batch_size, num_nodes, 1)
        
        # Supply category classification
        supply_categories = self.supply_classifier(last_hidden)  # (batch_size, num_nodes, 3)
        
        return {
            'supply_prediction': supply_pred.squeeze(-1),  # (batch_size, num_nodes)
            'supply_categories': supply_categories,
            'node_embeddings': last_hidden
        }
    
    def predict_region_supply(self, batch_data: Dict[str, torch.Tensor], 
                            target_location: Tuple[float, float], 
                            locations: torch.Tensor):
        """Predict supply for a specific location."""
        
        # Find nearest node to target location
        target_tensor = torch.tensor(target_location).unsqueeze(0)
        distances = torch.cdist(target_tensor, locations)
        nearest_node = distances.argmin().item()
        
        # Get predictions for all nodes
        predictions = self.forward(batch_data, locations)
        
        # Return prediction for nearest node
        return {
            'supply_prediction': predictions['supply_prediction'][:, nearest_node],
            'supply_category': predictions['supply_categories'][:, nearest_node],
            'nearest_node': nearest_node,
            'node_embedding': predictions['node_embeddings'][:, nearest_node]
        }
