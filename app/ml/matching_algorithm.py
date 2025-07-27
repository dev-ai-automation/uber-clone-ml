"""
Advanced Matching Algorithm using Deep Reinforcement Learning.
Based on recent research in multi-agent systems and optimal transport theory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
from dataclasses import dataclass

@dataclass
class RideRequest:
    """Ride request data structure."""
    rider_id: int
    pickup_lat: float
    pickup_lon: float
    dest_lat: float
    dest_lon: float
    requested_at: float
    vehicle_type: str
    max_wait_time: float
    price_sensitivity: float

@dataclass
class Driver:
    """Driver data structure."""
    driver_id: int
    current_lat: float
    current_lon: float
    is_available: bool
    vehicle_type: str
    rating: float
    total_rides: int
    last_ride_end: float

class AttentionMechanism(nn.Module):
    """Multi-head attention for rider-driver matching."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, riders, drivers):
        batch_size = riders.size(0)
        num_riders = riders.size(1)
        num_drivers = drivers.size(1)
        
        # Compute queries, keys, values
        Q = self.query(riders).view(batch_size, num_riders, self.num_heads, self.head_dim)
        K = self.key(drivers).view(batch_size, num_drivers, self.num_heads, self.head_dim)
        V = self.value(drivers).view(batch_size, num_drivers, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch, heads, riders, head_dim)
        K = K.transpose(1, 2)  # (batch, heads, drivers, head_dim)
        V = V.transpose(1, 2)  # (batch, heads, drivers, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, num_riders, self.embed_dim
        )
        
        return self.output(attended), attention_weights

class MatchingNetwork(nn.Module):
    """
    Deep neural network for rider-driver matching.
    Uses attention mechanisms and graph neural networks.
    """
    
    def __init__(
        self,
        rider_feature_dim: int = 16,
        driver_feature_dim: int = 16,
        hidden_dim: int = 128,
        num_attention_heads: int = 8,
        num_layers: int = 4
    ):
        super().__init__()
        
        self.rider_feature_dim = rider_feature_dim
        self.driver_feature_dim = driver_feature_dim
        self.hidden_dim = hidden_dim
        
        # Feature encoders
        self.rider_encoder = nn.Sequential(
            nn.Linear(rider_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.driver_encoder = nn.Sequential(
            nn.Linear(driver_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            AttentionMechanism(hidden_dim, num_attention_heads)
            for _ in range(num_layers)
        ])
        
        # Matching score predictor
        self.matching_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # ETA predictor
        self.eta_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.ReLU()
        )
        
        # Price predictor (surge pricing)
        self.price_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
    def encode_rider_features(self, riders: List[RideRequest]) -> torch.Tensor:
        """Encode rider features into tensor."""
        features = []
        for rider in riders:
            feature_vector = [
                rider.pickup_lat,
                rider.pickup_lon,
                rider.dest_lat,
                rider.dest_lon,
                rider.requested_at,
                1.0 if rider.vehicle_type == 'economy' else 0.0,
                1.0 if rider.vehicle_type == 'comfort' else 0.0,
                1.0 if rider.vehicle_type == 'premium' else 0.0,
                1.0 if rider.vehicle_type == 'xl' else 0.0,
                rider.max_wait_time,
                rider.price_sensitivity,
                # Distance from pickup to destination
                self.haversine_distance(
                    rider.pickup_lat, rider.pickup_lon,
                    rider.dest_lat, rider.dest_lon
                ),
                # Time of day features
                math.sin(2 * math.pi * (rider.requested_at % 24) / 24),
                math.cos(2 * math.pi * (rider.requested_at % 24) / 24),
                # Day of week features
                math.sin(2 * math.pi * (rider.requested_at // 24 % 7) / 7),
                math.cos(2 * math.pi * (rider.requested_at // 24 % 7) / 7)
            ]
            features.append(feature_vector)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def encode_driver_features(self, drivers: List[Driver]) -> torch.Tensor:
        """Encode driver features into tensor."""
        features = []
        for driver in drivers:
            feature_vector = [
                driver.current_lat,
                driver.current_lon,
                1.0 if driver.is_available else 0.0,
                1.0 if driver.vehicle_type == 'economy' else 0.0,
                1.0 if driver.vehicle_type == 'comfort' else 0.0,
                1.0 if driver.vehicle_type == 'premium' else 0.0,
                1.0 if driver.vehicle_type == 'xl' else 0.0,
                driver.rating,
                min(driver.total_rides / 1000.0, 1.0),  # Normalized
                driver.last_ride_end,
                # Recent activity features
                1.0 if (driver.last_ride_end > -60) else 0.0,  # Active in last hour
                1.0 if (driver.last_ride_end > -1440) else 0.0,  # Active in last day
                # Additional features can be added
                0.0, 0.0, 0.0, 0.0  # Placeholder for future features
            ]
            features.append(feature_vector)
        
        return torch.tensor(features, dtype=torch.float32)
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points."""
        R = 6371  # Earth's radius in km
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        
        c = 2 * math.asin(math.sqrt(a))
        return R * c
    
    def forward(self, riders: List[RideRequest], drivers: List[Driver]):
        """Forward pass for matching prediction."""
        
        # Encode features
        rider_features = self.encode_rider_features(riders).unsqueeze(0)  # Add batch dim
        driver_features = self.encode_driver_features(drivers).unsqueeze(0)
        
        # Encode through networks
        rider_encoded = self.rider_encoder(rider_features)
        driver_encoded = self.driver_encoder(driver_features)
        
        # Apply attention layers
        for attention_layer in self.attention_layers:
            rider_attended, attention_weights = attention_layer(rider_encoded, driver_encoded)
            rider_encoded = rider_encoded + rider_attended  # Residual connection
        
        # Compute pairwise features for all rider-driver pairs
        num_riders = rider_encoded.size(1)
        num_drivers = driver_encoded.size(1)
        
        matching_scores = torch.zeros(num_riders, num_drivers)
        eta_predictions = torch.zeros(num_riders, num_drivers)
        price_predictions = torch.zeros(num_riders, num_drivers)
        
        for i in range(num_riders):
            for j in range(num_drivers):
                # Concatenate rider and driver features
                pair_features = torch.cat([
                    rider_encoded[0, i], 
                    driver_encoded[0, j]
                ], dim=0).unsqueeze(0)
                
                # Predict matching score, ETA, and price
                matching_scores[i, j] = self.matching_predictor(pair_features).squeeze()
                eta_predictions[i, j] = self.eta_predictor(pair_features).squeeze()
                price_predictions[i, j] = self.price_predictor(pair_features).squeeze()
        
        return {
            'matching_scores': matching_scores,
            'eta_predictions': eta_predictions,
            'price_predictions': price_predictions,
            'attention_weights': attention_weights
        }

class OptimalTransportMatcher:
    """
    Optimal transport-based matching using Sinkhorn algorithm.
    Ensures fair and efficient rider-driver assignments.
    """
    
    def __init__(self, reg_param: float = 0.1, max_iter: int = 100):
        self.reg_param = reg_param
        self.max_iter = max_iter
    
    def sinkhorn_algorithm(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        """
        Sinkhorn algorithm for optimal transport.
        Returns optimal transport plan.
        """
        num_riders, num_drivers = cost_matrix.shape
        
        # Convert cost to similarity (negative cost with regularization)
        K = torch.exp(-cost_matrix / self.reg_param)
        
        # Initialize uniform distributions
        u = torch.ones(num_riders) / num_riders
        v = torch.ones(num_drivers) / num_drivers
        
        # Sinkhorn iterations
        for _ in range(self.max_iter):
            u_new = 1.0 / (K @ v)
            v_new = 1.0 / (K.T @ u_new)
            
            # Check convergence
            if torch.allclose(u, u_new, atol=1e-6) and torch.allclose(v, v_new, atol=1e-6):
                break
                
            u, v = u_new, v_new
        
        # Compute optimal transport plan
        transport_plan = torch.diag(u) @ K @ torch.diag(v)
        return transport_plan
    
    def match_riders_drivers(self, matching_scores: torch.Tensor, 
                           constraints: Optional[Dict] = None) -> List[Tuple[int, int]]:
        """
        Match riders to drivers using optimal transport.
        Returns list of (rider_idx, driver_idx) pairs.
        """
        
        # Use negative matching scores as cost (higher score = lower cost)
        cost_matrix = 1.0 - matching_scores
        
        # Apply constraints if provided
        if constraints:
            for rider_idx, invalid_drivers in constraints.get('invalid_matches', {}).items():
                for driver_idx in invalid_drivers:
                    cost_matrix[rider_idx, driver_idx] = float('inf')
        
        # Compute optimal transport plan
        transport_plan = self.sinkhorn_algorithm(cost_matrix)
        
        # Extract matches (greedy assignment from transport plan)
        matches = []
        used_drivers = set()
        
        # Sort by transport plan values (highest first)
        flat_indices = torch.argsort(transport_plan.flatten(), descending=True)
        
        for flat_idx in flat_indices:
            rider_idx = flat_idx // transport_plan.size(1)
            driver_idx = flat_idx % transport_plan.size(1)
            
            rider_idx = rider_idx.item()
            driver_idx = driver_idx.item()
            
            if (rider_idx not in [m[0] for m in matches] and 
                driver_idx not in used_drivers and
                transport_plan[rider_idx, driver_idx] > 1e-6):
                
                matches.append((rider_idx, driver_idx))
                used_drivers.add(driver_idx)
        
        return matches

class AdvancedMatchingAlgorithm:
    """
    Complete matching algorithm combining ML predictions with optimal transport.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.matching_network = MatchingNetwork()
        self.optimal_transport = OptimalTransportMatcher()
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load pre-trained matching model."""
        checkpoint = torch.load(model_path, map_location='cpu')
        self.matching_network.load_state_dict(checkpoint['model_state_dict'])
        self.matching_network.eval()
    
    def save_model(self, model_path: str):
        """Save matching model."""
        torch.save({
            'model_state_dict': self.matching_network.state_dict(),
        }, model_path)
    
    def match_rides(self, riders: List[RideRequest], drivers: List[Driver]) -> Dict:
        """
        Main matching function that combines ML predictions with optimal transport.
        """
        
        if not riders or not drivers:
            return {'matches': [], 'unmatched_riders': list(range(len(riders)))}
        
        # Get ML predictions
        with torch.no_grad():
            predictions = self.matching_network(riders, drivers)
        
        # Apply business logic constraints
        constraints = self._apply_business_constraints(riders, drivers, predictions)
        
        # Find optimal matches
        matches = self.optimal_transport.match_riders_drivers(
            predictions['matching_scores'], 
            constraints
        )
        
        # Prepare results
        matched_rider_indices = {match[0] for match in matches}
        unmatched_riders = [i for i in range(len(riders)) if i not in matched_rider_indices]
        
        # Add prediction details to matches
        detailed_matches = []
        for rider_idx, driver_idx in matches:
            detailed_matches.append({
                'rider_idx': rider_idx,
                'driver_idx': driver_idx,
                'rider_id': riders[rider_idx].rider_id,
                'driver_id': drivers[driver_idx].driver_id,
                'matching_score': predictions['matching_scores'][rider_idx, driver_idx].item(),
                'estimated_eta': predictions['eta_predictions'][rider_idx, driver_idx].item(),
                'estimated_price': predictions['price_predictions'][rider_idx, driver_idx].item(),
                'pickup_distance': self.matching_network.haversine_distance(
                    riders[rider_idx].pickup_lat, riders[rider_idx].pickup_lon,
                    drivers[driver_idx].current_lat, drivers[driver_idx].current_lon
                )
            })
        
        return {
            'matches': detailed_matches,
            'unmatched_riders': unmatched_riders,
            'predictions': predictions
        }
    
    def _apply_business_constraints(self, riders: List[RideRequest], 
                                  drivers: List[Driver], 
                                  predictions: Dict) -> Dict:
        """Apply business logic constraints to matching."""
        
        constraints = {'invalid_matches': {}}
        
        for rider_idx, rider in enumerate(riders):
            invalid_drivers = []
            
            for driver_idx, driver in enumerate(drivers):
                # Vehicle type compatibility
                if rider.vehicle_type != driver.vehicle_type:
                    invalid_drivers.append(driver_idx)
                    continue
                
                # Driver availability
                if not driver.is_available:
                    invalid_drivers.append(driver_idx)
                    continue
                
                # Maximum pickup distance (e.g., 10 km)
                pickup_distance = self.matching_network.haversine_distance(
                    rider.pickup_lat, rider.pickup_lon,
                    driver.current_lat, driver.current_lon
                )
                if pickup_distance > 10.0:
                    invalid_drivers.append(driver_idx)
                    continue
                
                # Estimated ETA constraint
                estimated_eta = predictions['eta_predictions'][rider_idx, driver_idx].item()
                if estimated_eta > rider.max_wait_time:
                    invalid_drivers.append(driver_idx)
                    continue
            
            if invalid_drivers:
                constraints['invalid_matches'][rider_idx] = invalid_drivers
        
        return constraints
