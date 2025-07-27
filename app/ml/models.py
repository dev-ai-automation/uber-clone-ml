"""
Model Manager for loading and managing ML models.
"""

import torch
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.core.config import settings
from app.ml.demand_predictor import DemandPredictor
from app.ml.supply_predictor import SupplyPredictor
from app.ml.matching_algorithm import AdvancedMatchingAlgorithm, RideRequest, Driver

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages all ML models for the ride-sharing platform."""
    
    def __init__(self):
        self.demand_model: Optional[DemandPredictor] = None
        self.supply_model: Optional[SupplyPredictor] = None
        self.matching_algorithm: Optional[AdvancedMatchingAlgorithm] = None
        self.device = torch.device(settings.DEVICE)
        self.models_loaded = False
        
        # Create models directory if it doesn't exist
        Path(settings.ML_MODEL_PATH).mkdir(exist_ok=True)
        
    async def load_models(self):
        """Load all ML models."""
        try:
            logger.info("Loading ML models...")
            
            # Initialize models
            self.demand_model = DemandPredictor().to(self.device)
            self.supply_model = SupplyPredictor().to(self.device)
            self.matching_algorithm = AdvancedMatchingAlgorithm()
            
            # Try to load pre-trained weights
            await self._load_pretrained_weights()
            
            # Set models to evaluation mode
            self.demand_model.eval()
            self.supply_model.eval()
            self.matching_algorithm.matching_network.eval()
            
            self.models_loaded = True
            logger.info("ML models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            # Initialize with random weights if loading fails
            await self._initialize_random_models()
    
    async def _load_pretrained_weights(self):
        """Load pre-trained model weights if available."""
        
        # Load demand model
        demand_path = Path(settings.ML_MODEL_PATH) / settings.DEMAND_MODEL_NAME
        if demand_path.exists():
            checkpoint = torch.load(demand_path, map_location=self.device)
            self.demand_model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded pre-trained demand model")
        else:
            logger.info("No pre-trained demand model found, using random weights")
        
        # Load supply model
        supply_path = Path(settings.ML_MODEL_PATH) / settings.SUPPLY_MODEL_NAME
        if supply_path.exists():
            checkpoint = torch.load(supply_path, map_location=self.device)
            self.supply_model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded pre-trained supply model")
        else:
            logger.info("No pre-trained supply model found, using random weights")
        
        # Load matching model
        matching_path = Path(settings.ML_MODEL_PATH) / settings.MATCHING_MODEL_NAME
        if matching_path.exists():
            self.matching_algorithm.load_model(str(matching_path))
            logger.info("Loaded pre-trained matching model")
        else:
            logger.info("No pre-trained matching model found, using random weights")
    
    async def _initialize_random_models(self):
        """Initialize models with random weights as fallback."""
        self.demand_model = DemandPredictor().to(self.device)
        self.supply_model = SupplyPredictor().to(self.device)
        self.matching_algorithm = AdvancedMatchingAlgorithm()
        
        self.demand_model.eval()
        self.supply_model.eval()
        self.matching_algorithm.matching_network.eval()
        
        self.models_loaded = True
        logger.info("Initialized models with random weights")
    
    async def predict_demand(self, location: Tuple[float, float], 
                           time_features: Dict) -> Dict:
        """Predict demand at a specific location and time."""
        
        if not self.models_loaded or self.demand_model is None:
            raise RuntimeError("Models not loaded")
        
        try:
            # Prepare input data
            batch_data = self._prepare_demand_input(location, time_features)
            
            with torch.no_grad():
                # Get prediction with confidence intervals
                result = self.demand_model.predict_with_confidence(batch_data)
                
                return {
                    'demand_prediction': float(result['prediction'].item()),
                    'confidence_lower': float(result['lower_bound'].item()),
                    'confidence_upper': float(result['upper_bound'].item()),
                    'uncertainty': float(result['uncertainty'].item()),
                    'location': location,
                    'timestamp': time_features.get('timestamp', 0)
                }
                
        except Exception as e:
            logger.error(f"Error predicting demand: {e}")
            # Return default values on error
            return {
                'demand_prediction': 1.0,
                'confidence_lower': 0.5,
                'confidence_upper': 2.0,
                'uncertainty': 0.5,
                'location': location,
                'timestamp': time_features.get('timestamp', 0)
            }
    
    async def predict_supply(self, location: Tuple[float, float], 
                           time_features: Dict,
                           region_locations: Optional[torch.Tensor] = None) -> Dict:
        """Predict supply at a specific location and time."""
        
        if not self.models_loaded or self.supply_model is None:
            raise RuntimeError("Models not loaded")
        
        try:
            # Prepare input data
            batch_data = self._prepare_supply_input(location, time_features)
            
            # Use default region locations if not provided
            if region_locations is None:
                region_locations = self._generate_default_regions(location)
            
            with torch.no_grad():
                result = self.supply_model.predict_region_supply(
                    batch_data, location, region_locations
                )
                
                return {
                    'supply_prediction': float(result['supply_prediction'].item()),
                    'supply_category': result['supply_category'].argmax().item(),
                    'nearest_node': result['nearest_node'],
                    'location': location,
                    'timestamp': time_features.get('timestamp', 0)
                }
                
        except Exception as e:
            logger.error(f"Error predicting supply: {e}")
            # Return default values on error
            return {
                'supply_prediction': 5.0,
                'supply_category': 1,  # Medium supply
                'nearest_node': 0,
                'location': location,
                'timestamp': time_features.get('timestamp', 0)
            }
    
    async def match_rides(self, ride_requests: List[Dict], 
                         available_drivers: List[Dict]) -> Dict:
        """Match ride requests with available drivers."""
        
        if not self.models_loaded or self.matching_algorithm is None:
            raise RuntimeError("Models not loaded")
        
        try:
            # Convert to model format
            riders = [self._dict_to_ride_request(req) for req in ride_requests]
            drivers = [self._dict_to_driver(drv) for drv in available_drivers]
            
            # Perform matching
            result = self.matching_algorithm.match_rides(riders, drivers)
            
            return result
            
        except Exception as e:
            logger.error(f"Error matching rides: {e}")
            # Return empty matches on error
            return {
                'matches': [],
                'unmatched_riders': list(range(len(ride_requests))),
                'predictions': None
            }
    
    def _prepare_demand_input(self, location: Tuple[float, float], 
                            time_features: Dict) -> Dict[str, torch.Tensor]:
        """Prepare input data for demand prediction."""
        
        # Create dummy batch data (in production, this would come from real data)
        batch_size = 1
        seq_len = 24  # 24 hours of historical data
        
        # Time features
        hour = torch.randint(0, 24, (batch_size, seq_len))
        day_of_week = torch.randint(0, 7, (batch_size, seq_len))
        month = torch.randint(0, 12, (batch_size, seq_len))
        
        # Location features
        latitude = torch.full((batch_size, seq_len), location[0])
        longitude = torch.full((batch_size, seq_len), location[1])
        
        # Weather features (dummy data)
        temperature = torch.randn(batch_size, seq_len) * 10 + 20  # Around 20°C
        humidity = torch.rand(batch_size, seq_len) * 100
        precipitation = torch.rand(batch_size, seq_len) * 10
        wind_speed = torch.rand(batch_size, seq_len) * 20
        visibility = torch.rand(batch_size, seq_len) * 10 + 5
        
        # Historical demand (dummy data)
        historical_demand = torch.rand(batch_size, seq_len) * 10
        
        return {
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'latitude': latitude,
            'longitude': longitude,
            'temperature': temperature,
            'humidity': humidity,
            'precipitation': precipitation,
            'wind_speed': wind_speed,
            'visibility': visibility,
            'historical_demand': historical_demand
        }
    
    def _prepare_supply_input(self, location: Tuple[float, float], 
                            time_features: Dict) -> Dict[str, torch.Tensor]:
        """Prepare input data for supply prediction."""
        
        batch_size = 1
        seq_len = 24
        
        # Time features
        hour = torch.randint(0, 24, (batch_size, seq_len))
        day_of_week = torch.randint(0, 7, (batch_size, seq_len))
        
        # Driver state features (dummy data)
        num_active_drivers = torch.rand(batch_size, seq_len) * 100
        num_busy_drivers = torch.rand(batch_size, seq_len) * 50
        avg_driver_rating = torch.rand(batch_size, seq_len) * 2 + 3  # 3-5 rating
        avg_driver_distance = torch.rand(batch_size, seq_len) * 10
        
        # Regional features (dummy data)
        population_density = torch.rand(batch_size, seq_len) * 1000
        poi_count = torch.rand(batch_size, seq_len) * 100
        avg_income = torch.rand(batch_size, seq_len) * 50000 + 30000
        traffic_level = torch.rand(batch_size, seq_len)
        event_indicator = torch.randint(0, 2, (batch_size, seq_len)).float()
        weather_score = torch.rand(batch_size, seq_len)
        
        return {
            'hour': hour,
            'day_of_week': day_of_week,
            'num_active_drivers': num_active_drivers,
            'num_busy_drivers': num_busy_drivers,
            'avg_driver_rating': avg_driver_rating,
            'avg_driver_distance': avg_driver_distance,
            'population_density': population_density,
            'poi_count': poi_count,
            'avg_income': avg_income,
            'traffic_level': traffic_level,
            'event_indicator': event_indicator,
            'weather_score': weather_score
        }
    
    def _generate_default_regions(self, center_location: Tuple[float, float]) -> torch.Tensor:
        """Generate default region locations around a center point."""
        
        lat, lon = center_location
        num_regions = 100
        
        # Create a grid of regions around the center
        lat_range = np.linspace(lat - 0.1, lat + 0.1, 10)
        lon_range = np.linspace(lon - 0.1, lon + 0.1, 10)
        
        regions = []
        for lat_val in lat_range:
            for lon_val in lon_range:
                regions.append([lat_val, lon_val])
        
        return torch.tensor(regions, dtype=torch.float32)
    
    def _dict_to_ride_request(self, req_dict: Dict) -> RideRequest:
        """Convert dictionary to RideRequest object."""
        return RideRequest(
            rider_id=req_dict['rider_id'],
            pickup_lat=req_dict['pickup_lat'],
            pickup_lon=req_dict['pickup_lon'],
            dest_lat=req_dict['dest_lat'],
            dest_lon=req_dict['dest_lon'],
            requested_at=req_dict.get('requested_at', 0),
            vehicle_type=req_dict.get('vehicle_type', 'economy'),
            max_wait_time=req_dict.get('max_wait_time', 10.0),
            price_sensitivity=req_dict.get('price_sensitivity', 1.0)
        )
    
    def _dict_to_driver(self, drv_dict: Dict) -> Driver:
        """Convert dictionary to Driver object."""
        return Driver(
            driver_id=drv_dict['driver_id'],
            current_lat=drv_dict['current_lat'],
            current_lon=drv_dict['current_lon'],
            is_available=drv_dict.get('is_available', True),
            vehicle_type=drv_dict.get('vehicle_type', 'economy'),
            rating=drv_dict.get('rating', 5.0),
            total_rides=drv_dict.get('total_rides', 0),
            last_ride_end=drv_dict.get('last_ride_end', 0)
        )
    
    async def save_models(self):
        """Save all trained models."""
        try:
            models_path = Path(settings.ML_MODEL_PATH)
            
            if self.demand_model:
                torch.save({
                    'model_state_dict': self.demand_model.state_dict(),
                }, models_path / settings.DEMAND_MODEL_NAME)
            
            if self.supply_model:
                torch.save({
                    'model_state_dict': self.supply_model.state_dict(),
                }, models_path / settings.SUPPLY_MODEL_NAME)
            
            if self.matching_algorithm:
                self.matching_algorithm.save_model(
                    str(models_path / settings.MATCHING_MODEL_NAME)
                )
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def get_model_status(self) -> Dict:
        """Get status of all models."""
        return {
            'models_loaded': self.models_loaded,
            'demand_model_loaded': self.demand_model is not None,
            'supply_model_loaded': self.supply_model is not None,
            'matching_algorithm_loaded': self.matching_algorithm is not None,
            'device': str(self.device)
        }
