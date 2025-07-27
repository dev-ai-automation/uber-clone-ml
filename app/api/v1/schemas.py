"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class VehicleType(str, Enum):
    ECONOMY = "economy"
    COMFORT = "comfort"
    PREMIUM = "premium"
    XL = "xl"

class RideStatus(str, Enum):
    REQUESTED = "requested"
    MATCHED = "matched"
    ACCEPTED = "accepted"
    DRIVER_ARRIVED = "driver_arrived"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class UserType(str, Enum):
    RIDER = "rider"
    DRIVER = "driver"

# User schemas
class UserCreate(BaseModel):
    email: str = Field(..., description="User email address")
    phone: str = Field(..., description="User phone number")
    first_name: str = Field(..., description="User first name")
    last_name: str = Field(..., description="User last name")
    user_type: UserType = Field(..., description="Type of user")
    vehicle_type: Optional[str] = Field(None, description="Vehicle type for drivers")
    license_plate: Optional[str] = Field(None, description="License plate for drivers")

class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    current_latitude: Optional[float] = None
    current_longitude: Optional[float] = None
    is_available: Optional[bool] = None
    vehicle_type: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    email: str
    phone: str
    first_name: str
    last_name: str
    user_type: UserType
    current_latitude: Optional[float]
    current_longitude: Optional[float]
    is_available: Optional[bool]
    vehicle_type: Optional[str]
    license_plate: Optional[str]
    rating: float
    total_rides: int
    created_at: datetime

    class Config:
        from_attributes = True

# Ride schemas
class RideRequest(BaseModel):
    pickup_latitude: float = Field(..., ge=-90, le=90, description="Pickup latitude")
    pickup_longitude: float = Field(..., ge=-180, le=180, description="Pickup longitude")
    pickup_address: Optional[str] = Field(None, description="Pickup address")
    destination_latitude: float = Field(..., ge=-90, le=90, description="Destination latitude")
    destination_longitude: float = Field(..., ge=-180, le=180, description="Destination longitude")
    destination_address: Optional[str] = Field(None, description="Destination address")
    vehicle_type: VehicleType = Field(VehicleType.ECONOMY, description="Requested vehicle type")
    notes: Optional[str] = Field(None, description="Additional notes")

class RideResponse(BaseModel):
    id: int
    rider_id: int
    driver_id: Optional[int]
    status: RideStatus
    vehicle_type: VehicleType
    pickup_latitude: float
    pickup_longitude: float
    pickup_address: Optional[str]
    destination_latitude: float
    destination_longitude: float
    destination_address: Optional[str]
    estimated_distance_km: Optional[float]
    estimated_duration_minutes: Optional[int]
    estimated_fare: Optional[float]
    surge_multiplier: float
    demand_score: Optional[float]
    supply_score: Optional[float]
    matching_score: Optional[float]
    requested_at: datetime
    matched_at: Optional[datetime]
    notes: Optional[str]

    class Config:
        from_attributes = True

class RideUpdate(BaseModel):
    status: Optional[RideStatus] = None
    driver_id: Optional[int] = None
    actual_distance_km: Optional[float] = None
    actual_duration_minutes: Optional[int] = None
    final_fare: Optional[float] = None
    cancellation_reason: Optional[str] = None

# ML Prediction schemas
class DemandPredictionRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    timestamp: Optional[datetime] = Field(None, description="Prediction timestamp")

class DemandPredictionResponse(BaseModel):
    demand_prediction: float = Field(..., description="Predicted demand level")
    confidence_lower: float = Field(..., description="Lower confidence bound")
    confidence_upper: float = Field(..., description="Upper confidence bound")
    uncertainty: float = Field(..., description="Prediction uncertainty")
    location: List[float] = Field(..., description="[latitude, longitude]")
    timestamp: float

class SupplyPredictionRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    timestamp: Optional[datetime] = Field(None, description="Prediction timestamp")

class SupplyPredictionResponse(BaseModel):
    supply_prediction: float = Field(..., description="Predicted supply level")
    supply_category: int = Field(..., description="Supply category (0=low, 1=medium, 2=high)")
    nearest_node: int = Field(..., description="Nearest geographic node")
    location: List[float] = Field(..., description="[latitude, longitude]")
    timestamp: float

# Matching schemas
class MatchingRequest(BaseModel):
    ride_requests: List[Dict[str, Any]] = Field(..., description="List of ride requests")
    available_drivers: List[Dict[str, Any]] = Field(..., description="List of available drivers")

class MatchResult(BaseModel):
    rider_idx: int
    driver_idx: int
    rider_id: int
    driver_id: int
    matching_score: float
    estimated_eta: float
    estimated_price: float
    pickup_distance: float

class MatchingResponse(BaseModel):
    matches: List[MatchResult]
    unmatched_riders: List[int]
    total_requests: int
    total_drivers: int
    match_rate: float

# Driver location update
class DriverLocationUpdate(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    is_available: bool = Field(True, description="Driver availability status")

# Surge pricing schema
class SurgePricingResponse(BaseModel):
    base_fare: float
    surge_multiplier: float
    estimated_fare: float
    demand_level: str
    supply_level: str
    location: List[float]

# Analytics schemas
class RideAnalytics(BaseModel):
    total_rides: int
    completed_rides: int
    cancelled_rides: int
    average_rating: float
    average_fare: float
    average_distance: float
    average_duration: float
    peak_hours: List[int]
    popular_areas: List[Dict[str, Any]]

class DriverAnalytics(BaseModel):
    total_drivers: int
    active_drivers: int
    average_rating: float
    total_rides_completed: int
    average_earnings: float
    top_performers: List[Dict[str, Any]]

# Error schemas
class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None

# Health check schema
class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: datetime
    models_loaded: bool
    database_connected: bool
    redis_connected: bool
