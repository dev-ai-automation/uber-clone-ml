"""
Ride management API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import List, Optional
import logging
from datetime import datetime

from app.core.database import get_db
from app.models.ride import Ride, RideStatus, VehicleType
from app.models.user import User, UserType
from app.api.v1.schemas import (
    RideRequest, RideResponse, RideUpdate, 
    SurgePricingResponse, ErrorResponse
)
from app.ml.models import ModelManager
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

async def get_model_manager() -> ModelManager:
    """Dependency to get model manager from app state."""
    from app.main import app
    return app.state.model_manager

@router.post("/request", response_model=RideResponse)
async def request_ride(
    ride_request: RideRequest,
    rider_id: int,
    db: AsyncSession = Depends(get_db),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Request a new ride."""
    
    try:
        # Verify rider exists
        rider_query = select(User).where(
            User.id == rider_id, 
            User.user_type == UserType.RIDER
        )
        rider_result = await db.execute(rider_query)
        rider = rider_result.scalar_one_or_none()
        
        if not rider:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Rider not found"
            )
        
        # Calculate estimated distance and duration
        distance_km = calculate_distance(
            ride_request.pickup_latitude, ride_request.pickup_longitude,
            ride_request.destination_latitude, ride_request.destination_longitude
        )
        duration_minutes = estimate_duration(distance_km)
        
        # Get demand and supply predictions
        location = (ride_request.pickup_latitude, ride_request.pickup_longitude)
        time_features = {'timestamp': datetime.now().timestamp()}
        
        demand_prediction = await model_manager.predict_demand(location, time_features)
        supply_prediction = await model_manager.predict_supply(location, time_features)
        
        # Calculate surge pricing
        surge_multiplier = calculate_surge_multiplier(
            demand_prediction['demand_prediction'],
            supply_prediction['supply_prediction']
        )
        
        # Calculate estimated fare
        base_fare = settings.BASE_FARE
        estimated_fare = (
            base_fare + 
            (distance_km * settings.COST_PER_KM) + 
            (duration_minutes * settings.COST_PER_MINUTE)
        ) * surge_multiplier
        
        # Create ride record
        new_ride = Ride(
            rider_id=rider_id,
            status=RideStatus.REQUESTED,
            vehicle_type=VehicleType(ride_request.vehicle_type),
            pickup_latitude=ride_request.pickup_latitude,
            pickup_longitude=ride_request.pickup_longitude,
            pickup_address=ride_request.pickup_address,
            destination_latitude=ride_request.destination_latitude,
            destination_longitude=ride_request.destination_longitude,
            destination_address=ride_request.destination_address,
            estimated_distance_km=distance_km,
            estimated_duration_minutes=duration_minutes,
            base_fare=base_fare,
            surge_multiplier=surge_multiplier,
            estimated_fare=estimated_fare,
            demand_score=demand_prediction['demand_prediction'],
            supply_score=supply_prediction['supply_prediction'],
            notes=ride_request.notes
        )
        
        db.add(new_ride)
        await db.commit()
        await db.refresh(new_ride)
        
        logger.info(f"Ride requested: {new_ride.id} by rider {rider_id}")
        
        return new_ride
        
    except Exception as e:
        logger.error(f"Error requesting ride: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to request ride"
        )

@router.get("/{ride_id}", response_model=RideResponse)
async def get_ride(
    ride_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get ride details by ID."""
    
    query = select(Ride).where(Ride.id == ride_id)
    result = await db.execute(query)
    ride = result.scalar_one_or_none()
    
    if not ride:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ride not found"
        )
    
    return ride

@router.put("/{ride_id}", response_model=RideResponse)
async def update_ride(
    ride_id: int,
    ride_update: RideUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update ride status and details."""
    
    try:
        # Get existing ride
        query = select(Ride).where(Ride.id == ride_id)
        result = await db.execute(query)
        ride = result.scalar_one_or_none()
        
        if not ride:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Ride not found"
            )
        
        # Update fields
        update_data = ride_update.dict(exclude_unset=True)
        
        # Handle status transitions with timestamps
        if 'status' in update_data:
            new_status = update_data['status']
            current_time = datetime.utcnow()
            
            if new_status == RideStatus.MATCHED and not ride.matched_at:
                update_data['matched_at'] = current_time
            elif new_status == RideStatus.ACCEPTED and not ride.accepted_at:
                update_data['accepted_at'] = current_time
            elif new_status == RideStatus.IN_PROGRESS and not ride.started_at:
                update_data['started_at'] = current_time
            elif new_status == RideStatus.COMPLETED and not ride.completed_at:
                update_data['completed_at'] = current_time
            elif new_status == RideStatus.CANCELLED and not ride.cancelled_at:
                update_data['cancelled_at'] = current_time
        
        # Apply updates
        for field, value in update_data.items():
            setattr(ride, field, value)
        
        await db.commit()
        await db.refresh(ride)
        
        logger.info(f"Ride updated: {ride_id}")
        
        return ride
        
    except Exception as e:
        logger.error(f"Error updating ride: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update ride"
        )

@router.get("/rider/{rider_id}", response_model=List[RideResponse])
async def get_rider_rides(
    rider_id: int,
    status: Optional[RideStatus] = None,
    limit: int = 10,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """Get rides for a specific rider."""
    
    query = select(Ride).where(Ride.rider_id == rider_id)
    
    if status:
        query = query.where(Ride.status == status)
    
    query = query.order_by(Ride.requested_at.desc()).limit(limit).offset(offset)
    
    result = await db.execute(query)
    rides = result.scalars().all()
    
    return rides

@router.get("/driver/{driver_id}", response_model=List[RideResponse])
async def get_driver_rides(
    driver_id: int,
    status: Optional[RideStatus] = None,
    limit: int = 10,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """Get rides for a specific driver."""
    
    query = select(Ride).where(Ride.driver_id == driver_id)
    
    if status:
        query = query.where(Ride.status == status)
    
    query = query.order_by(Ride.requested_at.desc()).limit(limit).offset(offset)
    
    result = await db.execute(query)
    rides = result.scalars().all()
    
    return rides

@router.post("/{ride_id}/cancel", response_model=RideResponse)
async def cancel_ride(
    ride_id: int,
    cancellation_reason: str,
    db: AsyncSession = Depends(get_db)
):
    """Cancel a ride."""
    
    try:
        query = select(Ride).where(Ride.id == ride_id)
        result = await db.execute(query)
        ride = result.scalar_one_or_none()
        
        if not ride:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Ride not found"
            )
        
        if ride.status in [RideStatus.COMPLETED, RideStatus.CANCELLED]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot cancel completed or already cancelled ride"
            )
        
        ride.status = RideStatus.CANCELLED
        ride.cancelled_at = datetime.utcnow()
        ride.cancellation_reason = cancellation_reason
        
        await db.commit()
        await db.refresh(ride)
        
        logger.info(f"Ride cancelled: {ride_id}")
        
        return ride
        
    except Exception as e:
        logger.error(f"Error cancelling ride: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel ride"
        )

@router.get("/{ride_id}/surge-pricing", response_model=SurgePricingResponse)
async def get_surge_pricing(
    ride_id: int,
    db: AsyncSession = Depends(get_db),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get current surge pricing for a ride location."""
    
    try:
        query = select(Ride).where(Ride.id == ride_id)
        result = await db.execute(query)
        ride = result.scalar_one_or_none()
        
        if not ride:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Ride not found"
            )
        
        location = (ride.pickup_latitude, ride.pickup_longitude)
        time_features = {'timestamp': datetime.now().timestamp()}
        
        # Get current predictions
        demand_prediction = await model_manager.predict_demand(location, time_features)
        supply_prediction = await model_manager.predict_supply(location, time_features)
        
        # Calculate surge multiplier
        surge_multiplier = calculate_surge_multiplier(
            demand_prediction['demand_prediction'],
            supply_prediction['supply_prediction']
        )
        
        base_fare = settings.BASE_FARE
        estimated_fare = (
            base_fare + 
            (ride.estimated_distance_km * settings.COST_PER_KM) + 
            (ride.estimated_duration_minutes * settings.COST_PER_MINUTE)
        ) * surge_multiplier
        
        return SurgePricingResponse(
            base_fare=base_fare,
            surge_multiplier=surge_multiplier,
            estimated_fare=estimated_fare,
            demand_level=categorize_demand(demand_prediction['demand_prediction']),
            supply_level=categorize_supply(supply_prediction['supply_prediction']),
            location=[ride.pickup_latitude, ride.pickup_longitude]
        )
        
    except Exception as e:
        logger.error(f"Error getting surge pricing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get surge pricing"
        )

# Helper functions
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula."""
    import math
    
    R = 6371  # Earth's radius in km
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat/2)**2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(dlon/2)**2)
    
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def estimate_duration(distance_km: float) -> int:
    """Estimate trip duration based on distance."""
    # Assume average speed of 30 km/h in city
    average_speed_kmh = 30
    duration_hours = distance_km / average_speed_kmh
    return int(duration_hours * 60)  # Convert to minutes

def calculate_surge_multiplier(demand: float, supply: float) -> float:
    """Calculate surge multiplier based on demand and supply."""
    if supply == 0:
        return settings.SURGE_MULTIPLIER_MAX
    
    ratio = demand / supply
    
    if ratio < 0.5:
        return 1.0  # No surge
    elif ratio < 1.0:
        return 1.2
    elif ratio < 1.5:
        return 1.5
    elif ratio < 2.0:
        return 2.0
    else:
        return min(settings.SURGE_MULTIPLIER_MAX, 2.5)

def categorize_demand(demand: float) -> str:
    """Categorize demand level."""
    if demand < 2.0:
        return "low"
    elif demand < 5.0:
        return "medium"
    else:
        return "high"

def categorize_supply(supply: float) -> str:
    """Categorize supply level."""
    if supply < 3.0:
        return "low"
    elif supply < 7.0:
        return "medium"
    else:
        return "high"
