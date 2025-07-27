"""
Driver management API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import List, Optional
import logging
from datetime import datetime, timedelta

from app.core.database import get_db
from app.models.user import User, UserType, UserStatus
from app.models.ride import Ride, RideStatus
from app.api.v1.schemas import (
    UserResponse, UserUpdate, DriverLocationUpdate,
    MatchingRequest, MatchingResponse, ErrorResponse
)
from app.ml.models import ModelManager

logger = logging.getLogger(__name__)
router = APIRouter()

async def get_model_manager() -> ModelManager:
    """Dependency to get model manager from app state."""
    from app.main import app
    return app.state.model_manager

@router.get("/available", response_model=List[UserResponse])
async def get_available_drivers(
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    radius_km: float = 10.0,
    vehicle_type: Optional[str] = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """Get list of available drivers, optionally filtered by location and vehicle type."""
    
    try:
        query = select(User).where(
            User.user_type == UserType.DRIVER,
            User.status == UserStatus.ACTIVE,
            User.is_available == True
        )
        
        if vehicle_type:
            query = query.where(User.vehicle_type == vehicle_type)
        
        # Add location filtering if coordinates provided
        if latitude is not None and longitude is not None:
            # Simple bounding box filter (for production, use PostGIS or similar)
            lat_delta = radius_km / 111.0  # Rough conversion
            lon_delta = radius_km / (111.0 * abs(latitude))
            
            query = query.where(
                User.current_latitude.between(latitude - lat_delta, latitude + lat_delta),
                User.current_longitude.between(longitude - lon_delta, longitude + lon_delta)
            )
        
        query = query.limit(limit)
        
        result = await db.execute(query)
        drivers = result.scalars().all()
        
        return drivers
        
    except Exception as e:
        logger.error(f"Error getting available drivers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get available drivers"
        )

@router.get("/{driver_id}", response_model=UserResponse)
async def get_driver(
    driver_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get driver details by ID."""
    
    query = select(User).where(
        User.id == driver_id,
        User.user_type == UserType.DRIVER
    )
    result = await db.execute(query)
    driver = result.scalar_one_or_none()
    
    if not driver:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Driver not found"
        )
    
    return driver

@router.put("/{driver_id}/location", response_model=UserResponse)
async def update_driver_location(
    driver_id: int,
    location_update: DriverLocationUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update driver's current location and availability."""
    
    try:
        query = select(User).where(
            User.id == driver_id,
            User.user_type == UserType.DRIVER
        )
        result = await db.execute(query)
        driver = result.scalar_one_or_none()
        
        if not driver:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Driver not found"
            )
        
        # Update location and availability
        driver.current_latitude = location_update.latitude
        driver.current_longitude = location_update.longitude
        driver.is_available = location_update.is_available
        driver.last_active = datetime.utcnow()
        
        await db.commit()
        await db.refresh(driver)
        
        logger.info(f"Driver location updated: {driver_id}")
        
        return driver
        
    except Exception as e:
        logger.error(f"Error updating driver location: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update driver location"
        )

@router.put("/{driver_id}/availability", response_model=UserResponse)
async def toggle_driver_availability(
    driver_id: int,
    is_available: bool,
    db: AsyncSession = Depends(get_db)
):
    """Toggle driver availability status."""
    
    try:
        query = select(User).where(
            User.id == driver_id,
            User.user_type == UserType.DRIVER
        )
        result = await db.execute(query)
        driver = result.scalar_one_or_none()
        
        if not driver:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Driver not found"
            )
        
        driver.is_available = is_available
        driver.last_active = datetime.utcnow()
        
        await db.commit()
        await db.refresh(driver)
        
        logger.info(f"Driver availability updated: {driver_id} -> {is_available}")
        
        return driver
        
    except Exception as e:
        logger.error(f"Error updating driver availability: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update driver availability"
        )

@router.get("/{driver_id}/current-ride", response_model=Optional[dict])
async def get_driver_current_ride(
    driver_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get driver's current active ride."""
    
    try:
        query = select(Ride).where(
            Ride.driver_id == driver_id,
            Ride.status.in_([
                RideStatus.MATCHED,
                RideStatus.ACCEPTED,
                RideStatus.DRIVER_ARRIVED,
                RideStatus.IN_PROGRESS
            ])
        ).order_by(Ride.requested_at.desc())
        
        result = await db.execute(query)
        current_ride = result.scalar_one_or_none()
        
        if not current_ride:
            return None
        
        return {
            "ride_id": current_ride.id,
            "status": current_ride.status,
            "pickup_location": {
                "latitude": current_ride.pickup_latitude,
                "longitude": current_ride.pickup_longitude,
                "address": current_ride.pickup_address
            },
            "destination": {
                "latitude": current_ride.destination_latitude,
                "longitude": current_ride.destination_longitude,
                "address": current_ride.destination_address
            },
            "rider_id": current_ride.rider_id,
            "estimated_fare": current_ride.estimated_fare,
            "requested_at": current_ride.requested_at
        }
        
    except Exception as e:
        logger.error(f"Error getting driver current ride: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get current ride"
        )

@router.post("/{driver_id}/accept-ride/{ride_id}")
async def accept_ride(
    driver_id: int,
    ride_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Driver accepts a matched ride."""
    
    try:
        # Verify driver exists and is available
        driver_query = select(User).where(
            User.id == driver_id,
            User.user_type == UserType.DRIVER,
            User.is_available == True
        )
        driver_result = await db.execute(driver_query)
        driver = driver_result.scalar_one_or_none()
        
        if not driver:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Driver not found or not available"
            )
        
        # Get the ride
        ride_query = select(Ride).where(
            Ride.id == ride_id,
            Ride.driver_id == driver_id,
            Ride.status == RideStatus.MATCHED
        )
        ride_result = await db.execute(ride_query)
        ride = ride_result.scalar_one_or_none()
        
        if not ride:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Ride not found or not available for acceptance"
            )
        
        # Update ride status
        ride.status = RideStatus.ACCEPTED
        ride.accepted_at = datetime.utcnow()
        
        # Make driver unavailable
        driver.is_available = False
        
        await db.commit()
        
        logger.info(f"Ride accepted: {ride_id} by driver {driver_id}")
        
        return {"message": "Ride accepted successfully", "ride_id": ride_id}
        
    except Exception as e:
        logger.error(f"Error accepting ride: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to accept ride"
        )

@router.post("/{driver_id}/complete-ride/{ride_id}")
async def complete_ride(
    driver_id: int,
    ride_id: int,
    actual_distance_km: Optional[float] = None,
    actual_duration_minutes: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
):
    """Driver completes a ride."""
    
    try:
        # Get the ride
        ride_query = select(Ride).where(
            Ride.id == ride_id,
            Ride.driver_id == driver_id,
            Ride.status == RideStatus.IN_PROGRESS
        )
        ride_result = await db.execute(ride_query)
        ride = ride_result.scalar_one_or_none()
        
        if not ride:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Ride not found or not in progress"
            )
        
        # Update ride completion details
        ride.status = RideStatus.COMPLETED
        ride.completed_at = datetime.utcnow()
        
        if actual_distance_km:
            ride.actual_distance_km = actual_distance_km
        if actual_duration_minutes:
            ride.actual_duration_minutes = actual_duration_minutes
        
        # Calculate final fare (could be adjusted based on actual distance/time)
        if actual_distance_km and actual_duration_minutes:
            from app.core.config import settings
            final_fare = (
                settings.BASE_FARE + 
                (actual_distance_km * settings.COST_PER_KM) + 
                (actual_duration_minutes * settings.COST_PER_MINUTE)
            ) * ride.surge_multiplier
            ride.final_fare = final_fare
        else:
            ride.final_fare = ride.estimated_fare
        
        # Update driver and rider stats
        driver_query = select(User).where(User.id == driver_id)
        driver_result = await db.execute(driver_query)
        driver = driver_result.scalar_one_or_none()
        
        rider_query = select(User).where(User.id == ride.rider_id)
        rider_result = await db.execute(rider_query)
        rider = rider_result.scalar_one_or_none()
        
        if driver:
            driver.total_rides += 1
            driver.is_available = True  # Make driver available again
        
        if rider:
            rider.total_rides += 1
        
        await db.commit()
        
        logger.info(f"Ride completed: {ride_id} by driver {driver_id}")
        
        return {
            "message": "Ride completed successfully",
            "ride_id": ride_id,
            "final_fare": ride.final_fare
        }
        
    except Exception as e:
        logger.error(f"Error completing ride: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete ride"
        )

@router.get("/{driver_id}/earnings")
async def get_driver_earnings(
    driver_id: int,
    days: int = 7,
    db: AsyncSession = Depends(get_db)
):
    """Get driver earnings for specified period."""
    
    try:
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get completed rides in the period
        query = select(Ride).where(
            Ride.driver_id == driver_id,
            Ride.status == RideStatus.COMPLETED,
            Ride.completed_at >= start_date,
            Ride.completed_at <= end_date
        )
        
        result = await db.execute(query)
        rides = result.scalars().all()
        
        # Calculate earnings
        total_earnings = sum(ride.final_fare or ride.estimated_fare or 0 for ride in rides)
        total_rides = len(rides)
        total_distance = sum(ride.actual_distance_km or ride.estimated_distance_km or 0 for ride in rides)
        total_duration = sum(ride.actual_duration_minutes or ride.estimated_duration_minutes or 0 for ride in rides)
        
        average_fare = total_earnings / total_rides if total_rides > 0 else 0
        
        return {
            "driver_id": driver_id,
            "period_days": days,
            "total_earnings": round(total_earnings, 2),
            "total_rides": total_rides,
            "total_distance_km": round(total_distance, 2),
            "total_duration_minutes": total_duration,
            "average_fare": round(average_fare, 2),
            "earnings_per_km": round(total_earnings / total_distance, 2) if total_distance > 0 else 0,
            "earnings_per_hour": round(total_earnings / (total_duration / 60), 2) if total_duration > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting driver earnings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get driver earnings"
        )
