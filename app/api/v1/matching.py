"""
ML-powered matching API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import List, Dict, Any
import logging
from datetime import datetime

from app.core.database import get_db
from app.models.ride import Ride, RideStatus
from app.models.user import User, UserType
from app.api.v1.schemas import (
    MatchingRequest, MatchingResponse, 
    DemandPredictionRequest, DemandPredictionResponse,
    SupplyPredictionRequest, SupplyPredictionResponse
)
from app.ml.models import ModelManager

logger = logging.getLogger(__name__)
router = APIRouter()

async def get_model_manager() -> ModelManager:
    """Dependency to get model manager from app state."""
    from app.main import app
    return app.state.model_manager

@router.post("/match-rides", response_model=MatchingResponse)
async def match_rides(
    matching_request: MatchingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Match ride requests with available drivers using ML algorithm."""
    
    try:
        # Perform ML-based matching
        result = await model_manager.match_rides(
            matching_request.ride_requests,
            matching_request.available_drivers
        )
        
        # Update database with matches in background
        background_tasks.add_task(
            update_matched_rides,
            result['matches'],
            db
        )
        
        # Calculate match rate
        total_requests = len(matching_request.ride_requests)
        total_drivers = len(matching_request.available_drivers)
        match_rate = len(result['matches']) / total_requests if total_requests > 0 else 0
        
        return MatchingResponse(
            matches=result['matches'],
            unmatched_riders=result['unmatched_riders'],
            total_requests=total_requests,
            total_drivers=total_drivers,
            match_rate=match_rate
        )
        
    except Exception as e:
        logger.error(f"Error matching rides: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to match rides"
        )

@router.post("/predict-demand", response_model=DemandPredictionResponse)
async def predict_demand(
    request: DemandPredictionRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Predict demand at a specific location and time."""
    
    try:
        location = (request.latitude, request.longitude)
        timestamp = request.timestamp.timestamp() if request.timestamp else datetime.now().timestamp()
        time_features = {'timestamp': timestamp}
        
        prediction = await model_manager.predict_demand(location, time_features)
        
        return DemandPredictionResponse(**prediction)
        
    except Exception as e:
        logger.error(f"Error predicting demand: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to predict demand"
        )

@router.post("/predict-supply", response_model=SupplyPredictionResponse)
async def predict_supply(
    request: SupplyPredictionRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Predict supply at a specific location and time."""
    
    try:
        location = (request.latitude, request.longitude)
        timestamp = request.timestamp.timestamp() if request.timestamp else datetime.now().timestamp()
        time_features = {'timestamp': timestamp}
        
        prediction = await model_manager.predict_supply(location, time_features)
        
        return SupplyPredictionResponse(**prediction)
        
    except Exception as e:
        logger.error(f"Error predicting supply: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to predict supply"
        )

@router.post("/auto-match")
async def auto_match_pending_rides(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Automatically match all pending ride requests with available drivers."""
    
    try:
        # Get pending ride requests
        pending_rides_query = select(Ride).where(Ride.status == RideStatus.REQUESTED)
        pending_result = await db.execute(pending_rides_query)
        pending_rides = pending_result.scalars().all()
        
        if not pending_rides:
            return {"message": "No pending rides to match", "matched_count": 0}
        
        # Get available drivers
        available_drivers_query = select(User).where(
            User.user_type == UserType.DRIVER,
            User.is_available == True
        )
        available_result = await db.execute(available_drivers_query)
        available_drivers = available_result.scalars().all()
        
        if not available_drivers:
            return {"message": "No available drivers", "matched_count": 0}
        
        # Convert to format expected by ML model
        ride_requests = []
        for ride in pending_rides:
            ride_requests.append({
                'rider_id': ride.rider_id,
                'pickup_lat': ride.pickup_latitude,
                'pickup_lon': ride.pickup_longitude,
                'dest_lat': ride.destination_latitude,
                'dest_lon': ride.destination_longitude,
                'requested_at': ride.requested_at.timestamp(),
                'vehicle_type': ride.vehicle_type.value,
                'max_wait_time': 15.0,  # Default 15 minutes
                'price_sensitivity': 1.0
            })
        
        driver_list = []
        for driver in available_drivers:
            driver_list.append({
                'driver_id': driver.id,
                'current_lat': driver.current_latitude or 0.0,
                'current_lon': driver.current_longitude or 0.0,
                'is_available': driver.is_available,
                'vehicle_type': driver.vehicle_type or 'economy',
                'rating': driver.rating,
                'total_rides': driver.total_rides,
                'last_ride_end': 0
            })
        
        # Perform matching
        result = await model_manager.match_rides(ride_requests, driver_list)
        
        # Update database with matches
        matched_count = 0
        for match in result['matches']:
            ride_idx = match['rider_idx']
            driver_idx = match['driver_idx']
            
            ride = pending_rides[ride_idx]
            driver = available_drivers[driver_idx]
            
            # Update ride with matched driver
            ride.driver_id = driver.id
            ride.status = RideStatus.MATCHED
            ride.matched_at = datetime.utcnow()
            ride.matching_score = match['matching_score']
            
            # Update driver availability
            driver.is_available = False
            
            matched_count += 1
        
        await db.commit()
        
        logger.info(f"Auto-matched {matched_count} rides")
        
        return {
            "message": f"Successfully matched {matched_count} rides",
            "matched_count": matched_count,
            "total_pending": len(pending_rides),
            "total_drivers": len(available_drivers)
        }
        
    except Exception as e:
        logger.error(f"Error in auto-matching: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to auto-match rides"
        )

@router.get("/market-analysis")
async def get_market_analysis(
    latitude: float,
    longitude: float,
    radius_km: float = 5.0,
    db: AsyncSession = Depends(get_db),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Get comprehensive market analysis for a location."""
    
    try:
        location = (latitude, longitude)
        time_features = {'timestamp': datetime.now().timestamp()}
        
        # Get ML predictions
        demand_prediction = await model_manager.predict_demand(location, time_features)
        supply_prediction = await model_manager.predict_supply(location, time_features)
        
        # Get current ride statistics from database
        current_time = datetime.utcnow()
        
        # Active rides in area (simplified query)
        active_rides_query = select(Ride).where(
            Ride.status.in_([RideStatus.REQUESTED, RideStatus.MATCHED, RideStatus.IN_PROGRESS])
        )
        active_result = await db.execute(active_rides_query)
        active_rides = len(active_result.scalars().all())
        
        # Available drivers in area
        available_drivers_query = select(User).where(
            User.user_type == UserType.DRIVER,
            User.is_available == True
        )
        available_result = await db.execute(available_drivers_query)
        available_drivers = len(available_result.scalars().all())
        
        # Calculate market metrics
        supply_demand_ratio = supply_prediction['supply_prediction'] / max(demand_prediction['demand_prediction'], 0.1)
        
        market_condition = "balanced"
        if supply_demand_ratio < 0.5:
            market_condition = "high_demand"
        elif supply_demand_ratio > 2.0:
            market_condition = "oversupply"
        
        # Calculate recommended surge multiplier
        from app.api.v1.rides import calculate_surge_multiplier
        surge_multiplier = calculate_surge_multiplier(
            demand_prediction['demand_prediction'],
            supply_prediction['supply_prediction']
        )
        
        return {
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "radius_km": radius_km
            },
            "timestamp": current_time.isoformat(),
            "demand": {
                "predicted_level": demand_prediction['demand_prediction'],
                "confidence_interval": [
                    demand_prediction['confidence_lower'],
                    demand_prediction['confidence_upper']
                ],
                "uncertainty": demand_prediction['uncertainty'],
                "category": "high" if demand_prediction['demand_prediction'] > 5 else "medium" if demand_prediction['demand_prediction'] > 2 else "low"
            },
            "supply": {
                "predicted_level": supply_prediction['supply_prediction'],
                "category": supply_prediction['supply_category'],
                "available_drivers": available_drivers
            },
            "market_metrics": {
                "supply_demand_ratio": round(supply_demand_ratio, 2),
                "market_condition": market_condition,
                "recommended_surge_multiplier": surge_multiplier,
                "active_rides": active_rides
            },
            "recommendations": generate_market_recommendations(
                demand_prediction['demand_prediction'],
                supply_prediction['supply_prediction'],
                market_condition
            )
        }
        
    except Exception as e:
        logger.error(f"Error in market analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to perform market analysis"
        )

async def update_matched_rides(matches: List[Dict], db: AsyncSession):
    """Background task to update database with matched rides."""
    try:
        for match in matches:
            # This would need to be implemented based on your specific matching result format
            # Update ride and driver records
            pass
    except Exception as e:
        logger.error(f"Error updating matched rides: {e}")

def generate_market_recommendations(demand: float, supply: float, condition: str) -> List[str]:
    """Generate market recommendations based on supply/demand analysis."""
    recommendations = []
    
    if condition == "high_demand":
        recommendations.extend([
            "Consider implementing surge pricing to balance demand",
            "Incentivize more drivers to come online in this area",
            "Alert nearby drivers about high demand opportunity"
        ])
    elif condition == "oversupply":
        recommendations.extend([
            "Consider driver redistribution to higher demand areas",
            "Reduce driver incentives in this area temporarily",
            "Focus marketing efforts on rider acquisition"
        ])
    else:
        recommendations.extend([
            "Market conditions are balanced",
            "Maintain current pricing strategy",
            "Monitor for changes in demand patterns"
        ])
    
    # Add demand-specific recommendations
    if demand > 7:
        recommendations.append("Very high demand detected - consider premium service offerings")
    elif demand < 1:
        recommendations.append("Low demand period - good time for driver breaks or maintenance")
    
    return recommendations
