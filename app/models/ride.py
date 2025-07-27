"""
Ride model for managing ride requests and trips.
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Enum, ForeignKey, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum
from app.core.database import Base

class RideStatus(enum.Enum):
    REQUESTED = "requested"
    MATCHED = "matched"
    ACCEPTED = "accepted"
    DRIVER_ARRIVED = "driver_arrived"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class VehicleType(enum.Enum):
    ECONOMY = "economy"
    COMFORT = "comfort"
    PREMIUM = "premium"
    XL = "xl"

class Ride(Base):
    """Ride model for managing trips."""
    
    __tablename__ = "rides"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # User relationships
    rider_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    driver_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Ride details
    status = Column(Enum(RideStatus), default=RideStatus.REQUESTED)
    vehicle_type = Column(Enum(VehicleType), default=VehicleType.ECONOMY)
    
    # Location data
    pickup_latitude = Column(Float, nullable=False)
    pickup_longitude = Column(Float, nullable=False)
    pickup_address = Column(String, nullable=True)
    
    destination_latitude = Column(Float, nullable=False)
    destination_longitude = Column(Float, nullable=False)
    destination_address = Column(String, nullable=True)
    
    # Trip metrics
    estimated_distance_km = Column(Float, nullable=True)
    actual_distance_km = Column(Float, nullable=True)
    estimated_duration_minutes = Column(Integer, nullable=True)
    actual_duration_minutes = Column(Integer, nullable=True)
    
    # Pricing
    base_fare = Column(Float, nullable=True)
    surge_multiplier = Column(Float, default=1.0)
    estimated_fare = Column(Float, nullable=True)
    final_fare = Column(Float, nullable=True)
    
    # ML predictions
    demand_score = Column(Float, nullable=True)  # Predicted demand at pickup location
    supply_score = Column(Float, nullable=True)  # Available drivers nearby
    matching_score = Column(Float, nullable=True)  # ML matching confidence
    
    # Timestamps
    requested_at = Column(DateTime(timezone=True), server_default=func.now())
    matched_at = Column(DateTime(timezone=True), nullable=True)
    accepted_at = Column(DateTime(timezone=True), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    cancelled_at = Column(DateTime(timezone=True), nullable=True)
    
    # Additional info
    notes = Column(Text, nullable=True)
    cancellation_reason = Column(String, nullable=True)
    
    # Relationships
    rider = relationship("User", foreign_keys=[rider_id], back_populates="rider_rides")
    driver = relationship("User", foreign_keys=[driver_id], back_populates="driver_rides")
    
    def __repr__(self):
        return f"<Ride(id={self.id}, status={self.status}, rider_id={self.rider_id})>"
