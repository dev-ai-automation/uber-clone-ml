"""
User model for riders and drivers.
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum
from app.core.database import Base

class UserType(enum.Enum):
    RIDER = "rider"
    DRIVER = "driver"
    ADMIN = "admin"

class UserStatus(enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"

class User(Base):
    """User model for both riders and drivers."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    phone = Column(String, unique=True, index=True, nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    user_type = Column(Enum(UserType), nullable=False)
    status = Column(Enum(UserStatus), default=UserStatus.ACTIVE)
    
    # Location data
    current_latitude = Column(Float, nullable=True)
    current_longitude = Column(Float, nullable=True)
    
    # Driver-specific fields
    is_available = Column(Boolean, default=False)  # For drivers
    vehicle_type = Column(String, nullable=True)  # For drivers
    license_plate = Column(String, nullable=True)  # For drivers
    rating = Column(Float, default=5.0)
    total_rides = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_active = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    rider_rides = relationship("Ride", foreign_keys="Ride.rider_id", back_populates="rider")
    driver_rides = relationship("Ride", foreign_keys="Ride.driver_id", back_populates="driver")
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, type={self.user_type})>"
