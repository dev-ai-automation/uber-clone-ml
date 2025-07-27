"""
Configuration settings for the Uber Clone ML application.
"""

from pydantic_settings import BaseSettings
from pydantic import validator, Field
from typing import Optional
import os
import secrets
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Application settings with security best practices."""
    
    # Environment
    ENVIRONMENT: str = Field(default="development", description="Application environment")
    DEBUG: bool = Field(default=False, description="Debug mode")
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Uber Clone ML"
    
    # Database Settings - No default credentials
    DATABASE_URL: str = Field(..., description="Database connection URL")
    
    # Redis Settings - No default credentials
    REDIS_URL: str = Field(..., description="Redis connection URL")
    
    # ML Model Settings
    ML_MODEL_PATH: str = Field(default="./models", description="Path to ML models")
    DEMAND_MODEL_NAME: str = "demand_predictor.pth"
    SUPPLY_MODEL_NAME: str = "supply_predictor.pth"
    MATCHING_MODEL_NAME: str = "matching_model.pth"
    
    # Geolocation Settings
    DEFAULT_SEARCH_RADIUS_KM: float = Field(default=5.0, ge=0.1, le=50.0)
    MAX_SEARCH_RADIUS_KM: float = Field(default=20.0, ge=1.0, le=100.0)
    
    # Pricing Settings
    BASE_FARE: float = Field(default=2.50, ge=0.0)
    COST_PER_KM: float = Field(default=1.20, ge=0.0)
    COST_PER_MINUTE: float = Field(default=0.25, ge=0.0)
    SURGE_MULTIPLIER_MAX: float = Field(default=3.0, ge=1.0, le=10.0)
    
    # Security Settings - Generate secure defaults
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, ge=5, le=1440)
    
    # Password Security
    MIN_PASSWORD_LENGTH: int = 8
    REQUIRE_SPECIAL_CHARS: bool = True
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=100, ge=10, le=1000)
    
    # Logging Settings
    LOG_LEVEL: str = Field(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    LOG_FILE: str = "logs/app.log"
    
    # Celery Settings - Use Redis URL
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None
    
    # Model Training Settings
    BATCH_SIZE: int = Field(default=32, ge=1, le=512)
    LEARNING_RATE: float = Field(default=0.001, gt=0.0, le=1.0)
    EPOCHS: int = Field(default=100, ge=1, le=1000)
    DEVICE: str = Field(default="cpu", description="Training device")
    
    # Security Headers
    CORS_ORIGINS: list = Field(default=["http://localhost:3000"], description="Allowed CORS origins")
    ALLOWED_HOSTS: list = Field(default=["localhost", "127.0.0.1"], description="Allowed hosts")
    
    @validator('SECRET_KEY')
    def validate_secret_key(cls, v):
        if len(v) < 32:
            logger.warning("SECRET_KEY should be at least 32 characters long")
        if v == "your-secret-key-change-in-production":
            raise ValueError("SECRET_KEY must be changed from default value")
        return v
    
    @validator('DATABASE_URL')
    def validate_database_url(cls, v):
        if not v.startswith(('postgresql://', 'postgresql+asyncpg://')):
            raise ValueError("DATABASE_URL must be a valid PostgreSQL URL")
        return v
    
    @validator('REDIS_URL')
    def validate_redis_url(cls, v):
        if not v.startswith('redis://'):
            raise ValueError("REDIS_URL must be a valid Redis URL")
        return v
    
    @validator('CELERY_BROKER_URL', 'CELERY_RESULT_BACKEND', pre=True, always=True)
    def set_celery_urls(cls, v, values):
        if v is None and 'REDIS_URL' in values:
            return values['REDIS_URL']
        return v
    
    @validator('ENVIRONMENT')
    def validate_environment(cls, v):
        allowed_envs = ['development', 'staging', 'production']
        if v not in allowed_envs:
            raise ValueError(f"ENVIRONMENT must be one of: {allowed_envs}")
        return v
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == "development"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        validate_assignment = True
        extra = "forbid"  # Prevent extra fields

# Global settings instance with error handling
try:
    settings = Settings()
    if settings.is_production() and settings.DEBUG:
        logger.warning("DEBUG mode is enabled in production environment")
except Exception as e:
    logger.error(f"Failed to load settings: {e}")
    raise
