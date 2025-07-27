-- Database initialization script
-- This script will be run when the PostgreSQL container starts

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create indexes for better performance
-- These will be created after tables are created by SQLAlchemy

-- Function to create indexes after tables exist
CREATE OR REPLACE FUNCTION create_performance_indexes() RETURNS void AS $$
BEGIN
    -- User indexes
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users') THEN
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        CREATE INDEX IF NOT EXISTS idx_users_phone ON users(phone);
        CREATE INDEX IF NOT EXISTS idx_users_type ON users(user_type);
        CREATE INDEX IF NOT EXISTS idx_users_available ON users(is_available);
        CREATE INDEX IF NOT EXISTS idx_users_location ON users(current_latitude, current_longitude);
    END IF;

    -- Ride indexes
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'rides') THEN
        CREATE INDEX IF NOT EXISTS idx_rides_rider_id ON rides(rider_id);
        CREATE INDEX IF NOT EXISTS idx_rides_driver_id ON rides(driver_id);
        CREATE INDEX IF NOT EXISTS idx_rides_status ON rides(status);
        CREATE INDEX IF NOT EXISTS idx_rides_requested_at ON rides(requested_at);
        CREATE INDEX IF NOT EXISTS idx_rides_pickup_location ON rides(pickup_latitude, pickup_longitude);
    END IF;
END;
$$ LANGUAGE plpgsql;
