# Uber Clone ML - Advanced Ride-Sharing Platform

A sophisticated ride-sharing application that replicates Uber's supply and demand algorithm using state-of-the-art machine learning models built with PyTorch. The system features clean, scalable architecture with comprehensive REST API endpoints.

## 🚀 Features

### Core Functionality
- **Intelligent Ride Matching**: ML-powered algorithm using attention mechanisms and optimal transport theory
- **Dynamic Demand Prediction**: Transformer-based model for spatio-temporal demand forecasting
- **Supply Optimization**: Graph Neural Networks for driver availability prediction
- **Surge Pricing**: Real-time pricing based on supply/demand dynamics
- **Real-time Tracking**: Driver location updates and ride status management

### Machine Learning Models
- **Demand Predictor**: Transformer architecture with positional encoding and spatial attention
- **Supply Predictor**: Graph Neural Networks with temporal convolution layers
- **Matching Algorithm**: Deep reinforcement learning with multi-head attention
- **Uncertainty Quantification**: Monte Carlo dropout for prediction confidence intervals

### API Features
- **RESTful API**: Comprehensive endpoints for all operations
- **Real-time Updates**: WebSocket support for live tracking
- **Background Tasks**: Celery-based asynchronous processing
- **Market Analysis**: Advanced analytics and recommendations

## 🏗️ Architecture

```
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── core/                   # Core configuration and utilities
│   │   ├── config.py          # Application settings
│   │   ├── database.py        # Database configuration
│   │   ├── logging.py         # Logging setup
│   │   └── celery_app.py      # Celery configuration
│   ├── models/                 # Database models
│   │   ├── user.py            # User/Driver models
│   │   └── ride.py            # Ride models
│   ├── ml/                     # Machine Learning models
│   │   ├── demand_predictor.py # Demand prediction model
│   │   ├── supply_predictor.py # Supply prediction model
│   │   ├── matching_algorithm.py # Ride matching algorithm
│   │   └── models.py          # Model manager
│   └── api/v1/                # API endpoints
│       ├── users.py           # User management
│       ├── rides.py           # Ride operations
│       ├── drivers.py         # Driver operations
│       ├── matching.py        # ML matching endpoints
│       └── schemas.py         # Pydantic schemas
├── models/                     # Trained model files
├── logs/                       # Application logs
├── docker-compose.yml          # Docker services
├── Dockerfile                  # Application container
└── requirements.txt           # Python dependencies
```

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.11
- **ML Framework**: PyTorch, NumPy, Scikit-learn
- **Database**: PostgreSQL with SQLAlchemy (async)
- **Cache/Queue**: Redis, Celery
- **Containerization**: Docker, Docker Compose
- **API Documentation**: Swagger/OpenAPI

## 📊 Machine Learning Models

### 1. Demand Prediction Model
- **Architecture**: Transformer with spatial attention
- **Input Features**: Location, time, weather, historical data
- **Output**: Demand level with confidence intervals
- **Key Features**:
  - Positional encoding for temporal patterns
  - Spatial attention for geographic relationships
  - Uncertainty quantification via Monte Carlo dropout

### 2. Supply Prediction Model
- **Architecture**: Graph Neural Networks
- **Input Features**: Driver locations, availability, regional data
- **Output**: Supply levels across geographic regions
- **Key Features**:
  - Graph convolution for spatial relationships
  - Temporal convolution for time series patterns
  - Multi-scale attention mechanisms

### 3. Matching Algorithm
- **Architecture**: Deep RL with attention mechanisms
- **Input Features**: Rider requests, driver availability
- **Output**: Optimal rider-driver assignments
- **Key Features**:
  - Multi-head attention for feature interaction
  - Optimal transport for fair matching
  - Business constraint integration

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ (for local development)

### Using Docker (Recommended)

1. **Clone and navigate to the project**:
```bash
git clone <repository-url>
cd uber-clone-ml
```

2. **Start all services**:
```bash
docker-compose up -d
```

3. **Access the application**:
- API Documentation: http://localhost:8000/docs
- Application: http://localhost:8000
- Flower (Celery monitoring): http://localhost:5555

### Local Development

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Start the database**:
```bash
docker-compose up -d db redis
```

4. **Run the application**:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 📚 API Documentation

### Core Endpoints

#### Users
- `POST /api/v1/users/` - Create user (rider/driver)
- `GET /api/v1/users/{user_id}` - Get user details
- `PUT /api/v1/users/{user_id}` - Update user information

#### Rides
- `POST /api/v1/rides/request` - Request a ride
- `GET /api/v1/rides/{ride_id}` - Get ride details
- `PUT /api/v1/rides/{ride_id}` - Update ride status
- `POST /api/v1/rides/{ride_id}/cancel` - Cancel ride

#### Drivers
- `GET /api/v1/drivers/available` - Get available drivers
- `PUT /api/v1/drivers/{driver_id}/location` - Update driver location
- `POST /api/v1/drivers/{driver_id}/accept-ride/{ride_id}` - Accept ride

#### ML Matching
- `POST /api/v1/matching/match-rides` - ML-powered ride matching
- `POST /api/v1/matching/predict-demand` - Predict demand
- `POST /api/v1/matching/predict-supply` - Predict supply
- `GET /api/v1/matching/market-analysis` - Market analysis

### Example API Usage

#### Request a Ride
```bash
curl -X POST "http://localhost:8000/api/v1/rides/request?rider_id=1" \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_latitude": 40.7128,
    "pickup_longitude": -74.0060,
    "destination_latitude": 40.7589,
    "destination_longitude": -73.9851,
    "vehicle_type": "economy"
  }'
```

#### Predict Demand
```bash
curl -X POST "http://localhost:8000/api/v1/matching/predict-demand" \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 40.7128,
    "longitude": -74.0060
  }'
```

## 🔧 Configuration

### Environment Variables

```env
# Database
DATABASE_URL=postgresql://uber_user:uber_pass@localhost:5432/uber_db

# Redis
REDIS_URL=redis://localhost:6379/0

# ML Models
ML_MODEL_PATH=./models
DEVICE=cpu  # or cuda

# Pricing
BASE_FARE=2.50
COST_PER_KM=1.20
COST_PER_MINUTE=0.25
SURGE_MULTIPLIER_MAX=3.0

# Security
SECRET_KEY=your-secret-key-here
```

## 🧪 Testing

### Run Tests
```bash
pytest tests/ -v
```

### Load Testing
```bash
# Install locust
pip install locust

# Run load tests
locust -f tests/load_test.py --host=http://localhost:8000
```

## 📈 Monitoring and Analytics

### Celery Tasks Monitoring
Access Flower dashboard at http://localhost:5555 to monitor:
- Task execution status
- Worker performance
- Queue statistics

### Application Logs
```bash
# View application logs
docker-compose logs -f app

# View specific service logs
docker-compose logs -f celery_worker
```

### Database Monitoring
```bash
# Connect to PostgreSQL
docker-compose exec db psql -U uber_user -d uber_db

# View ride statistics
SELECT status, COUNT(*) FROM rides GROUP BY status;
```

## 🔄 Background Tasks

The application includes several automated background tasks:

1. **Auto-matching**: Automatically matches pending rides with available drivers every 30 seconds
2. **Prediction Updates**: Updates demand/supply predictions every 5 minutes
3. **Data Cleanup**: Removes old completed rides hourly

## 🚀 Deployment

### Production Deployment

1. **Update configuration for production**:
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  app:
    environment:
      - DEBUG=false
      - SECRET_KEY=production-secret-key
    deploy:
      replicas: 3
```

2. **Deploy with production settings**:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Scaling

Scale individual services based on load:
```bash
# Scale API servers
docker-compose up -d --scale app=3

# Scale Celery workers
docker-compose up -d --scale celery_worker=5
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on recent research in spatio-temporal forecasting
- Inspired by Uber's matching algorithms
- Uses state-of-the-art ML techniques from top-tier conferences

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the logs for troubleshooting

---

**Built with ❤️ using modern ML and software engineering practices**
