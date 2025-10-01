# Strategy Builder API Server

Backend API server for the multi-timeframe trading strategy builder system. This server provides HTTP endpoints with full authentication for frontend integration, strategy management, and backtesting capabilities.

## 🚀 Quick Start for Frontend Developers

### 1. Start the Authentication Server

```bash
# Start the authentication-focused server
python3 auth_server.py
```

The server will be available at:
- **API Base URL**: `http://localhost:8000`
- **API Documentation**: `http://localhost:8000/docs` (Interactive Swagger UI)
- **Health Check**: `http://localhost:8000/api/health`

### 2. Test the Server

Access the interactive API documentation at `http://localhost:8000/docs` to test all endpoints, or use the provided Postman collection in `POSTMAN_COLLECTION_STRUCTURE.md`.

## 📋 API Endpoints

### 🌐 Public Endpoints

#### Root Health Check
```
GET /
```
Basic server status check.

**Response:**
```json
{
  "status": "online",
  "service": "Strategy Builder Authentication API",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

#### Detailed Health Check
```
GET /api/health
```
Comprehensive system health check.

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "authentication": "ready",
    "database": "ready",
    "json_processor": "ready"
  },
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

#### Get Example Strategy
```
GET /api/strategies/example
```
Returns a complete example strategy configuration for frontend testing.

**Response:**
```json
{
  "strategy": {
    "name": "Example Multi-Timeframe Strategy",
    "version": "1.0",
    "symbol_block": {
      "symbol": "EURUSD"
    },
    "timeframe_blocks": [
      {
        "timeframe": "H4",
        "sequence": 1,
        "indicators": [
          {
            "type": "liquidity_grab_detector",
            "sequence": 1,
            "config": {
              "swing_lookback": 10,
              "liquidity_threshold": 0.001
            }
          }
        ]
      }
    ],
    "risk_management_block": {
      "stop_loss_pips": 25,
      "take_profit_pips": 50,
      "position_size_percent": 2.0
    },
    "entry_block": {
      "conditions": "all_timeframes_complete"
    }
  }
}
```

### 🔐 Authentication Endpoints

#### Register User
```
POST /api/auth/register
Content-Type: application/json
```

**Request Body:**
```json
{
  "email": "user@example.com",
  "username": "username",
  "password": "SecurePass123!"
}
```

**Response:**
```json
{
  "user": {
    "id": 1,
    "email": "user@example.com",
    "username": "username",
    "is_active": true,
    "created_at": "2024-01-15T10:30:45.123456Z"
  },
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

#### Login User
```
POST /api/auth/login
Content-Type: application/json
```

**Request Body:**
```json
{
  "username": "username",
  "password": "SecurePass123!"
}
```

**Response:** Same as registration response.

#### Get Current User
```
GET /api/auth/me
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "id": 1,
  "email": "user@example.com",
  "username": "username",
  "is_active": true,
  "created_at": "2024-01-15T10:30:45.123456Z"
}
```

#### Refresh Token
```
POST /api/auth/refresh
Content-Type: application/json
```

**Request Body:**
```json
{
  "refresh_token": "{refresh_token}"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

#### Logout
```
POST /api/auth/logout
Content-Type: application/json
```

**Request Body:**
```json
{
  "refresh_token": "{refresh_token}"
}
```

**Response:**
```json
{
  "message": "Successfully logged out"
}
```

### 📊 Strategy Management (Protected Endpoints)

All strategy endpoints require authentication via `Authorization: Bearer {access_token}` header.

#### Create Strategy
```
POST /api/strategies
Authorization: Bearer {access_token}
Content-Type: application/json
```

Save a JSON strategy configuration to the database.

**Request Body:**
```json
{
  "strategy": {
    "name": "My Trading Strategy",
    "version": "1.0",
    "description": "My custom strategy description",
    "symbol_block": {
      "symbol": "EURUSD"
    },
    "timeframe_blocks": [
      {
        "timeframe": "H4",
        "sequence": 1,
        "indicators": [
          {
            "type": "liquidity_grab_detector",
            "sequence": 1,
            "config": {
              "swing_lookback": 10,
              "liquidity_threshold": 0.001
            }
          }
        ]
      }
    ],
    "risk_management_block": {
      "stop_loss_pips": 25,
      "take_profit_pips": 50,
      "position_size_percent": 2.0
    },
    "entry_block": {
      "conditions": "all_timeframes_complete"
    }
  }
}
```

**Response:**
```json
{
  "id": 1,
  "name": "My Trading Strategy",
  "description": "My custom strategy description",
  "created_at": "2024-01-15T10:30:45.123456Z",
  "status": "saved",
  "message": "Strategy saved successfully"
}
```

#### Get All User Strategies
```
GET /api/strategies/
Authorization: Bearer {access_token}
```

**Response:**
```json
[
  {
    "id": 1,
    "name": "My Trading Strategy",
    "description": "My custom strategy description",
    "created_at": "2024-01-15T10:30:45.123456Z",
    "updated_at": "2024-01-15T10:30:45.123456Z",
    "configuration": { /* Strategy JSON */ }
  }
]
```

#### Get Specific Strategy
```
GET /api/strategies/{strategy_id}
Authorization: Bearer {access_token}
```

**Response:** Same as individual strategy object above.

#### Update Strategy
```
PUT /api/strategies/{strategy_id}
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request Body:** Same as Create Strategy.

**Response:** Updated strategy object.

#### Delete Strategy
```
DELETE /api/strategies/{strategy_id}
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "message": "Strategy deleted successfully"
}
```

#### Backtest Strategy
```
POST /api/strategies/{strategy_id}/backtest
Authorization: Bearer {access_token}
Content-Type: application/json
```

Run a backtest for a saved strategy (currently returns mock results).

**Request Body (Optional):**
```json
{
  "start_date": "2024-01-01",
  "end_date": "2024-03-01",
  "initial_capital": 10000.0
}
```

**Response:**
```json
{
  "strategy_id": 1,
  "strategy_name": "My Trading Strategy",
  "execution_time": "2024-01-15T10:30:45.123456",
  "results": {
    "total_trades": 15,
    "indicators_used": ["liquidity_grab_detector", "choch_detector"],
    "performance_summary": {
      "total_trades": 15,
      "winning_trades": 9,
      "losing_trades": 6,
      "win_rate": 60.0,
      "total_return": 12.5,
      "max_drawdown": -3.2
    },
    "raw_results": {
      "status": "backtest_complete",
      "message": "Backtest executed for strategy: My Trading Strategy",
      "dates": {"start": "2024-01-01", "end": "2024-03-01"},
      "strategy_config": { /* Strategy configuration */ },
      "mock_trades": [
        {"entry": "2024-02-05", "exit": "2024-02-06", "pnl": 150},
        {"entry": "2024-02-10", "exit": "2024-02-12", "pnl": -75}
      ]
    }
  }
}
``` 

## 🌐 CORS Configuration

The server is configured to accept requests from common localhost ports:
- `http://localhost:3000` (React default)
- `http://localhost:3001`
- `http://localhost:5173` (Vite default)

If your frontend runs on a different port, modify the CORS settings in `config/settings.py`.

## 🧪 Testing

### Manual Testing with curl

```bash
# Health check
curl http://localhost:8000/api/health

# Register user
curl -X POST "http://localhost:8000/api/auth/register" \
     -H "Content-Type: application/json" \
     -d '{"email":"test@example.com","username":"testuser","password":"SecurePass123!"}'

# Login (save the access_token from response)
curl -X POST "http://localhost:8000/api/auth/login" \
     -H "Content-Type: application/json" \
     -d '{"username":"testuser","password":"SecurePass123!"}'

# Create strategy (replace YOUR_TOKEN with actual token)
curl -X POST "http://localhost:8000/api/strategies" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -d '{"strategy":{"name":"Test Strategy","symbol_block":{"symbol":"EURUSD"},"entry_block":{"conditions":"test"}}}'
```

### Postman Collection

Use the complete Postman collection documented in `POSTMAN_COLLECTION_STRUCTURE.md` for comprehensive API testing.

## 🔧 Troubleshooting

### Server Won't Start
1. Check if Python 3.10+ is installed: `python3 --version`
2. Check if port 8000 is available: `netstat -an | grep 8000`
3. Install dependencies manually: `pip install fastapi uvicorn sqlalchemy pydantic bcrypt python-jose`

### Authentication Errors
- Ensure you're using the correct token in the Authorization header
- Check if tokens have expired (access tokens expire after 15 minutes)
- Use the refresh endpoint to get new tokens
- Verify user credentials are correct

### CORS Errors
- Ensure your frontend URL is in the CORS allow list
- Check browser developer console for specific CORS error messages
- Verify the server is accessible: `curl http://localhost:8000/api/health`

### Strategy Processing Errors
- Check the API documentation at `http://localhost:8000/docs`
- Validate your JSON structure matches the expected schema
- Check server logs for detailed error messages

## 🏗️ Development

### Server Structure
```
api/
├── __init__.py           # Package initialization
├── server.py            # Full FastAPI application
├── models.py            # Pydantic data models
└── auth/
    ├── __init__.py
    ├── router.py        # Authentication routes
    ├── dependencies.py  # Auth dependencies
    ├── models.py        # Auth data models
    └── utils.py         # Auth utilities

auth_server.py           # Simplified auth-focused server
config/
└── settings.py          # Configuration settings
database/
├── connection.py        # Database connection
└── models.py           # SQLAlchemy models
requirements.txt         # Python dependencies
```

### Environment Variables

Create a `.env` file for configuration:

```bash
# Database Configuration
DB_HOST=localhost
DB_NAME=strategy_builder
DB_USER=postgres
DB_PASSWORD=your_password
DB_PORT=5432

# JWT Configuration
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001,http://localhost:5173
```

## 📋 Links
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Strategy Builder Architecture](./CLAUDE.md)
- [Interactive API Docs](http://localhost:8000/docs) (when server is running)
- [Postman Collection Guide](./POSTMAN_COLLECTION_STRUCTURE.md)