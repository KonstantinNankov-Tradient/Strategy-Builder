# Strategy Builder API Documentation

Complete API reference for the Strategy Builder backend. This document contains all endpoint specifications, request/response schemas, authentication details, and error handling.

## Base URL
```
http://localhost:8000
```

## Authentication

The API uses JWT (JSON Web Token) based authentication.

### Authentication Flow

1. Register a new user (`POST /api/auth/register`) or login (`POST /api/auth/login`)
2. Receive access token (expires in 15 minutes) and refresh token (expires in 7 days)
3. Include access token in `Authorization: Bearer <token>` header for protected endpoints
4. Use refresh token (`POST /api/auth/refresh`) to get new access tokens when expired

### Token Format
```
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

---

## Authentication Endpoints

### Register User
**POST** `/api/auth/register`

Create a new user account.

**Request Body:**
```json
{
  "email": "user@example.com",
  "username": "johndoe",
  "password": "SecurePassword123!"
}
```

**Password Requirements:**
- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one number

**Response (201 Created):**
```json
{
  "user": {
    "id": 1,
    "email": "user@example.com",
    "username": "johndoe",
    "is_active": true,
    "created_at": "2024-01-15T10:00:00Z"
  },
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer"
}
```

**Error Responses:**
- `400 Bad Request`: Email/username already exists or weak password
- `422 Unprocessable Entity`: Invalid input format

---

### Login User
**POST** `/api/auth/login`

Authenticate with username/email and password.

**Request Body:**
```json
{
  "username": "johndoe",  // Can be username or email
  "password": "SecurePassword123!"
}
```

**Response (200 OK):**
```json
{
  "user": {
    "id": 1,
    "email": "user@example.com",
    "username": "johndoe",
    "is_active": true,
    "created_at": "2024-01-15T10:00:00Z"
  },
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer"
}
```

**Error Responses:**
- `401 Unauthorized`: Invalid credentials

---

### Refresh Token
**POST** `/api/auth/refresh`

Get a new access token using refresh token.

**Request Body:**
```json
{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer"
}
```

**Error Responses:**
- `401 Unauthorized`: Invalid or expired refresh token

---

### Logout User
**POST** `/api/auth/logout`

Invalidate user session and refresh token.

**Request Body:**
```json
{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

**Response (200 OK):**
```json
{
  "message": "Successfully logged out"
}
```

---

### Get Current User
**GET** `/api/auth/me`

Get current authenticated user information.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```json
{
  "id": 1,
  "email": "user@example.com",
  "username": "johndoe",
  "is_active": true,
  "created_at": "2024-01-15T10:00:00Z"
}
```

**Error Responses:**
- `401 Unauthorized`: Missing or invalid access token

---

## Strategy Endpoints (Protected)

All strategy endpoints require authentication via Bearer token.

### Process Strategy
**POST** `/api/strategies`

Execute a multi-timeframe strategy configuration.

**Headers:**
```
Authorization: Bearer <access_token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "strategy": {
    "name": "My Trading Strategy",
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

**Response (200 OK):**
```json
{
  "status": "success",
  "strategy_name": "My Trading Strategy",
  "execution_time": "2024-01-15T14:30:00Z",
  "results": {
    "total_trades": 15,
    "indicators_used": ["liquidity_grab_detector"],
    "performance_summary": {
      "total_trades": 15,
      "winning_trades": 9,
      "losing_trades": 6,
      "win_rate": 60.0,
      "total_return": 12.5,
      "max_drawdown": -3.2
    },
    "raw_results": {
      // Complete execution results
    }
  }
}
```

---

### Get User Strategies
**GET** `/api/strategies/`

Get all strategies for the current user.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```json
[
  {
    "id": 1,
    "name": "Strategy 1",
    "description": "Strategy created by johndoe",
    "created_at": "2024-01-15T10:00:00Z",
    "updated_at": "2024-01-15T10:00:00Z",
    "configuration": {
      // Strategy JSON configuration
    }
  }
]
```

---

### Get Specific Strategy
**GET** `/api/strategies/{strategy_id}`

Get a specific strategy by ID.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```json
{
  "id": 1,
  "name": "Strategy 1",
  "description": "Strategy created by johndoe",
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:00:00Z",
  "configuration": {
    // Strategy JSON configuration
  }
}
```

**Error Responses:**
- `404 Not Found`: Strategy not found or doesn't belong to user

---

### Update Strategy
**PUT** `/api/strategies/{strategy_id}`

Update a specific strategy.

**Headers:**
```
Authorization: Bearer <access_token>
Content-Type: application/json
```

**Request Body:** Same as Process Strategy

**Response (200 OK):** Same as Get Specific Strategy

---

### Delete Strategy
**DELETE** `/api/strategies/{strategy_id}`

Delete a specific strategy (soft delete).

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```json
{
  "message": "Strategy deleted successfully"
}
```

---

## Public Endpoints

### Health Check
**GET** `/api/health`

Check API health status.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "components": {
    "json_processor": "ready",
    "strategy_runner": "ready",
    "indicators": "ready"
  },
  "timestamp": "2024-01-15T14:30:00Z"
}
```

---

### Example Strategy
**GET** `/api/strategies/example`

Get an example strategy configuration for testing.

**Response (200 OK):**
```json
{
  "strategy": {
    // Example strategy configuration
  }
}
```

---

### Building Blocks Schema
**GET** `/api/building-blocks`

Get the complete schema of available building blocks (timeframes and indicators) for the visual strategy builder.

**Response (200 OK):**
```json
[
  {
    "id": "timeframe_daily",
    "type": "timeframe",
    "display_name": "Daily Timeframe",
    "timeframe_value": "1d",
    "premium": false,
    "description": "Analyze indicators on daily candles",
    "config": {}
  },
  {
    "id": "bos_detector",
    "type": "indicator",
    "display_name": "Break of Structure (BOS)",
    "description": "Detects trend continuations when price breaks swing highs/lows",
    "config": {
      "direction": {
        "type": "select",
        "display_name": "Direction",
        "description": "Select the BOS direction to detect",
        "required": true,
        "default": "None",
        "options": [
          { "value": "bullish", "label": "Bullish" },
          { "value": "bearish", "label": "Bearish" }
        ]
      }
    }
  }
  // ... more building blocks
]
```

**Response Fields:**
- `id`: Unique identifier for the building block
- `type`: Either "timeframe" or "indicator"
- `display_name`: Human-readable name for UI display
- `description`: Description of the building block functionality
- `premium`: Boolean indicating if premium subscription is required
- `config`: Configuration schema for the building block
  - Each config field contains: `type`, `display_name`, `description`, `required`, `default`, and validation rules

**Error Responses:**
- `404 Not Found`: Building blocks schema file not found
- `500 Internal Server Error`: Invalid JSON format or file read error

---

## Error Handling

All endpoints return structured error responses:

```json
{
  "error": "Error description",
  "details": "Detailed error information",
  "validation_errors": ["field1: error", "field2: error"]
}
```

**Common HTTP Status Codes:**
- `200 OK`: Request successful
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Authentication required or failed
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation errors
- `500 Internal Server Error`: Server error

---

## Rate Limiting

- Login attempts: 10 requests per minute per IP
- General API calls: No limit currently implemented

---

## Security Features

- JWT tokens with configurable expiration
- Password hashing with bcrypt
- CORS protection
- Input validation and sanitization
- User session management
- Secure password requirements

---

## JSON Strategy Schema

### Complete Strategy Definition

```json
{
  "strategy": {
    "name": "string",                    // Strategy name
    "version": "string",                 // Strategy version (e.g., "1.0")
    "description": "string (optional)",  // Strategy description
    "symbol_block": {
      "symbol": "string"                 // Trading symbol (e.g., "EURUSD")
    },
    "timeframe_blocks": [
      {
        "timeframe": "string",           // Timeframe (e.g., "H4", "H1", "M15")
        "sequence": number,              // Processing order (1, 2, 3...)
        "indicators": [
          {
            "type": "string",            // Indicator type (see below)
            "sequence": number,          // Indicator order within timeframe
            "config": {
              // Indicator-specific configuration
            }
          }
        ]
      }
    ],
    "risk_management_block": {
      "stop_loss_pips": number,          // Stop loss in pips
      "take_profit_pips": number,        // Take profit in pips
      "position_size_percent": number    // Position size as % of capital
    },
    "entry_block": {
      "conditions": "string"             // Entry conditions
    }
  }
}
```

### Available Indicator Types

#### 1. Liquidity Grab Detector
```json
{
  "type": "liquidity_grab_detector",
  "config": {
    "swing_lookback": 10,              // Swing point lookback period
    "liquidity_threshold": 0.001       // Threshold for liquidity grab detection
  }
}
```

#### 2. Change of Character (CHoCH)
```json
{
  "type": "choch_detector",
  "config": {
    "direction": "bullish|bearish|None",  // CHoCH direction to detect
    "swing_lookback": 5                   // Swing point lookback period
  }
}
```

#### 3. Break of Structure (BOS)
```json
{
  "type": "bos_detector",
  "config": {
    "direction": "bullish|bearish|None",  // BOS direction to detect
    "swing_lookback": 5                   // Swing point lookback period
  }
}
```

#### 4. Fair Value Gap (FVG)
```json
{
  "type": "fvg_detector",
  "config": {
    "direction": "bullish|bearish|None",  // FVG direction to detect
    "min_gap_size": 0.0005                // Minimum gap size threshold
  }
}
```

#### 5. Order Block
```json
{
  "type": "order_block_detector",
  "config": {
    "direction": "bullish|bearish|None",  // Order block direction
    "lookback": 20                        // Lookback period for detection
  }
}
```

### Supported Timeframes
- `M1`, `M5`, `M15`, `M30` - Minutes
- `H1`, `H4`, `H12` - Hours
- `D1`, `W1`, `MN` - Daily, Weekly, Monthly

---

## Frontend Integration Guide

### Basic Request Example (JavaScript)

```javascript
// Register user
const registerResponse = await fetch('http://localhost:8000/api/auth/register', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    email: 'user@example.com',
    username: 'username',
    password: 'SecurePass123!'
  })
});
const { access_token, refresh_token } = await registerResponse.json();

// Create strategy (authenticated)
const strategyResponse = await fetch('http://localhost:8000/api/strategies', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${access_token}`
  },
  body: JSON.stringify({ strategy: { /* strategy config */ } })
});
```

### Token Refresh Pattern

```javascript
async function fetchWithAuth(url, options = {}) {
  let token = getAccessToken();

  let response = await fetch(url, {
    ...options,
    headers: {
      ...options.headers,
      'Authorization': `Bearer ${token}`
    }
  });

  // Auto-refresh on 401
  if (response.status === 401) {
    const newToken = await refreshAccessToken();
    response = await fetch(url, {
      ...options,
      headers: {
        ...options.headers,
        'Authorization': `Bearer ${newToken}`
      }
    });
  }

  return response;
}
```

### Secure Token Storage

**Recommended for production:**
- Use httpOnly cookies for tokens
- Implement CSRF protection
- Use secure flag for HTTPS only

**Development/Testing:**
- Store in memory or sessionStorage
- Clear tokens on logout
- Never store in localStorage for production

---

## WebSocket Support (Future)

Real-time updates for strategy execution and backtesting progress will be added in a future release via WebSocket connections.

---

## Versioning

Current API version: **v1**

All endpoints are prefixed with `/api/` and the API follows semantic versioning principles. Breaking changes will result in a new API version.