# Strategy Builder API Documentation

## Authentication System

The Strategy Builder API uses JWT (JSON Web Token) based authentication with PostgreSQL/TimescaleDB for user management.

### Base URL
```
http://localhost:8000
```

### Authentication Flow

1. Register a new user or login with existing credentials
2. Receive access token (15 minutes) and refresh token (7 days)
3. Include access token in `Authorization: Bearer <token>` header for protected endpoints
4. Use refresh token to get new access tokens when needed

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

## Frontend Integration

### Token Storage
Store tokens securely:
- Use httpOnly cookies for production
- Or secure localStorage with proper cleanup

### Request Headers
Include authentication header in all protected requests:
```javascript
headers: {
  'Authorization': `Bearer ${accessToken}`,
  'Content-Type': 'application/json'
}
```

### Token Refresh Flow
```javascript
// Automatic token refresh on 401 responses
if (response.status === 401) {
  const newToken = await refreshToken();
  // Retry original request with new token
}
```

### Error Handling
Handle authentication errors gracefully:
```javascript
if (response.status === 401) {
  // Redirect to login page
  redirectToLogin();
}
```