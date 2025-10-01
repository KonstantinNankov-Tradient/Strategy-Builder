# Strategy Builder Backend

Multi-timeframe trading strategy builder backend system. Processes JSON strategy definitions from a visual frontend, executes backtests, and provides comprehensive authentication and strategy management APIs.

## 🎯 Project Overview

**Purpose**: Production-ready trading system backend that processes multi-timeframe JSON strategy definitions with 5 core indicators, comprehensive visualization, and backtesting capabilities.

**Key Features**:
- Multi-timeframe sequential processing (H4 → H1 → entry)
- 5 core indicators: Liquidity Grab, CHoCH, BOS, FVG, Order Block
- JWT-based authentication with PostgreSQL/TimescaleDB
- RESTful API for strategy management and backtesting
- JSON-first architecture for visual frontend integration

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- PostgreSQL with TimescaleDB extension
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Strategy-Builder
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
Create a `.env` file:
```bash
DB_HOST=localhost
DB_NAME=strategy_builder
DB_USER=postgres
DB_PASSWORD=your_password
DB_PORT=5432

JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

4. **Start the server**
```bash
python3 auth_server.py
```

Server will be available at:
- **API Base URL**: `http://localhost:8000`
- **API Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/api/health`

## 📖 API Documentation

Complete API documentation with all endpoints, request/response schemas, and authentication details is available in:
- **[API_DOCUMENTATION.md](./API_DOCUMENTATION.md)** - Full API reference
- **[Interactive Swagger UI](http://localhost:8000/docs)** - When server is running

Quick API overview:
- **Authentication**: `/api/auth/register`, `/api/auth/login`, `/api/auth/refresh`
- **Strategy Management**: `/api/strategies` (CRUD operations)
- **Backtesting**: `/api/strategies/{id}/backtest`
- **Health Check**: `/api/health`
- **Building Blocks**: `/api/building-blocks`

## 🧪 Running Tests

Execute the test suite:
```bash
python3 /home/lordargus/Tradient/strategy_builder/run_tests.py
```

Tests are located in `tests/` directory with coverage for:
- Multi-timeframe data loading and synchronization
- All 5 core indicators (Liquidity Grab, CHoCH, BOS, FVG, Order Block)
- JSON strategy parsing and validation
- Authentication and authorization flows
- Database operations

## 🏗️ Project Structure

```
Strategy-Builder/
├── api/                          # FastAPI application
│   ├── auth/                     # Authentication module
│   │   ├── router.py            # Auth endpoints
│   │   ├── dependencies.py      # JWT verification
│   │   └── utils.py             # Password hashing, token generation
│   ├── server.py                # Main FastAPI app
│   └── models.py                # Pydantic request/response models
├── config/
│   └── settings.py              # Environment configuration
├── database/
│   ├── connection.py            # Database connection & session
│   └── models.py                # SQLAlchemy ORM models
├── indicators/                   # Trading indicators
│   ├── liquidity_grab_detector.py
│   ├── choch_detector.py
│   ├── bos_detector.py
│   ├── fvg_detector.py
│   └── order_block_detector.py
├── data/                        # Market data & datasets
├── tests/                       # Test suite
├── auth_server.py               # Server entry point
├── requirements.txt             # Python dependencies
└── CLAUDE.md                    # Development roadmap
```

## 🔧 Development Guidelines

### File Execution Rules
- **Always use absolute paths** when running Python files
- **Verify working directory** with `pwd` before execution
- **Working directory**: `/home/lordargus/Tradient/strategy_builder`

### Code Development Process
1. Write and explain code in CLI before file creation
2. Create and complete **one file at a time**
3. **Every class/function must have tests** in `tests/` directory
4. Maintain high test coverage

### Architecture Principles
- **JSON-first**: All strategies defined in JSON format
- **Multi-timeframe sequential processing**: Complete each timeframe before moving to next
- **Indicator independence**: Each indicator works independently within its timeframe
- **State machine coordination**: Manages cross-timeframe transitions

## 🚀 Development Roadmap

See [CLAUDE.md](./CLAUDE.md) for detailed sprint planning:
- **Sprint 1**: ✅ Multi-timeframe data infrastructure, Liquidity Grab, CHoCH
- **Sprint 2**: ✅ BOS, FVG, Order Block indicators + cross-timeframe state machine
- **Sprint 3**: 🚧 Risk management & complete backtesting system
- **Sprint 4**: 📋 Testing, optimization & statistics framework

## 📚 Additional Resources

- [API Documentation](./API_DOCUMENTATION.md) - Complete endpoint reference
- [Postman Collection](./POSTMAN_COLLECTION_STRUCTURE.md) - API testing guide
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - Framework docs
- [Interactive Swagger UI](http://localhost:8000/docs) - Live API testing (when server running)

## 🔒 Security Features

- JWT-based authentication with access & refresh tokens
- Password hashing with bcrypt
- CORS protection for allowed origins
- Input validation and sanitization
- User session management
- Token expiration and refresh mechanisms

## 🤝 Contributing

1. Follow the development guidelines in CLAUDE.md
2. Write tests for all new features
3. Maintain code coverage above 80%
4. Use absolute paths for file operations
5. Document all API endpoints