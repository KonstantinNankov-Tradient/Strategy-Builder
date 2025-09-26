## 🚀 Strategy Builder Development Roadmap

### Executive Summary

**Project Goal**: Build a production-ready trading system backend that processes multi-timeframe JSON strategy definitions from a visual frontend builder, focusing on 5 core indicators with comprehensive visualization and backtesting capabilities.

**Architecture**:
- **Frontend**: Visual building blocks connected by edges organized in timeframe hierarchy (not in scope)
- **Backend**: JSON-first multi-timeframe processing system that converts visual strategies to executable trading logic
- **Interface**: JSON strategy definitions containing symbol selection, timeframe blocks, indicator sequences, risk management, and entry conditions
- **Execution**: Multi-timeframe sequential processing - complete each timeframe's indicators before moving to next timeframe

**Multi-Timeframe Execution Model**:
1. **Symbol Selection**: Choose markets/symbols from starting block
2. **Timeframe Sequential Processing**: Complete H4 indicators → switch to H1 from current candle → complete H1 indicators
3. **Independent Timeframes**: Each timeframe's indicators work independently within their sequence
4. **Timeframe Switching**: When timeframe completes, convert candle index to next timeframe and continue from that position
5. **Final Processing**: Risk Management → Entry after all timeframes complete

**Core Indicators**:
1. **Liquidity Grab Detection** - Identifies liquidity sweeps for entry setups
2. **Change of Character (CHoCH)** - Detects trend direction changes
3. **Break of Structure (BOS)** - Validates structural market breaks
4. **Fair Value Gap (FVG)** - Locates precise entry zones
5. **Order Block Detection** - Identifies institutional trading levels

**Data Format**: Custom datasets in standard OHLCV format (datetime, open, high, low, close, volume)

**Input Format**: Multi-timeframe JSON strategy definitions containing symbol selection, timeframe blocks with indicator sequences, risk management parameters, and entry conditions

**Testing Philosophy**: Universal testing across multiple multi-timeframe JSON strategy configurations and market conditions without hardcoding specific scenarios. Continuous validation throughout development using JSON-based test cases with cross-timeframe synchronization testing.

**Visualization Requirements**: Individual validation plots for each indicator during development, plus comprehensive backtesting visualization showing only used indicators with all executed trades.

---

## Sprint 1: Foundation & Multi-Timeframe Data Infrastructure (Week 1)
**Sprint Goal**: Build multi-timeframe data infrastructure, basic JSON processing, and first two indicators with single-timeframe validation

### User Stories
- **As a backend system**, I need to load and synchronize multiple timeframes of market data
- **As a developer**, I need basic JSON strategy parsing for timeframe hierarchy
- **As a developer**, I need timeframe index conversion utilities for cross-timeframe switching
- **As a developer**, I need Liquidity Grab and CHoCH indicators working on single timeframes with validation plots
- **As Tradient**, I want to see foundation for multi-timeframe processing established

### Definition of Done
- [+] Multi-timeframe dataset loading with synchronization (H4, H1, etc.)
- [+] Timeframe index conversion system (H4 candle 100 → H1 candle 400)
- [+] Basic JSON strategy parser for timeframe hierarchy (structure validation)
- [+] Liquidity Grab indicator fully implemented with validation plot
- [+] CHoCH indicator fully implemented with validation plot
- [+] Basic state machine foundation (single timeframe sequential execution)
- [+] Multi-timeframe data testing framework
- [+] JSON schema definition for timeframe hierarchy structure

### Sprint Deliverables
- Multi-timeframe data loading system for custom OHLCV datasets with synchronization
- Timeframe index conversion utilities for cross-timeframe switching
- Basic JSON strategy parser for timeframe hierarchy structure
- Foundation state machine for single timeframe sequential execution
- Liquidity Grab Detection indicator (single timeframe)
- Change of Character (CHoCH) indicator (single timeframe)
- Visualization system with validation plots for both indicators
- Multi-timeframe data testing framework
- JSON schema definition for timeframe hierarchy

### Business Impact
- Establish multi-timeframe data infrastructure foundation
- Create JSON parsing foundation for timeframe hierarchy
- Validate first two indicators with individual timeframe capability
- Establish timeframe conversion utilities for cross-timeframe coordination
- Foundation for multi-timeframe state machine in Sprint 2

---

## Sprint 2: Cross-Timeframe State Machine & Remaining Indicators (Week 2)
**Sprint Goal**: Implement cross-timeframe state machine, complete remaining indicators, and enable multi-timeframe JSON execution

### User Stories
- **As a backend system**, I need cross-timeframe state machine to handle timeframe transitions
- **As a developer**, I need BOS, FVG, and Order Block indicators working on individual timeframes
- **As a developer**, I need the state machine to coordinate indicator sequences across timeframes
- **As a developer**, I need JSON strategies to execute across multiple timeframes sequentially
- **As a developer**, I need validation for cross-timeframe execution flows

### Definition of Done
- [ ] Enhanced state machine handles timeframe transitions (H4 complete → H1 start)
- [+] BOS indicator implemented with validation plot
- [+] FVG indicator implemented with validation plot
- [+] Order Block indicator implemented with validation plot
- [ ] Multi-timeframe JSON strategy execution (timeframe sequence processing)
- [ ] Cross-timeframe state transitions work correctly
- [ ] All 5 indicators working in multi-timeframe JSON-defined sequences
- [ ] Multi-timeframe execution testing framework
- [ ] JSON strategy validation for cross-timeframe logic

### Sprint Deliverables
- Enhanced state machine with cross-timeframe transition support
- Break of Structure (BOS) indicator
- Fair Value Gap (FVG) indicator
- Order Block Detection indicator
- Multi-timeframe JSON strategy execution engine
- Cross-timeframe coordination and validation system
- Validation plots for all 5 indicators
- Multi-timeframe testing framework

### Business Impact
- Complete cross-timeframe execution capability
- All 5 core indicators operational with multi-timeframe support
- Enable complex multi-timeframe strategies from JSON definitions
- Foundation for full backtesting with risk management in Sprint 3

---

## Sprint 3: Risk Management & Complete Backtesting System (Week 3)
**Sprint Goal**: Implement JSON-configurable risk management, complete backtesting system, and comprehensive trade visualization

### User Stories
- **As a backend system**, I need JSON-configurable risk management for multi-timeframe strategies
- **As a developer**, I need complete backtesting engine for multi-timeframe JSON strategy execution
- **As a developer**, I need comprehensive visualization showing multi-timeframe strategy performance
- **As Tradient**, I want to see complete multi-timeframe strategies executing trades with proper risk management

### Definition of Done
- [ ] JSON-configurable risk management (stop loss, take profit, position sizing)
- [ ] Complete multi-timeframe JSON strategies execute through backtesting engine
- [ ] Trades executed based on multi-timeframe JSON-defined indicator signals
- [ ] Risk management integrated with cross-timeframe execution
- [ ] Comprehensive backtesting plots display multi-timeframe execution flow
- [ ] Multiple complete JSON strategies tested with custom datasets
- [ ] Multi-timeframe trade execution logic fully validated
- [ ] Backtesting results returned in structured format for frontend consumption

### Sprint Deliverables
- JSON-configurable risk management system (stop loss, take profit, position sizing)
- Complete multi-timeframe backtesting execution engine
- Comprehensive backtesting visualization system showing timeframe transitions
- Trade execution integrated with risk management from JSON configuration
- Multi-timeframe strategy testing with complete trade lifecycle
- Production-ready plotting system with structured JSON output
- Complete JSON strategy results formatting for frontend integration

### Business Impact
- First complete multi-timeframe backtests with JSON strategies and actual trades
- Visual proof of multi-timeframe JSON strategy effectiveness
- Complete risk management integration with multi-timeframe execution
- Full system validation ready for production optimization in Sprint 4

---

## Sprint 4: State Testing, Optimization & Statistics (Week 4)
**Sprint Goal**: Extensive JSON strategy testing, performance optimization, and statistics framework

### User Stories
- **As a backend system**, I need thorough JSON strategy processing testing for reliability
- **As a developer**, I need performance optimization for production-scale JSON strategy processing
- **As a developer**, I need comprehensive backtesting statistics for JSON strategy evaluation
- **As Tradient**, I need a robust, optimized JSON-processing backend ready for production frontend integration

### Definition of Done
- [ ] JSON strategy parser tested with complex multi-indicator sequences and edge cases
- [ ] JSON validation and error scenarios thoroughly tested
- [ ] State machine tested with all possible JSON strategy configurations
- [ ] Performance optimized for speed and memory usage with large JSON strategies
- [ ] Parallel JSON strategy testing implemented
- [ ] Backtesting statistics framework operational (metrics TBD)
- [ ] JSON response formatting optimized for frontend consumption
- [ ] Production deployment ready for frontend integration

### Sprint Deliverables
- Comprehensive JSON strategy processing test suite
- Performance optimization for JSON parsing and execution (memory, speed, parallelization)
- JSON validation edge case handling and error recovery mechanisms
- Backtesting statistics framework with JSON output formatting
- Production deployment package for JSON-processing backend
- Complete documentation of JSON strategy behavior and API

### Business Impact
- Ensure JSON processing system reliability through extensive testing
- Achieve production-level performance for JSON strategy execution
- Enable large-scale JSON strategy testing and validation
- Deliver robust, optimized JSON-first backend ready for frontend integration

---

## 📊 Visualization Requirements

### Individual Indicator Validation Plots
Each of the 5 core indicators requires its own validation plot during development:
- **Purpose**: Verify correct indicator behavior and signal generation from JSON configuration
- **Content**: Price chart with indicator signals clearly marked
- **Testing**: Visual confirmation that JSON-configured indicators detect intended patterns
- **Format**: Clear, labeled plots showing indicator activation points
- **Output**: Both visual files and structured data for potential frontend integration

### Comprehensive Backtesting Visualization
The backtesting system produces a master plot containing:
- **Price Chart**: Displayed on the lowest timeframe used in the JSON strategy
- **Indicators**: Only the indicators actually used in the specific JSON strategy
- **Trade Markers**: All executed trades with clear entry/exit points
- **Trade Logic**: Visual connection between JSON-defined indicators and trade decisions
- **Layout**: Organized display with price as main chart, indicators as overlays or subplots
- **Output Format**: Both visualization files and structured JSON data for frontend consumption

### Backtesting Statistics
- Comprehensive framework for calculating backtesting statistics from JSON strategies
- Specific metrics to be determined based on requirements
- Flexible architecture to add new metrics as needed
- Clear presentation of JSON strategy performance
- **Output Format**: Structured JSON format suitable for frontend display

---

## 📁 Data Requirements

### Custom Dataset Format
- **Structure**: datetime, open, high, low, close, volume
- **Source**: Pre-existing custom datasets (no additional data handling needed)
- **Integration**: Direct loading into JSON-driven backtesting system
- **Testing**: All development and testing uses these custom datasets with JSON strategy configurations

### Multi-Timeframe JSON Strategy Input Format
- **Structure**: Symbol selection block, timeframe blocks with indicator sequences, risk management block, entry block
- **Hierarchy**: Symbol → Timeframe 1 (indicators) → Timeframe 2 (indicators) → Risk Management → Entry
- **Source**: Generated by frontend visual strategy builder (not in backend scope)
- **Integration**: Primary input method for all multi-timeframe strategy execution
- **Validation**: Schema validation and error handling for timeframe hierarchy and cross-timeframe logic

### JSON Example Structure
```json
{
  "strategy": {
    "symbol_block": {"symbol": "EURUSD"},
    "timeframe_blocks": [
      {
        "timeframe": "H4",
        "sequence": 1,
        "indicators": [
          {"type": "liquidity_grab_detector", "sequence": 1, "config": {...}},
          {"type": "choch_detector", "sequence": 2, "config": {...}}
        ]
      },
      {
        "timeframe": "H1",
        "sequence": 2,
        "indicators": [
          {"type": "order_block_detector", "sequence": 1, "config": {...}}
        ]
      }
    ],
    "risk_management_block": {"stop_loss_pips": 25, "take_profit_pips": 50},
    "entry_block": {"conditions": "all_timeframes_complete"}
  }
}
```

---

## Success Metrics & Risk Mitigation

### Sprint Success Metrics
- **Sprint 1**: Multi-timeframe data infrastructure operational, basic JSON parsing works, LG and CHoCH indicators validated, timeframe conversion utilities working
- **Sprint 2**: Cross-timeframe state machine operational, all 5 indicators working, multi-timeframe JSON execution proven, timeframe transitions validated
- **Sprint 3**: Complete risk management integrated, full multi-timeframe backtests with trades, comprehensive visualization, structured JSON output for frontend
- **Sprint 4**: Multi-timeframe system thoroughly tested, performance optimized, statistics framework operational, production-ready backend

### Project Success Criteria
- ✅ All 5 indicators implemented and validated with multi-timeframe JSON configuration and custom datasets
- ✅ Multi-timeframe JSON strategy parser converts visual timeframe blocks to executable cross-timeframe strategies
- ✅ Individual validation plots confirm correct indicator behavior from multi-timeframe JSON input
- ✅ Comprehensive backtesting visualization shows multi-timeframe JSON strategy execution clearly with timeframe transitions
- ✅ Cross-timeframe coordination works seamlessly with JSON timeframe hierarchy definitions
- ✅ Timeframe index synchronization and switching proven reliable
- ✅ Production-ready multi-timeframe JSON-first backend with complete testing using custom datasets

---

## 📝 Development Guidelines

### File Execution Rules
- **ALWAYS use absolute paths when running Python files**: `/home/lordargus/Tradient/strategy_builder/run_tests.py`
- **Never assume current working directory**: Always verify with `pwd` first
- **The working directory is**: `/home/lordargus/Tradient/strategy_builder`

### Testing Requirements
- **Every class and function MUST have corresponding tests** in the `tests/` directory
- Maintain high test coverage for reliability
- **Run tests using**: `python3 /home/lordargus/Tradient/strategy_builder/run_tests.py`

### Code Development Process
1. **Before writing code to a new file**:
   - First write and explain the code in the CLI`
   - Demonstrate functionality with examples
   - Validate the approach before file creation
2. **Single file development**:
   - Create and complete ONE file at a time
   - Ensure each file is fully tested before moving to the next
   - Maintain focus and completeness

### Architecture Notes
- When developing the Strategy Builder, reference other Tradient repositories for patterns and conventions but make architectural decisions independently based on the specific requirements of this system.