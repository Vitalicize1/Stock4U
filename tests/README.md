# 🧪 Tests Directory

This directory contains all test files for the Agentic Stock Predictor project.

## 📋 Available Tests

### Core Functionality Tests
- **`test_orchestrator_tools.py`** - Tests the orchestrator tools functionality
  - Ticker validation
  - Market status checking
  - Analysis parameter determination
  - Workflow state initialization
  - Error handling

- **`test_tavily.py`** - Tests Tavily search integration
  - LangGraph tutorial implementation
  - Tool binding and execution
  - Search functionality

- **`test_chatbot.py`** - Tests chatbot functionality
  - Natural language processing
  - Query interpretation
  - Response generation

- **`test_integrated_workflow.py`** - Tests complete workflow integration
  - End-to-end workflow execution
  - Orchestrator tools integration
  - Error handling scenarios

## 🚀 Running Tests

### Run All Tests
```bash
python tests/run_all_tests.py
```

### Run Specific Test
```bash
python tests/run_all_tests.py --test test_orchestrator_tools
```

### List Available Tests
```bash
python tests/run_all_tests.py --list
```

### Run Individual Test Files
```bash
# From project root
python tests/test_orchestrator_tools.py
python tests/test_tavily.py
python tests/test_chatbot.py
python tests/test_integrated_workflow.py
```

## 📊 Test Results

The test runner provides:
- ✅ **Success**: Test passed successfully
- ❌ **Failed**: Test encountered an error
- ⚠️ **Skipped**: Test was skipped (no main function)

## 🎯 Test Coverage

### Orchestrator Tools
- ✅ Ticker validation with yfinance
- ✅ Market status checking (holidays, weekends, hours)
- ✅ Analysis parameter determination
- ✅ Workflow state initialization
- ✅ Error handling and recovery

### LangGraph Integration
- ✅ Tool binding and execution
- ✅ State management
- ✅ Conditional edge routing
- ✅ Error path handling

### Workflow Integration
- ✅ Complete pipeline execution
- ✅ Agent coordination
- ✅ Data flow between stages
- ✅ Error propagation

## 🔧 Test Structure

Each test file follows this structure:
```python
#!/usr/bin/env python3
"""
Test description
"""

def test_functionality():
    """Test specific functionality"""
    # Test implementation
    
def main():
    """Main test function"""
    # Run all tests
    
if __name__ == "__main__":
    main()
```

## 📈 Adding New Tests

To add a new test:

1. Create a new test file in the `tests/` directory
2. Follow the naming convention: `test_<component>.py`
3. Include a `main()` function
4. Add the test to the `test_files` list in `run_all_tests.py`

Example:
```python
# tests/test_new_component.py
def test_new_functionality():
    """Test new component functionality"""
    # Test implementation
    
def main():
    """Main test function"""
    test_new_functionality()
    print("✅ New component tests passed!")

if __name__ == "__main__":
    main()
```

## 🐛 Debugging Tests

If a test fails:

1. **Check imports**: Ensure all required modules are available
2. **Check environment**: Verify API keys and dependencies
3. **Check logs**: Look for specific error messages
4. **Run individually**: Test the specific component in isolation

## 📝 Test Best Practices

- ✅ Test both success and error scenarios
- ✅ Include comprehensive logging
- ✅ Test edge cases and boundary conditions
- ✅ Verify data integrity and state management
- ✅ Test integration points between components 