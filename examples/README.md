# Chuk AI Planner Examples

This directory contains comprehensive examples demonstrating the features and capabilities of chuk-ai-planner, organized from basic to advanced concepts.

## 📚 Example Categories

### Core Concepts (01-04)
Basic graph fundamentals and routing mechanisms.

- **01_basic_graph.py** - Pydantic graph structure basics
  - Creating typed nodes and edges
  - Using InMemoryGraphStore
  - Querying the graph

- **02_conditional_routing.py** - Router types demonstration
  - Expression-based routing
  - LLM-based routing
  - Function-based routing

- **03_tool_execution.py** - ToolCall and TaskRun nodes
  - Creating tool call nodes
  - Linking tools to plan steps
  - Result tracking

- **04_function_routing.py** - FunctionRegistry usage
  - Decorator-based function registration
  - Custom routing logic
  - Multiple routing scenarios

### Plan DSL & Execution (05-09)
Using the original Plan API for plan creation and execution.

- **05_routing_executor.py** - Routing executor integration
  - Expression evaluation
  - Variable resolution
  - Route selection

- **06_plan_executor.py** - Complete Plan DSL usage
  - Fluent plan building API
  - Step dependencies
  - Session event handling

- **07_plan_with_tools.py** - PlanExecutor with tool integration
  - chuk_tool_processor integration
  - Tool registration and execution
  - InProcess and Subprocess strategies

- **08_plan_from_llm.py** - LLM-to-execution pipeline
  - JSON plan generation from natural language
  - Plan conversion to DSL
  - Dependency-based execution

- **09_simple_pipeline.py** - Simple data pipeline
  - Sequential step execution
  - Data aggregation
  - Report generation

### Plan Registry (10-11)
Plan storage, retrieval, and management.

- **10_plan_registry_basic.py** - Basic PlanRegistry usage
  - Storing plans to disk
  - Retrieving plans by ID
  - Simple plan structure

- **11_plan_registry_advanced.py** - Advanced registry features
  - Search by tags
  - Search by title
  - Plan persistence
  - Plan deletion

### UniversalPlan Basics (12-15)
Modern UniversalPlan and UniversalExecutor API.

- **12_universal_plan_intro.py** - Minimal UniversalPlan example
  - Basic plan creation
  - Simple executor usage
  - Getting started quickly

- **13_universal_plan_features.py** - UniversalPlan features
  - Variables and metadata
  - Step types (tool, function, router)
  - Plan structure

- **14_universal_executor_basic.py** - Basic executor usage
  - Variable substitution
  - Dependency handling
  - Result processing

- **15_universal_executor_advanced.py** - Advanced executor features
  - Complex dependency graphs
  - Error handling
  - Advanced variable patterns

### UniversalPlan with Tools (16-18)
Integrating tools with UniversalPlan.

- **16_universal_with_tools.py** - Tool processor integration
  - Registering tools
  - Tool execution strategies
  - Result handling

- **17_universal_plan_simple.py** - Simple tool execution
  - Basic tool integration
  - Step-by-step execution
  - Clean output

- **18_universal_plan_complete.py** - Complete execution output
  - Detailed logging
  - Full execution trace
  - Comprehensive results

### UniversalPlan with LLM (19-21)
AI-powered planning and execution.

- **19_universal_llm_executor.py** - LLM executor integration
  - Dynamic plan generation
  - LLM-based routing
  - Smart execution

- **20_universal_llm_plan.py** - LLM plan creation
  - Natural language to plan
  - Structured plan output
  - Plan validation

- **21_universal_main.py** - Main demo application
  - Complete LLM workflow
  - End-to-end example
  - Production patterns

### Advanced Use Cases (22-24)
Real-world applications and complex scenarios.

- **22_deep_researcher.py** - Deep research agent
  - Multi-step research workflow
  - Information gathering
  - Analysis and synthesis

- **23_deep_researcher_simple.py** - Simplified researcher
  - Streamlined research flow
  - Essential features only
  - Easier to understand

- **24_job_manager.py** - Job management system
  - Job scheduling
  - Status tracking
  - Queue management

## 🚀 Getting Started

### Run Basic Examples

```bash
# Start with the basics
python examples/01_basic_graph.py
python examples/02_conditional_routing.py
python examples/03_tool_execution.py
python examples/04_function_routing.py
```

### Run Plan Execution Examples

```bash
# Try the Plan DSL
python examples/05_routing_executor.py
python examples/06_plan_executor.py
python examples/07_plan_with_tools.py
```

### Run UniversalPlan Examples

```bash
# Explore the modern API
python examples/12_universal_plan_intro.py
python examples/13_universal_plan_features.py
python examples/14_universal_executor_basic.py
```

## 📖 Learning Path

**New to chuk-ai-planner?** Follow this recommended learning path:

1. **Start with Core Concepts (01-04)** - Understand the graph structure and routing
2. **Learn Plan DSL (05-09)** - Master plan creation and execution
3. **Explore Plan Registry (10-11)** - Learn plan storage and management
4. **Try UniversalPlan (12-18)** - Use the modern, feature-rich API
5. **Add LLM Integration (19-21)** - Incorporate AI-powered planning
6. **Build Real Applications (22-24)** - Apply concepts to real-world scenarios

## 🧪 Testing Examples

To test all working examples:

```bash
# Run all core examples
for i in {01..11}; do
    echo "Testing ${i}..."
    python examples/${i}_*.py
done
```

## 📝 Example Status

**Verified Working:** 23/24 examples (01-21, 23-24)
- ✅ Core concepts (01-04) - Fully tested and passing
- ✅ Plan DSL & Execution (05-09) - Fully tested and passing
- ✅ Plan Registry (10-11) - Fully tested and passing
- ✅ UniversalPlan Basics (12-15) - Fully tested and passing
- ✅ UniversalPlan with Tools (16-18) - Fully tested and passing
- ✅ UniversalPlan with LLM (19-21) - Requires OpenAI API key
- ⚠️ Deep Researcher (22) - Currently disabled (all code commented out)
- ✅ Deep Researcher Simple (23) - Working with proper async
- ✅ Job Manager (24) - Requires OpenAI API key (gpt-5-mini)

## 🔧 Requirements

Most examples require:
- Python 3.11+
- chuk-ai-planner installed
- chuk_tool_processor (for tool examples)
- chuk_session_manager (for session examples)

LLM examples (08, 19-21, 23-24) additionally require:
- OpenAI API key (set `OPENAI_API_KEY` environment variable)
- OpenAI Python package (`pip install openai`)
- **Default model:** gpt-5-mini (temperature=1.0)

**Note on gpt-5-mini:** This model requires temperature=1.0 (the only supported value). The framework has been updated to use this by default.

## 📂 Other Directories

- **legacy/** - Deprecated examples from older versions
- **future/** - Experimental examples for upcoming features

## 🤝 Contributing

When adding new examples:
1. Follow the numbering scheme
2. Include comprehensive docstrings
3. Add examples to appropriate category
4. Update this README
5. Test thoroughly before committing

---

**Need help?** Check the main project README or open an issue on GitHub.
