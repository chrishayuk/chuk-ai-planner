# Examples Reorganization - November 16, 2025

## Overview

All example files have been renamed and reorganized into a clear, numbered progression from basic to advanced concepts. This makes it much easier for new users to follow a learning path.

## Summary

- **Total examples reorganized:** 20 files renamed
- **New numbering scheme:** 01-24 (sequential)
- **Categories:** 6 distinct groups
- **Verified working:** 11/11 core examples (01-11)

## Complete Renaming Map

### Core Concepts (01-04) - No changes
- 01_basic_graph.py ✅ (unchanged)
- 02_conditional_routing.py ✅ (unchanged)
- 03_tool_execution.py ✅ (unchanged)
- 04_function_routing.py ✅ (unchanged)

### Plan DSL & Execution (05-09)
- conditional_routing_simple.py → **05_routing_executor.py** ✅
- plan_executor_demo.py → **06_plan_executor.py** ✅
- plan_executor_tool_processor.py → **07_plan_with_tools.py** ✅
- plan_llm_executor_demo.py → **08_plan_from_llm.py** ✅
- simple_weather_analysis.py → **09_simple_pipeline.py** ✅

### Plan Registry (10-11)
- simple_plan_registry_demo.py → **10_plan_registry_basic.py** ✅
- plan_registry_demo.py → **11_plan_registry_advanced.py** ✅

### UniversalPlan Basics (12-15)
- universal_minimal_demo.py → **12_universal_plan_intro.py**
- universal_plan_demo.py → **13_universal_plan_features.py**
- universal_executor_demo.py → **14_universal_executor_basic.py**
- universal_executor_demo_advanced.py → **15_universal_executor_advanced.py**

### UniversalPlan with Tools (16-18)
- universal_plan_executor_tool_processor.py → **16_universal_with_tools.py**
- universal_plan_executor_demo_simple.py → **17_universal_plan_simple.py**
- universal_plan_executor_complete_output.py → **18_universal_plan_complete.py**

### UniversalPlan with LLM (19-21)
- universal_llm_executor_demo.py → **19_universal_llm_executor.py**
- universal_llm_plan_demo.py → **20_universal_llm_plan.py**
- universal_main_demo.py → **21_universal_main.py**

### Advanced Use Cases (22-24)
- universal_deep_researcher.py → **22_deep_researcher.py**
- universal_deep_researcher_simple.py → **23_deep_researcher_simple.py**
- job_manager_demo.py → **24_job_manager.py**

## Benefits

1. **Clear Learning Path**: Users can follow examples 01→24 in order
2. **Better Organization**: Related examples grouped together
3. **Easy Discovery**: Numbered files are easier to browse
4. **Consistent Naming**: All examples follow same pattern
5. **Preserved History**: Used `git mv` to maintain file history

## New Documentation

Created `examples/README.md` with:
- Complete category descriptions
- Learning path recommendations
- Usage examples
- Requirements for each category
- Testing instructions

## Testing Status

**Verified Working (11/11):**
- ✅ 01_basic_graph.py
- ✅ 02_conditional_routing.py
- ✅ 03_tool_execution.py
- ✅ 04_function_routing.py
- ✅ 05_routing_executor.py
- ✅ 06_plan_executor.py
- ✅ 07_plan_with_tools.py
- ✅ 08_plan_from_llm.py
- ✅ 09_simple_pipeline.py
- ✅ 10_plan_registry_basic.py
- ✅ 11_plan_registry_advanced.py

**Require Testing (13):**
- Examples 12-24 (UniversalPlan and advanced examples)
- May require API keys or additional dependencies

## Migration Guide

If you have scripts or documentation referencing old names:

```bash
# Old
python examples/conditional_routing_simple.py

# New
python examples/05_routing_executor.py
```

Use the mapping above to update references.

## Next Steps

1. Test UniversalPlan examples (12-24)
2. Fix any broken universal examples
3. Update main project README if it references old names
4. Consider adding index/catalog of all examples to docs

---

**Reorganization completed:** November 16, 2025
**Files renamed:** 20
**Git history:** Preserved via `git mv`
**Status:** ✅ Complete
