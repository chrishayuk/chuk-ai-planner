#!/usr/bin/env python
# examples/universal_executor_demo.py
"""
Universal Executor Demo – final working version
==============================================

• Shares the plan’s GraphStore with the executor
• Robustly unwraps ToolResult wrappers
• Functions use keyword-argument signatures that match the plan
• Prints whatever payloads come back (no hard-coded keys)
• Serialises datetime objects with `default=str`
"""

import asyncio
import json
import pprint
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from chuk_ai_planner.planner.universal_plan import UniversalPlan
from chuk_ai_planner.planner.universal_plan_executor import UniversalExecutor
from chuk_ai_planner.models.edges import EdgeKind


# --------------------------------------------------------------------------- helper
def unwrap(obj: Any) -> Any:
    """Return one sensible payload layer from a ToolResult / wrapper."""
    if isinstance(obj, (dict, list)):
        for k in ("result", "payload", "data"):
            if k in obj and isinstance(obj[k], (dict, list)):
                return obj[k]
        return obj

    for attr in ("result", "payload", "data"):
        if hasattr(obj, attr):
            inner = getattr(obj, attr)
            if inner is not None:
                return inner

    if is_dataclass(obj):
        return asdict(obj)

    if hasattr(obj, "model_dump"):       # pydantic v2
        return obj.model_dump()
    if hasattr(obj, "dict"):             # pydantic v1
        return obj.dict()

    return getattr(obj, "__dict__", obj)


# --------------------------------------------------------------------------- robust executor
class RobustExecutor(UniversalExecutor):
    """Stores unwrapped payloads under each result_variable."""

    def _process_step_results(self, step_id, results, context):
        context["results"][step_id] = results
        first = results[0] if isinstance(results, list) else results
        payload = unwrap(first)

        for edge in self.graph_store.get_edges(src=step_id, kind=EdgeKind.CUSTOM):
            if edge.data.get("type") == "result_variable":
                context["variables"][edge.data["variable"]] = payload
        return True


# --------------------------------------------------------------------------- custom tools / fns
async def weather_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    await asyncio.sleep(0.05)
    location = args.get("location", "Unknown")
    samples = {
        "New York": {"temperature": 72, "conditions": "Partly cloudy", "humidity": 65},
        "London":   {"temperature": 62, "conditions": "Rainy",          "humidity": 80},
        "Tokyo":    {"temperature": 78, "conditions": "Sunny",          "humidity": 70},
        "Sydney":   {"temperature": 68, "conditions": "Clear",          "humidity": 60},
        "Cairo":    {"temperature": 90, "conditions": "Hot",            "humidity": 30},
    }
    return samples.get(location, {"temperature": 75, "conditions": "Unknown", "humidity": 50})


async def batch_weather_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    locs = args.get("locations", [])
    return {"results": {loc: await weather_tool({"location": loc}) for loc in locs}}


def analyze_weather_function(weather_data: Dict[str, Any]) -> Dict[str, Any]:
    wd = weather_data["results"]
    n = len(wd)
    avg_t = sum(v["temperature"] for v in wd.values()) / n
    avg_h = sum(v["humidity"]    for v in wd.values()) / n
    conds: Dict[str, int] = {}
    for d in wd.values():
        conds[d["conditions"]] = conds.get(d["conditions"], 0) + 1
    most_common = max(conds, key=conds.get)
    return {
        "average_temperature": round(avg_t, 1),
        "average_humidity":    round(avg_h, 1),
        "most_common_condition": most_common,
        "condition_distribution": conds,
        "locations_analyzed": n,
    }


def create_report_function(analysis: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": "Global Weather Analysis Report",
        "summary": (
            f"{analysis['locations_analyzed']} cities analysed. "
            f"Avg T = {analysis['average_temperature']} °F, "
            f"Avg RH = {analysis['average_humidity']} %. "
            f"Most common: {analysis['most_common_condition']}."
        ),
        "details": analysis,
    }


def format_visualization_function(weather_data: Dict[str, Any],
                                  analysis: Dict[str, Any]) -> Dict[str, Any]:
    wd = weather_data["results"]
    temps = sorted(
        [{"location": k, "temperature": v["temperature"]} for k, v in wd.items()],
        key=lambda x: x["temperature"],
        reverse=True,
    )
    conds = [{"condition": c, "count": n} for c, n in analysis["condition_distribution"].items()]
    return {
        "title": "Global Weather Visualization",
        "temperature_data": temps,
        "condition_data": conds,
    }


# --------------------------------------------------------------------------- plan factory
def make_plan(store=None) -> UniversalPlan:
    plan = UniversalPlan(
        title="Global Weather Analysis",
        description="Analyse weather for multiple cities",
        tags=["weather", "analysis", "demo"],
        graph=store,
    )
    plan.set_variable("target_cities", ["New York", "London", "Tokyo", "Sydney", "Cairo"])
    plan.save()

    s1 = plan.add_tool_step(
        "Collect Weather Data",
        tool="batch_weather",
        args={"locations": "${target_cities}"},
        result_variable="weather_data",
    )
    s2 = plan.add_function_step(
        "Analyze Weather Data",
        function="analyze_weather",
        args={"weather_data": "${weather_data}"},
        result_variable="analysis",
        depends_on=[s1],
    )
    plan.add_function_step(
        "Generate Weather Report",
        function="create_report",
        args={"analysis": "${analysis}"},
        result_variable="report",
        depends_on=[s2],
    )
    plan.add_function_step(
        "Format Visualization Data",
        function="format_visualization",
        args={"weather_data": "${weather_data}", "analysis": "${analysis}"},
        result_variable="viz",
        depends_on=[s2],
    )
    plan.save()
    return plan


# --------------------------------------------------------------------------- main
async def main():
    print("🌤️  Universal Executor Demo\n" + "=" * 35)

    executor = RobustExecutor()
    plan     = make_plan(executor.graph_store)

    # Register tools / functions
    executor.register_tool("batch_weather", batch_weather_tool)
    executor.register_function("analyze_weather", analyze_weather_function)
    executor.register_function("create_report", create_report_function)
    executor.register_function("format_visualization", format_visualization_function)

    print("\n▶️ Executing …")
    res = await executor.execute_plan(plan)

    if not res["success"]:
        print(f"\n❌ Plan failed: {res['error']}")
        return

    report = unwrap(res["variables"].get("report"))
    viz    = unwrap(res["variables"].get("viz"))

    print("\n✅ Success!\n")

    print("=== REPORT ===")
    pprint.pprint(report, width=100, sort_dicts=False)

    print("\n=== VIZ DATA ===")
    pprint.pprint(viz, width=100, sort_dicts=False)

    with open("weather_analysis_results.json", "w") as fp:
        json.dump(res["variables"], fp, indent=2, default=str)
    print("\n💾  Results written to weather_analysis_results.json")


# --------------------------------------------------------------------------- entry point
if __name__ == "__main__":
    asyncio.run(main())
