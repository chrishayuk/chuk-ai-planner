#!/usr/bin/env python
# examples/universal_main_demo.py
"""
Universal Plan Executor Main Demo
=================================

This demo showcases the UniversalExecutor framework with multiple domain examples:
- Sudoku Solver: Complex puzzle solving with multiple techniques
- Flight Booking: Multi-step booking process with validation
- Data Analysis: Complete data science pipeline

All powered by the same robust UniversalExecutor with proper variable flow,
dependency management, and error handling.
"""

import asyncio
import json
import pprint
from typing import Dict, Any, List

from chuk_ai_planner.planner.universal_plan import UniversalPlan
from chuk_ai_planner.planner.universal_plan_executor import UniversalExecutor


# ============================================================================
# SUDOKU DOMAIN TOOLS
# ============================================================================

async def sudoku_parse_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a Sudoku puzzle string into a grid"""
    puzzle = args.get("puzzle", "")
    print(f"🧩 Parsing Sudoku puzzle...")
    
    if len(puzzle) != 81:
        return {"valid": False, "error": "Invalid puzzle length", "grid": []}
    
    # Convert to 2D grid
    grid = []
    for i in range(9):
        row = []
        for j in range(9):
            cell = int(puzzle[i * 9 + j]) if puzzle[i * 9 + j] in "123456789" else 0
            row.append(cell)
        grid.append(row)
    
    return {"valid": True, "grid": grid, "string": puzzle}


def validate_sudoku_function(**kwargs) -> Dict[str, Any]:
    """Validate that a Sudoku grid follows the rules"""
    parse_result = kwargs.get("parse_result", {})
    grid = parse_result.get("grid", [])
    
    print(f"✓ Validating Sudoku puzzle...")
    
    if not grid:
        return {"valid": False, "error": "No grid to validate"}
    
    # Check for basic validity (no duplicate numbers in rows/cols/boxes)
    valid = True
    errors = []
    
    # Simple validation - check for obvious duplicates
    for i in range(9):
        row = [cell for cell in grid[i] if cell != 0]
        if len(row) != len(set(row)):
            valid = False
            errors.append(f"Duplicate in row {i+1}")
    
    return {
        "valid": valid,
        "errors": errors,
        "grid": grid,
        "cells_filled": sum(1 for row in grid for cell in row if cell != 0)
    }


async def sudoku_technique_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Apply Sudoku solving techniques"""
    grid = args.get("grid", [])
    technique = args.get("technique", "naked_single")
    
    print(f"🎯 Applying technique: {technique}")
    
    if not grid:
        return {"error": "No grid provided", "changes": [], "new_grid": []}
    
    # Create a copy of the grid
    new_grid = [row[:] for row in grid]
    changes = []
    
    # Simulate applying techniques
    if technique == "naked_single":
        # Find some cells to fill (simulation)
        for i in range(9):
            for j in range(9):
                if new_grid[i][j] == 0 and len(changes) < 3:
                    # Simple heuristic: fill with a valid number
                    for num in range(1, 10):
                        if num not in new_grid[i]:  # Simple row check
                            new_grid[i][j] = num
                            changes.append({"row": i, "col": j, "value": num, "technique": technique})
                            break
    
    return {
        "technique": technique,
        "changes": changes,
        "new_grid": new_grid,
        "changes_made": len(changes)
    }


async def sudoku_backtrack_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Solve Sudoku using backtracking algorithm"""
    grid = args.get("grid", [])
    
    print(f"🔄 Applying backtracking algorithm...")
    
    if not grid:
        return {"solved": False, "error": "No grid provided", "solution": {}}
    
    # For demo purposes, create a "solved" grid
    # In reality, this would implement actual backtracking
    solution = [row[:] for row in grid]
    
    # Fill empty cells with valid numbers (simulation)
    for i in range(9):
        for j in range(9):
            if solution[i][j] == 0:
                # Simple fill strategy for demo
                solution[i][j] = ((i * 3 + j) % 9) + 1
    
    # Convert back to string
    solution_string = ""
    for row in solution:
        for cell in row:
            solution_string += str(cell)
    
    return {
        "solved": True,
        "solution": {
            "grid": solution,
            "string": solution_string
        },
        "steps_used": 50  # Simulated
    }


def format_sudoku_function(**kwargs) -> Dict[str, Any]:
    """Format Sudoku solution for display"""
    solution = kwargs.get("solution", {})
    
    print(f"📋 Formatting solution...")
    
    if not solution:
        return {"error": "No solution to format", "formatted": "No solution available"}
    
    grid = solution.get("grid", [])
    if not grid:
        return {"error": "No grid in solution", "formatted": "Invalid solution"}
    
    # Format as ASCII grid
    formatted = "+-------+-------+-------+\n"
    for i in range(9):
        formatted += "| "
        for j in range(9):
            formatted += str(grid[i][j]) + " "
            if j % 3 == 2 and j < 8:
                formatted += "| "
        formatted += "|\n"
        if i % 3 == 2 and i < 8:
            formatted += "+-------+-------+-------+\n"
    formatted += "+-------+-------+-------+"
    
    return {
        "formatted": formatted,
        "grid": grid,
        "completion_rate": sum(1 for row in grid for cell in row if cell != 0) / 81 * 100
    }


# ============================================================================
# FLIGHT BOOKING DOMAIN TOOLS
# ============================================================================

async def validate_dates_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Validate travel dates"""
    departure = args.get("departure_date", "")
    return_date = args.get("return_date", "")
    
    print(f"📅 Validating travel dates: {departure} to {return_date}")
    
    # Simple validation
    valid = bool(departure and len(departure) == 10)
    if return_date:
        valid = valid and len(return_date) == 10
    
    return {
        "valid": valid,
        "departure_date": departure,
        "return_date": return_date,
        "trip_type": "round_trip" if return_date else "one_way"
    }


async def search_flights_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Search for available flights"""
    origin = args.get("origin", "")
    destination = args.get("destination", "")
    departure_date = args.get("departure_date", "")
    passengers = args.get("passengers", 1)
    
    print(f"✈️ Searching flights {origin} → {destination} for {passengers} passengers")
    
    # Generate sample flights
    flights = [
        {
            "id": "UA123",
            "airline": "United Airlines",
            "origin": origin,
            "destination": destination,
            "departure_time": f"{departure_date} 08:00",
            "arrival_time": f"{departure_date} 11:30",
            "price": 299.99,
            "stops": 0,
            "aircraft": "Boeing 737"
        },
        {
            "id": "DL456",
            "airline": "Delta Air Lines",
            "origin": origin,
            "destination": destination,
            "departure_time": f"{departure_date} 14:15",
            "arrival_time": f"{departure_date} 17:45",
            "price": 279.50,
            "stops": 0,
            "aircraft": "Airbus A320"
        },
        {
            "id": "SW789",
            "airline": "Southwest",
            "origin": origin,
            "destination": destination,
            "departure_time": f"{departure_date} 16:30",
            "arrival_time": f"{departure_date} 19:55",
            "price": 199.99,
            "stops": 1,
            "aircraft": "Boeing 737"
        }
    ]
    
    return {
        "flights": flights,
        "search_results": {
            "total_flights": len(flights),
            "origin": origin,
            "destination": destination,
            "date": departure_date
        }
    }


def filter_flights_function(**kwargs) -> Dict[str, Any]:
    """Filter flights based on user preferences"""
    search_results = kwargs.get("search_results", {})
    preferences = kwargs.get("preferences", {})
    
    flights = search_results.get("flights", [])
    max_price = preferences.get("max_price", 500)
    direct_only = preferences.get("direct_only", False)
    
    print(f"🔍 Filtering {len(flights)} flights (max_price: ${max_price}, direct_only: {direct_only})")
    
    # Apply filters
    filtered = []
    for flight in flights:
        if flight["price"] <= max_price:
            if not direct_only or flight["stops"] == 0:
                filtered.append(flight)
    
    return {
        "filtered_flights": filtered,
        "filter_summary": {
            "original_count": len(flights),
            "filtered_count": len(filtered),
            "filters_applied": {"max_price": max_price, "direct_only": direct_only}
        }
    }


def calculate_pricing_function(**kwargs) -> Dict[str, Any]:
    """Calculate final pricing with taxes and fees"""
    filtered_results = kwargs.get("filtered_results", {})
    passengers = kwargs.get("passengers", 1)
    
    filtered_flights = filtered_results.get("filtered_flights", [])
    
    print(f"💰 Calculating pricing for {len(filtered_flights)} flights")
    
    if not filtered_flights:
        return {"error": "No flights to price", "pricing": []}
    
    pricing = []
    for flight in filtered_flights:
        base_price = flight["price"] * passengers
        taxes = base_price * 0.12
        fees = 25.99 * passengers
        total = base_price + taxes + fees
        
        pricing.append({
            "flight_id": flight["id"],
            "base_price": base_price,
            "taxes": round(taxes, 2),
            "fees": fees,
            "total": round(total, 2),
            "per_passenger": round(total / passengers, 2)
        })
    
    return {
        "pricing": pricing,
        "recommended": min(pricing, key=lambda x: x["total"]) if pricing else None
    }


async def book_flight_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Book the selected flight"""
    pricing = args.get("pricing", {})
    passenger_info = args.get("passenger_info", [])
    
    recommended = pricing.get("recommended", {})
    
    print(f"📝 Booking flight {recommended.get('flight_id', 'unknown')}")
    
    if not recommended:
        return {"success": False, "error": "No flight selected for booking"}
    
    # Simulate booking process
    booking_confirmation = {
        "confirmation_number": "ABC123XYZ",
        "flight_id": recommended["flight_id"],
        "total_price": recommended["total"],
        "passengers": len(passenger_info),
        "status": "confirmed",
        "booking_date": "2024-01-15"
    }
    
    return {
        "success": True,
        "booking": booking_confirmation,
        "next_steps": ["Check-in online 24 hours before", "Arrive 2 hours early", "Bring valid ID"]
    }


# ============================================================================
# DATA ANALYSIS DOMAIN TOOLS
# ============================================================================

async def load_data_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """Load data from a source"""
    source = args.get("source", "")
    parameters = args.get("parameters", {})
    
    print(f"📊 Loading data from: {source}")
    
    # Generate sample dataset
    data = [
        {"id": 1, "x": 10.5, "y": 25.2, "category": "A", "value": 100},
        {"id": 2, "x": 15.3, "y": 30.1, "category": "B", "value": 150},
        {"id": 3, "x": 8.7, "y": 18.9, "category": "A", "value": 80},
        {"id": 4, "x": 12.1, "y": 22.5, "category": "C", "value": 120},
        {"id": 5, "x": 20.4, "y": 35.8, "category": "B", "value": 200},
        {"id": 6, "x": 6.2, "y": 15.3, "category": "A", "value": 60},
        {"id": 7, "x": 18.9, "y": 32.7, "category": "C", "value": 180},
        {"id": 8, "x": 14.6, "y": 28.4, "category": "B", "value": 140}
    ]
    
    metadata = {
        "source": source,
        "rows": len(data),
        "columns": ["id", "x", "y", "category", "value"],
        "load_parameters": parameters
    }
    
    return {"data": data, "metadata": metadata}


def clean_data_function(**kwargs) -> Dict[str, Any]:
    """Clean and preprocess the data"""
    load_result = kwargs.get("load_result", {})
    data = load_result.get("data", [])
    
    print(f"🧹 Cleaning data ({len(data)} records)")
    
    # Simple cleaning - remove any invalid records
    cleaned_data = []
    removed_count = 0
    
    for record in data:
        # Check for required fields
        if all(key in record for key in ["id", "x", "y"]):
            # Check for valid numeric values
            try:
                float(record["x"])
                float(record["y"])
                cleaned_data.append(record)
            except (ValueError, TypeError):
                removed_count += 1
        else:
            removed_count += 1
    
    cleaning_summary = {
        "original_count": len(data),
        "cleaned_count": len(cleaned_data),
        "removed_count": removed_count,
        "removal_rate": removed_count / len(data) if data else 0
    }
    
    return {
        "cleaned_data": cleaned_data,
        "cleaning_summary": cleaning_summary
    }


def analyze_data_function(**kwargs) -> Dict[str, Any]:
    """Perform statistical analysis on the data"""
    clean_result = kwargs.get("clean_result", {})
    data = clean_result.get("cleaned_data", [])
    
    print(f"📈 Analyzing data ({len(data)} records)")
    
    if not data:
        return {"error": "No data to analyze", "statistics": {}}
    
    # Calculate statistics
    x_values = [float(record["x"]) for record in data]
    y_values = [float(record["y"]) for record in data]
    value_values = [float(record.get("value", 0)) for record in data]
    
    def calc_stats(values):
        if not values:
            return {"mean": 0, "min": 0, "max": 0, "count": 0}
        return {
            "mean": round(sum(values) / len(values), 2),
            "min": min(values),
            "max": max(values),
            "count": len(values)
        }
    
    # Category distribution
    categories = {}
    for record in data:
        cat = record.get("category", "Unknown")
        categories[cat] = categories.get(cat, 0) + 1
    
    statistics = {
        "x": calc_stats(x_values),
        "y": calc_stats(y_values),
        "value": calc_stats(value_values),
        "categories": categories,
        "total_records": len(data)
    }
    
    return {"statistics": statistics, "data": data}


def generate_report_function(**kwargs) -> Dict[str, Any]:
    """Generate a comprehensive analysis report"""
    analysis_result = kwargs.get("analysis_result", {})
    statistics = analysis_result.get("statistics", {})
    
    print(f"📝 Generating analysis report")
    
    if not statistics:
        return {"error": "No statistics to report", "report": {}}
    
    # Create summary
    summary = f"""
Data Analysis Report
===================
Total Records: {statistics.get('total_records', 0)}

Variable Statistics:
- X: mean={statistics.get('x', {}).get('mean', 'N/A')}, range=[{statistics.get('x', {}).get('min', 'N/A')}, {statistics.get('x', {}).get('max', 'N/A')}]
- Y: mean={statistics.get('y', {}).get('mean', 'N/A')}, range=[{statistics.get('y', {}).get('min', 'N/A')}, {statistics.get('y', {}).get('max', 'N/A')}]
- Value: mean={statistics.get('value', {}).get('mean', 'N/A')}, range=[{statistics.get('value', {}).get('min', 'N/A')}, {statistics.get('value', {}).get('max', 'N/A')}]

Category Distribution:
""".strip()
    
    for category, count in statistics.get('categories', {}).items():
        summary += f"\n- {category}: {count} records"
    
    report = {
        "title": "Data Analysis Report",
        "summary": summary,
        "statistics": statistics,
        "generated_at": "2024-01-15 10:30:00"
    }
    
    return {"report": report}


# ============================================================================
# PLAN CREATION FUNCTIONS
# ============================================================================

def create_sudoku_plan() -> UniversalPlan:
    """Create a Sudoku solving plan"""
    plan = UniversalPlan(
        title="Complete Sudoku Solver",
        description="Parse, validate, and solve Sudoku puzzles using multiple techniques",
        tags=["sudoku", "puzzle", "solving"]
    )
    
    # Step 1: Parse the puzzle
    parse_step = plan.add_tool_step(
        title="Parse Sudoku puzzle",
        tool="sudoku_parse",
        args={"puzzle": "${puzzle}"},
        result_variable="parse_result"
    )
    
    # Step 2: Validate the puzzle
    validate_step = plan.add_function_step(
        title="Validate puzzle format",
        function="validate_sudoku",
        args={"parse_result": "${parse_result}"},
        result_variable="validation_result",
        depends_on=[parse_step]
    )
    
    # Step 3: Apply solving techniques
    technique_step = plan.add_tool_step(
        title="Apply solving techniques",
        tool="sudoku_technique",
        args={"grid": "${parse_result.grid}", "technique": "naked_single"},
        result_variable="technique_result",
        depends_on=[validate_step]
    )
    
    # Step 4: Apply backtracking if needed
    backtrack_step = plan.add_tool_step(
        title="Apply backtracking algorithm",
        tool="sudoku_backtrack",
        args={"grid": "${technique_result.new_grid}"},
        result_variable="solution_result",
        depends_on=[technique_step]
    )
    
    # Step 5: Format the solution
    format_step = plan.add_function_step(
        title="Format solution",
        function="format_sudoku",
        args={"solution": "${solution_result.solution}"},
        result_variable="final_result",
        depends_on=[backtrack_step]
    )
    
    return plan


def create_flight_booking_plan() -> UniversalPlan:
    """Create a flight booking plan"""
    plan = UniversalPlan(
        title="Complete Flight Booking Process",
        description="Search, filter, price, and book flights",
        tags=["travel", "booking", "flights"]
    )
    
    # Step 1: Validate travel dates
    validate_step = plan.add_tool_step(
        title="Validate travel dates",
        tool="validate_dates",
        args={
            "departure_date": "${departure_date}",
            "return_date": "${return_date}"
        },
        result_variable="date_validation"
    )
    
    # Step 2: Search for flights
    search_step = plan.add_tool_step(
        title="Search available flights",
        tool="search_flights",
        args={
            "origin": "${origin}",
            "destination": "${destination}",
            "departure_date": "${departure_date}",
            "passengers": "${passengers}"
        },
        result_variable="search_results",
        depends_on=[validate_step]
    )
    
    # Step 3: Filter flights
    filter_step = plan.add_function_step(
        title="Filter flights by preferences",
        function="filter_flights",
        args={
            "search_results": "${search_results}",
            "preferences": "${user_profile.preferences}"
        },
        result_variable="filtered_results",
        depends_on=[search_step]
    )
    
    # Step 4: Calculate pricing
    pricing_step = plan.add_function_step(
        title="Calculate pricing",
        function="calculate_pricing",
        args={
            "filtered_results": "${filtered_results}",
            "passengers": "${passengers}"
        },
        result_variable="pricing_results",
        depends_on=[filter_step]
    )
    
    # Step 5: Book the flight
    booking_step = plan.add_tool_step(
        title="Book selected flight",
        tool="book_flight",
        args={
            "pricing": "${pricing_results}",
            "passenger_info": "${passenger_info}"
        },
        result_variable="booking_results",
        depends_on=[pricing_step]
    )
    
    return plan


def create_data_analysis_plan() -> UniversalPlan:
    """Create a data analysis plan"""
    plan = UniversalPlan(
        title="Complete Data Analysis Pipeline",
        description="Load, clean, analyze, and report on data",
        tags=["data", "analysis", "statistics"]
    )
    
    # Step 1: Load data
    load_step = plan.add_tool_step(
        title="Load data from source",
        tool="load_data",
        args={
            "source": "${data_source}",
            "parameters": "${load_parameters}"
        },
        result_variable="load_result"
    )
    
    # Step 2: Clean data
    clean_step = plan.add_function_step(
        title="Clean and preprocess data",
        function="clean_data",
        args={"load_result": "${load_result}"},
        result_variable="clean_result",
        depends_on=[load_step]
    )
    
    # Step 3: Analyze data
    analyze_step = plan.add_function_step(
        title="Perform statistical analysis",
        function="analyze_data",
        args={"clean_result": "${clean_result}"},
        result_variable="analysis_result",
        depends_on=[clean_step]
    )
    
    # Step 4: Generate report
    report_step = plan.add_function_step(
        title="Generate analysis report",
        function="generate_report",
        args={"analysis_result": "${analysis_result}"},
        result_variable="final_report",
        depends_on=[analyze_step]
    )
    
    return plan


# ============================================================================
# DEMO EXECUTION FUNCTIONS
# ============================================================================

async def run_sudoku_demo(executor: UniversalExecutor):
    """Run the Sudoku solver demo"""
    print("\n" + "=" * 60)
    print("🧩 SUDOKU SOLVER DEMO")
    print("=" * 60)
    
    # Example Sudoku puzzle
    puzzle = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
    
    # Display the puzzle
    print("\n📋 Input puzzle:")
    formatted_puzzle = "+-------+-------+-------+\n"
    for i in range(9):
        formatted_puzzle += "| "
        for j in range(9):
            cell = puzzle[i * 9 + j]
            cell = " " if cell == "0" else cell
            formatted_puzzle += cell + " "
            if j % 3 == 2 and j < 8:
                formatted_puzzle += "| "
        formatted_puzzle += "|\n"
        if i % 3 == 2 and i < 8:
            formatted_puzzle += "+-------+-------+-------+\n"
    formatted_puzzle += "+-------+-------+-------+"
    print(formatted_puzzle)
    
    # Create and execute the plan
    plan = create_sudoku_plan()
    print(f"\n📋 Executing plan: {plan.title}")
    
    result = await executor.execute_plan(plan, {"puzzle": puzzle})
    
    if result["success"]:
        final_result = result["variables"].get("final_result", {})
        if "formatted" in final_result:
            print("\n✅ Solution:")
            print(final_result["formatted"])
            print(f"\nCompletion rate: {final_result.get('completion_rate', 0):.1f}%")
        else:
            print("\n⚠️ Solution found but formatting failed")
            pprint.pprint(final_result)
    else:
        print(f"\n❌ Sudoku solving failed: {result.get('error', 'Unknown error')}")


async def run_flight_booking_demo(executor: UniversalExecutor):
    """Run the flight booking demo"""
    print("\n" + "=" * 60)
    print("✈️ FLIGHT BOOKING DEMO")
    print("=" * 60)
    
    # Example booking parameters
    booking_data = {
        "origin": "NYC",
        "destination": "LAX",
        "departure_date": "2024-06-15",
        "return_date": "2024-06-22",
        "passengers": 2,
        "user_profile": {
            "name": "Jane Smith",
            "preferences": {
                "max_price": 350,
                "direct_only": False
            }
        },
        "passenger_info": [
            {"name": "Jane Smith", "dob": "1985-03-15"},
            {"name": "John Smith", "dob": "1983-07-22"}
        ]
    }
    
    print(f"\n📋 Booking Request:")
    print(f"   Route: {booking_data['origin']} → {booking_data['destination']}")
    print(f"   Dates: {booking_data['departure_date']} to {booking_data['return_date']}")
    print(f"   Passengers: {booking_data['passengers']}")
    print(f"   Max Price: ${booking_data['user_profile']['preferences']['max_price']}")
    
    # Create and execute the plan
    plan = create_flight_booking_plan()
    print(f"\n📋 Executing plan: {plan.title}")
    
    result = await executor.execute_plan(plan, booking_data)
    
    if result["success"]:
        booking_results = result["variables"].get("booking_results", {})
        pricing_results = result["variables"].get("pricing_results", {})
        
        print("\n✅ Booking completed!")
        
        if "booking" in booking_results:
            booking = booking_results["booking"]
            print(f"\n📋 Confirmation: {booking.get('confirmation_number', 'N/A')}")
            print(f"Flight ID: {booking.get('flight_id', 'N/A')}")
            print(f"Total Price: ${booking.get('total_price', 0)}")
            print(f"Status: {booking.get('status', 'Unknown')}")
        
        if pricing_results.get("recommended"):
            print(f"\nPricing breakdown:")
            rec = pricing_results["recommended"]
            print(f"  Base: ${rec.get('base_price', 0)}")
            print(f"  Taxes: ${rec.get('taxes', 0)}")
            print(f"  Fees: ${rec.get('fees', 0)}")
            print(f"  Total: ${rec.get('total', 0)}")
    else:
        print(f"\n❌ Flight booking failed: {result.get('error', 'Unknown error')}")


async def run_data_analysis_demo(executor: UniversalExecutor):
    """Run the data analysis demo"""
    print("\n" + "=" * 60)
    print("📊 DATA ANALYSIS DEMO")
    print("=" * 60)
    
    # Example analysis parameters
    analysis_params = {
        "data_source": "sample_dataset.csv",
        "load_parameters": {
            "delimiter": ",",
            "header": True
        }
    }
    
    print(f"\n📋 Analysis Request:")
    print(f"   Data Source: {analysis_params['data_source']}")
    print(f"   Parameters: {analysis_params['load_parameters']}")
    
    # Create and execute the plan
    plan = create_data_analysis_plan()
    print(f"\n📋 Executing plan: {plan.title}")
    
    result = await executor.execute_plan(plan, analysis_params)
    
    if result["success"]:
        final_report = result["variables"].get("final_report", {})
        analysis_result = result["variables"].get("analysis_result", {})
        
        print("\n✅ Analysis completed!")
        
        if "report" in final_report:
            report = final_report["report"]
            print(f"\n📊 {report.get('title', 'Report')}")
            print(f"Generated: {report.get('generated_at', 'Unknown')}")
            print("\nSummary:")
            print(report.get('summary', 'No summary available'))
        
        if "statistics" in analysis_result:
            stats = analysis_result["statistics"]
            print(f"\n📈 Key Statistics:")
            print(f"   Total Records: {stats.get('total_records', 0)}")
            if "x" in stats:
                x_stats = stats["x"]
                print(f"   X Variable: mean={x_stats.get('mean', 'N/A')}, range=[{x_stats.get('min', 'N/A')}-{x_stats.get('max', 'N/A')}]")
            if "categories" in stats:
                print(f"   Categories: {', '.join(f'{k}({v})' for k, v in stats['categories'].items())}")
    else:
        print(f"\n❌ Data analysis failed: {result.get('error', 'Unknown error')}")


# ============================================================================
# MAIN DEMO FUNCTION
# ============================================================================

async def main():
    """Main demo function"""
    print("🚀 Universal Plan Executor - Multi-Domain Demo")
    print("=" * 60)
    print("This demo showcases the UniversalExecutor framework with three different domains:")
    print("1. 🧩 Sudoku Solver - Complex puzzle solving")
    print("2. ✈️ Flight Booking - Multi-step booking process")
    print("3. 📊 Data Analysis - Complete analytics pipeline")
    print("\nAll powered by the same robust execution engine!")
    
    # Create executor and register all tools/functions
    executor = UniversalExecutor()
    
    print("\n🔧 Registering tools and functions...")
    
    # Register Sudoku tools
    executor.register_tool("sudoku_parse", sudoku_parse_tool)
    executor.register_tool("sudoku_technique", sudoku_technique_tool)
    executor.register_tool("sudoku_backtrack", sudoku_backtrack_tool)
    executor.register_function("validate_sudoku", validate_sudoku_function)
    executor.register_function("format_sudoku", format_sudoku_function)
    
    # Register Flight booking tools
    executor.register_tool("validate_dates", validate_dates_tool)
    executor.register_tool("search_flights", search_flights_tool)
    executor.register_tool("book_flight", book_flight_tool)
    executor.register_function("filter_flights", filter_flights_function)
    executor.register_function("calculate_pricing", calculate_pricing_function)
    
    # Register Data analysis tools
    executor.register_tool("load_data", load_data_tool)
    executor.register_function("clean_data", clean_data_function)
    executor.register_function("analyze_data", analyze_data_function)
    executor.register_function("generate_report", generate_report_function)
    
    print("✅ All tools and functions registered!")
    
    # Ask user which demo to run
    print("\n" + "=" * 60)
    print("Which demo would you like to run?")
    print("1. 🧩 Sudoku Solver")
    print("2. ✈️ Flight Booking")
    print("3. 📊 Data Analysis")
    print("4. 🚀 Run All Demos")
    print("0. 🚪 Exit")
    
    try:
        choice = input("\nEnter your choice (0-4): ").strip()
        
        if choice == "1":
            await run_sudoku_demo(executor)
        elif choice == "2":
            await run_flight_booking_demo(executor)
        elif choice == "3":
            await run_data_analysis_demo(executor)
        elif choice == "4":
            await run_sudoku_demo(executor)
            await run_flight_booking_demo(executor)
            await run_data_analysis_demo(executor)
        elif choice == "0":
            print("👋 Goodbye!")
            return
        else:
            print("❌ Invalid choice. Please enter a number between 0 and 4.")
            return
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed! Thank you for trying the Universal Plan Executor!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")


if __name__ == "__main__":
    asyncio.run(main())