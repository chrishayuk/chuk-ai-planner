"""
main_demo.py - Demo of the generic plan engine with different domain plans

This script shows how the same generic plan engine can execute plans for
completely different domains (Sudoku, flight booking, data analysis) without
any domain-specific code.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional

from generic_plan_engine import PlanDatabase, GenericPlanExecutor
from plan_examples import store_plans_in_database

# Sample tool implementations for Sudoku domain
async def sudoku_parse(args: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a Sudoku puzzle string"""
    puzzle = args.get("puzzle", "")
    print(f"Parsing Sudoku puzzle: {puzzle[:10]}...")
    
    # Simple validation and parsing
    if len(puzzle) != 81:
        return {"valid": False, "error": "Invalid puzzle length"}
    
    # Convert to 2D grid
    grid = []
    for i in range(9):
        row = []
        for j in range(9):
            cell = int(puzzle[i * 9 + j]) if puzzle[i * 9 + j] in "123456789" else 0
            row.append(cell)
        grid.append(row)
    
    return {"valid": True, "grid": grid, "string": puzzle}

async def sudoku_technique(args: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a Sudoku solving technique"""
    grid = args.get("grid", [])
    technique = args.get("technique", "")
    print(f"Applying technique: {technique}")
    
    # Simple simulation - in reality would implement actual techniques
    # Just make some changes for demonstration
    new_grid = [row[:] for row in grid]  # Copy grid
    
    changes = []
    if technique == "naked_single":
        # Simulate finding 2 naked singles
        if new_grid[0][3] == 0:
            new_grid[0][3] = 7
            changes.append({"row": 0, "col": 3, "value": 7})
        if new_grid[2][6] == 0:
            new_grid[2][6] = 5
            changes.append({"row": 2, "col": 6, "value": 5})
    
    elif technique == "hidden_single":
        # Simulate finding 1 hidden single
        if new_grid[5][7] == 0:
            new_grid[5][7] = 9
            changes.append({"row": 5, "col": 7, "value": 9})
    
    return {
        "technique": technique,
        "changes": changes,
        "changed_cells": len(changes),
        "original_grid": grid,
        "new_grid": new_grid
    }

async def sudoku_backtrack(args: Dict[str, Any]) -> Dict[str, Any]:
    """Solve Sudoku using backtracking"""
    grid = args.get("grid", [])
    print("Solving using backtracking algorithm")
    
    # For demo, just return a "solved" grid (not actually solving)
    # In reality would implement actual backtracking
    solution = [row[:] for row in grid]  # Copy grid
    
    # Fill in remaining cells with dummy values
    for i in range(9):
        for j in range(9):
            if solution[i][j] == 0:
                solution[i][j] = (i + j) % 9 + 1
    
    # Convert to string
    solution_string = ""
    for row in solution:
        for cell in row:
            solution_string += str(cell)
    
    return {"solved": True, "solution": {"grid": solution, "string": solution_string}}

def format_solution(args: Dict[str, Any]) -> Dict[str, Any]:
    """Format the Sudoku solution for display"""
    solution = args.get("solution", {})
    grid = solution.get("grid", [])
    
    if not grid:
        return {"error": "No solution grid provided"}
    
    # Format as string with row/column separators
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
    
    return {"formatted_solution": formatted, "grid": grid}

# Sample tool implementations for flight booking domain
async def validate_dates(args: Dict[str, Any]) -> Dict[str, Any]:
    """Validate travel dates"""
    departure = args.get("departure_date", "")
    return_date = args.get("return_date", "")
    print(f"Validating dates: {departure} to {return_date}")
    
    # Simple validation - in reality would check date formats, etc.
    return {"valid": True, "departure": departure, "return": return_date}

async def search_flights(args: Dict[str, Any]) -> Dict[str, Any]:
    """Search for flights"""
    origin = args.get("origin", "")
    destination = args.get("destination", "")
    departure = args.get("departure_date", "")
    print(f"Searching flights from {origin} to {destination} on {departure}")
    
    # Generate dummy flight results
    flights = [
        {
            "id": "FL123",
            "airline": "Example Air",
            "origin": origin,
            "destination": destination,
            "departure_time": f"{departure} 08:00",
            "arrival_time": f"{departure} 10:30",
            "price": 299.99,
            "stops": 0
        },
        {
            "id": "FL456",
            "airline": "Budget Airways",
            "origin": origin,
            "destination": destination,
            "departure_time": f"{departure} 11:15",
            "arrival_time": f"{departure} 14:25",
            "price": 249.50,
            "stops": 1
        },
        {
            "id": "FL789",
            "airline": "Luxury Lines",
            "origin": origin,
            "destination": destination,
            "departure_time": f"{departure} 13:45",
            "arrival_time": f"{departure} 15:55",
            "price": 399.99,
            "stops": 0
        }
    ]
    
    return {"flights": flights, "count": len(flights)}

async def filter_flights(args: Dict[str, Any]) -> Dict[str, Any]:
    """Filter flights by preferences"""
    flights = args.get("flights", [])
    preferences = args.get("preferences", {})
    print(f"Filtering {len(flights)} flights by preferences")
    
    # Apply filters - for demo, just return the first flight
    filtered = flights[:1] if flights else []
    
    return {"filtered_flights": filtered, "count": len(filtered)}

async def get_flight_pricing(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get pricing for a flight"""
    flight = args.get("flight", {})
    print(f"Getting pricing for flight {flight.get('id', 'unknown')}")
    
    # Generate pricing info
    base_price = flight.get("price", 0)
    taxes = base_price * 0.12
    fees = 24.99
    
    return {
        "flight_id": flight.get("id"),
        "price": base_price,
        "taxes": taxes,
        "fees": fees,
        "total": base_price + taxes + fees
    }

async def check_promotions(args: Dict[str, Any]) -> Dict[str, Any]:
    """Check for applicable promotions"""
    flight = args.get("flight", {})
    user = args.get("user", {})
    print(f"Checking promotions for flight {flight.get('id', 'unknown')}")
    
    # Sample promotions
    promotions = [
        {
            "code": "SAVE10",
            "description": "10% off base fare",
            "discount_percentage": 10,
            "applicable": True
        }
    ]
    
    return {"available_promotions": promotions, "count": len(promotions)}

async def calculate_price(args: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate final price with promotions"""
    base_price = args.get("base_price", 0)
    promotions = args.get("promotions", [])
    fees = args.get("fees", 0)
    print(f"Calculating final price from base price {base_price}")
    
    # Apply promotions
    discount = 0
    for promo in promotions:
        if promo.get("applicable", False):
            discount += base_price * (promo.get("discount_percentage", 0) / 100)
    
    final_price = base_price - discount + fees
    
    return {
        "original_price": base_price,
        "discount": discount,
        "fees": fees,
        "total": final_price
    }

# Sample tool implementations for data analysis domain
async def validate_data_source(args: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a data source"""
    source = args.get("source", "")
    print(f"Validating data source: {source}")
    
    # Simple validation
    return {"valid": True, "source": source}

async def load_data(args: Dict[str, Any]) -> Dict[str, Any]:
    """Load data from a source"""
    source = args.get("source", "")
    print(f"Loading data from source: {source}")
    
    # Generate dummy data
    data = [
        {"id": 1, "x": 10, "y": 20, "category": "A"},
        {"id": 2, "x": 15, "y": 22, "category": "B"},
        {"id": 3, "x": 8, "y": 16, "category": "A"},
        {"id": 4, "x": 12, "y": 21, "category": "C"}
    ]
    
    return {"data": data, "count": len(data)}

async def validate_data(args: Dict[str, Any]) -> Dict[str, Any]:
    """Validate loaded data"""
    data = args.get("data", [])
    print(f"Validating {len(data)} data records")
    
    # Check for required fields
    valid = all("id" in item and "x" in item and "y" in item for item in data)
    
    summary = {
        "count": len(data),
        "valid": valid,
        "fields": ["id", "x", "y", "category"]
    }
    
    return {"data": data, "valid": valid, "summary": summary}

async def handle_missing(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handle missing values in data"""
    data = args.get("data", [])
    strategy = args.get("strategy", "mean")
    print(f"Handling missing values using strategy: {strategy}")
    
    # For demo, just return the data unchanged
    return {"data": data, "strategy_used": strategy}

async def normalize_data(args: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize data"""
    data = args.get("data", [])
    method = args.get("method", "z-score")
    print(f"Normalizing data using method: {method}")
    
    # For demo, just return the data unchanged
    return {"data": data, "method_used": method}

async def remove_outliers(args: Dict[str, Any]) -> Dict[str, Any]:
    """Remove outliers from data"""
    data = args.get("data", [])
    method = args.get("method", "iqr")
    print(f"Removing outliers using method: {method}")
    
    # For demo, just return the data unchanged
    cleaned_data = data
    
    summary = {
        "original_count": len(data),
        "cleaned_count": len(cleaned_data),
        "removed_count": len(data) - len(cleaned_data)
    }
    
    return {"data": cleaned_data, "summary": summary}

async def descriptive_statistics(args: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate descriptive statistics"""
    data = args.get("data", [])
    print(f"Calculating statistics for {len(data)} records")
    
    # Calculate simple statistics for x and y
    x_values = [item.get("x", 0) for item in data]
    y_values = [item.get("y", 0) for item in data]
    
    x_mean = sum(x_values) / len(x_values) if x_values else 0
    y_mean = sum(y_values) / len(y_values) if y_values else 0
    
    stats = {
        "x": {"mean": x_mean, "min": min(x_values) if x_values else 0, "max": max(x_values) if x_values else 0},
        "y": {"mean": y_mean, "min": min(y_values) if y_values else 0, "max": max(y_values) if y_values else 0}
    }
    
    return {"statistics": stats}

async def register_tools(db: PlanDatabase):
    """Register all the tool implementations with the database"""
    # Sudoku tools
    db.register_tool("sudoku_parse", sudoku_parse)
    db.register_tool("sudoku_technique", sudoku_technique)
    db.register_tool("sudoku_backtrack", sudoku_backtrack)
    db.register_tool("format_solution", format_solution)
    
    # Flight booking tools
    db.register_tool("validate_dates", validate_dates)
    db.register_tool("search_flights", search_flights)
    db.register_tool("filter_flights", filter_flights)
    db.register_tool("get_flight_pricing", get_flight_pricing)
    db.register_tool("check_promotions", check_promotions)
    db.register_tool("calculate_price", calculate_price)
    
    # Data analysis tools
    db.register_tool("validate_data_source", validate_data_source)
    db.register_tool("load_data", load_data)
    db.register_tool("validate_data", validate_data)
    db.register_tool("handle_missing", handle_missing)
    db.register_tool("normalize_data", normalize_data)
    db.register_tool("remove_outliers", remove_outliers)
    db.register_tool("descriptive_statistics", descriptive_statistics)
    
    # Register stub implementations for remaining tools to avoid errors
    async def stub_tool(args: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = args.get("_tool_name", "unknown")
        print(f"[STUB] Executing {tool_name}")
        return {"stub": True, "tool": tool_name, "args": args}
    
    for tool_name in [
        # Flight booking stubs
        "validate_payment", "reserve_seats", "process_payment", "confirm_booking",
        "generate_itinerary", "send_email",
        
        # Data analysis stubs
        "correlation_analysis", "regression_analysis", "create_charts", 
        "create_heatmap", "create_regression_plots", "compile_report",
        "generate_analysis_report"
    ]:
        db.register_tool(tool_name, lambda args, tn=tool_name: stub_tool({**args, "_tool_name": tn}))

async def run_sudoku_demo(executor: GenericPlanExecutor, plan_id: str):
    """Run the Sudoku solver demo"""
    print("\n===== SUDOKU SOLVER DEMO =====")
    
    # Example Sudoku puzzle - 0s represent empty cells
    puzzle = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
    
    # Display the puzzle
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
    
    print("Input puzzle:")
    print(formatted_puzzle)
    
    # Execute the Sudoku solver plan
    print("\nExecuting Sudoku solver plan...")
    result = await executor.execute_plan(plan_id, {"puzzle": puzzle})
    
    if result.get("success", False):
        # Display the solution
        print("\nSolution:")
        formatted_solution = result.get("variables", {}).get("formatted_solution", {}).get("formatted_solution", "")
        print(formatted_solution)
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

async def run_flight_booking_demo(executor: GenericPlanExecutor, plan_id: str):
    """Run the flight booking demo"""
    print("\n===== FLIGHT BOOKING DEMO =====")
    
    # Example flight booking parameters
    booking_params = {
        "origin": "NYC",
        "destination": "LAX",
        "departure_date": "2023-06-15",
        "return_date": "2023-06-22",
        "passengers": 2,
        "user_profile": {
            "name": "Jane Smith",
            "email": "jane.smith@example.com",
            "preferences": {
                "seat_type": "window",
                "meal_preference": "vegetarian"
            }
        },
        "payment_details": {
            "type": "credit_card",
            "number": "****-****-****-4321",
            "expiry": "05/25"
        },
        "passenger_info": [
            {
                "name": "Jane Smith",
                "dob": "1985-03-15",
                "passport": "P123456789"
            },
            {
                "name": "John Smith",
                "dob": "1983-07-22",
                "passport": "P987654321"
            }
        ]
    }
    
    print(f"Booking flight from {booking_params['origin']} to {booking_params['destination']}")
    print(f"Departure: {booking_params['departure_date']}, Return: {booking_params['return_date']}")
    print(f"Passengers: {booking_params['passengers']}")
    
    # Execute the flight booking plan
    print("\nExecuting flight booking plan...")
    result = await executor.execute_plan(plan_id, booking_params)
    
    if result.get("success", False):
        # Display booking confirmation
        booking = result.get("variables", {}).get("booking_results", {}).get("booking_confirmation", {})
        itinerary = result.get("variables", {}).get("itinerary", {})
        
        print("\nBooking confirmed!")
        print(f"Confirmation details: {json.dumps(booking, indent=2)}")
        print(f"Itinerary: {json.dumps(itinerary, indent=2)}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

async def run_data_analysis_demo(executor: GenericPlanExecutor, plan_id: str):
    """Run the data analysis demo"""
    print("\n===== DATA ANALYSIS DEMO =====")
    
    # Example data analysis parameters
    analysis_params = {
        "data_source": "sample_dataset.csv",
        "load_parameters": {
            "delimiter": ",",
            "header": True
        },
        "dependent_variable": "y",
        "independent_variables": ["x", "category"],
        "missing_strategy": "mean",
        "normalization_method": "z-score",
        "outlier_method": "iqr",
        "outlier_threshold": 1.5,
        "correlation_method": "pearson",
        "report_format": "html"
    }
    
    print(f"Analyzing data from source: {analysis_params['data_source']}")
    print(f"Dependent variable: {analysis_params['dependent_variable']}")
    print(f"Independent variables: {', '.join(analysis_params['independent_variables'])}")
    
    # Execute the data analysis plan
    print("\nExecuting data analysis plan...")
    result = await executor.execute_plan(plan_id, analysis_params)
    
    if result.get("success", False):
        # Display analysis results
        statistics = result.get("variables", {}).get("analysis_results", {}).get("statistics", {})
        report = result.get("variables", {}).get("final_report", {})
        
        print("\nAnalysis completed!")
        print(f"Statistics: {json.dumps(statistics, indent=2)}")
        print(f"Report: {json.dumps(report, indent=2) if report else 'No report generated'}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

async def main():
    # Create the plan database
    db = PlanDatabase()
    
    # Register all tools
    await register_tools(db)
    
    # Store example plans in the database
    main_plan_ids = store_plans_in_database(db)
    
    # Create a plan executor
    executor = GenericPlanExecutor(db)
    
    # Print available plans
    print("\nAvailable plans in database:")
    for domain, plan_id in main_plan_ids.items():
        print(f"  {domain}: {plan_id}")
    
    # Ask the user which demo to run
    print("\nWhich demo would you like to run?")
    print("1. Sudoku Solver")
    print("2. Flight Booking")
    print("3. Data Analysis")
    print("4. Run All Demos")
    
    try:
        choice = int(input("Enter choice (1-4): "))
        
        if choice == 1 or choice == 4:
            await run_sudoku_demo(executor, main_plan_ids["sudoku"])
        
        if choice == 2 or choice == 4:
            await run_flight_booking_demo(executor, main_plan_ids["flight"])
        
        if choice == 3 or choice == 4:
            await run_data_analysis_demo(executor, main_plan_ids["data_analysis"])
            
    except ValueError:
        print("Invalid choice. Please enter a number between 1 and 4.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())