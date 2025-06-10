"""
plan_examples.py - Example plans for different domains

This file demonstrates how the same generic plan engine can execute
plans for entirely different domains: Sudoku solving, flight booking,
data analysis, etc.
"""

import json

def create_sudoku_plan():
    """Create a Sudoku solver plan"""
    
    # Validation subplan for Sudoku
    validation_plan = {
        "id": "sudoku_validation_001",
        "title": "Sudoku Validation Plan",
        "description": "Validates a Sudoku puzzle",
        "domains": ["sudoku", "validation"],
        "variables": {},
        "steps": [
            {
                "id": "parse_step",
                "type": "tool",
                "title": "Parse Sudoku string",
                "tool": "sudoku_parse",
                "args": {"puzzle": "${puzzle}"},
                "result_variable": "parsed_puzzle"
            }
        ]
    }
    
    # Solving techniques subplan
    techniques_plan = {
        "id": "sudoku_techniques_001",
        "title": "Sudoku Solving Techniques",
        "description": "Applies standard solving techniques",
        "domains": ["sudoku", "solving"],
        "variables": {},
        "steps": [
            {
                "id": "naked_single",
                "type": "tool",
                "title": "Apply naked single technique",
                "tool": "sudoku_technique",
                "args": {
                    "grid": "${grid}",
                    "technique": "naked_single"
                },
                "result_variable": "naked_single_result"
            },
            {
                "id": "hidden_single",
                "type": "tool",
                "title": "Apply hidden single technique",
                "tool": "sudoku_technique",
                "args": {
                    "grid": "${naked_single_result.new_grid}",
                    "technique": "hidden_single"
                },
                "result_variable": "hidden_single_result",
                "depends_on": ["naked_single"]
            }
        ]
    }
    
    # Backtracking solver subplan
    backtracking_plan = {
        "id": "sudoku_backtracking_001",
        "title": "Sudoku Backtracking Solver",
        "description": "Solves Sudoku using backtracking algorithm",
        "domains": ["sudoku", "algorithm"],
        "variables": {},
        "steps": [
            {
                "id": "backtrack_step",
                "type": "tool",
                "title": "Solve using backtracking",
                "tool": "sudoku_backtrack",
                "args": {"grid": "${grid}"},
                "result_variable": "solution"
            }
        ]
    }
    
    # Main Sudoku solver plan
    main_sudoku_plan = {
        "id": "sudoku_solver_001",
        "title": "Complete Sudoku Solver",
        "description": "Comprehensive Sudoku puzzle solver",
        "domains": ["sudoku", "puzzle", "solver"],
        "variables": {},
        "steps": [
            {
                "id": "validate_step",
                "type": "subplan",
                "title": "Validate puzzle",
                "plan_id": "sudoku_validation_001",
                "args": {"puzzle": "${puzzle}"},
                "result_variable": "validation"
            },
            {
                "id": "check_validity",
                "type": "logic",
                "title": "Check if puzzle is valid",
                "metadata": {
                    "check": "validation.parsed_puzzle.valid"
                },
                "depends_on": ["validate_step"]
            },
            {
                "id": "techniques_step",
                "type": "subplan",
                "title": "Apply solving techniques",
                "plan_id": "sudoku_techniques_001",
                "args": {"grid": "${validation.parsed_puzzle.grid}"},
                "result_variable": "techniques_result",
                "depends_on": ["check_validity"]
            },
            {
                "id": "check_progress",
                "type": "logic",
                "title": "Check solving progress",
                "metadata": {
                    "check": "techniques_result.hidden_single_result.changed_cells"
                },
                "depends_on": ["techniques_step"]
            },
            {
                "id": "backtracking_step",
                "type": "subplan",
                "title": "Apply backtracking algorithm",
                "plan_id": "sudoku_backtracking_001",
                "args": {"grid": "${techniques_result.hidden_single_result.new_grid}"},
                "result_variable": "final_solution",
                "depends_on": ["check_progress"]
            },
            {
                "id": "format_solution",
                "type": "tool",
                "title": "Format final solution",
                "tool": "format_solution",
                "args": {"solution": "${final_solution.solution}"},
                "result_variable": "formatted_solution",
                "depends_on": ["backtracking_step"]
            }
        ]
    }
    
    return {
        "main": main_sudoku_plan,
        "validation": validation_plan,
        "techniques": techniques_plan,
        "backtracking": backtracking_plan
    }

def create_flight_booking_plan():
    """Create a flight booking plan"""
    
    # Search subplan
    search_plan = {
        "id": "flight_search_001",
        "title": "Flight Search Plan",
        "description": "Searches for available flights",
        "domains": ["travel", "flights", "search"],
        "variables": {},
        "steps": [
            {
                "id": "validate_dates",
                "type": "tool",
                "title": "Validate travel dates",
                "tool": "validate_dates",
                "args": {
                    "departure_date": "${departure_date}",
                    "return_date": "${return_date}"
                },
                "result_variable": "validated_dates"
            },
            {
                "id": "search_flights",
                "type": "tool",
                "title": "Search for flights",
                "tool": "search_flights",
                "args": {
                    "origin": "${origin}",
                    "destination": "${destination}",
                    "departure_date": "${validated_dates.departure}",
                    "return_date": "${validated_dates.return}",
                    "passengers": "${passengers}"
                },
                "result_variable": "available_flights",
                "depends_on": ["validate_dates"]
            },
            {
                "id": "filter_flights",
                "type": "tool",
                "title": "Filter flights by preferences",
                "tool": "filter_flights",
                "args": {
                    "flights": "${available_flights.flights}",
                    "preferences": "${preferences}"
                },
                "result_variable": "filtered_flights",
                "depends_on": ["search_flights"]
            }
        ]
    }
    
    # Pricing subplan
    pricing_plan = {
        "id": "flight_pricing_001",
        "title": "Flight Pricing Plan",
        "description": "Gets pricing for selected flights",
        "domains": ["travel", "flights", "pricing"],
        "variables": {},
        "steps": [
            {
                "id": "get_pricing",
                "type": "tool",
                "title": "Get pricing for selected flight",
                "tool": "get_flight_pricing",
                "args": {"flight": "${selected_flight}"},
                "result_variable": "flight_pricing"
            },
            {
                "id": "check_promotions",
                "type": "tool",
                "title": "Check for applicable promotions",
                "tool": "check_promotions",
                "args": {
                    "flight": "${selected_flight}",
                    "user": "${user_profile}"
                },
                "result_variable": "promotions"
            },
            {
                "id": "calculate_final_price",
                "type": "tool",
                "title": "Calculate final price",
                "tool": "calculate_price",
                "args": {
                    "base_price": "${flight_pricing.price}",
                    "promotions": "${promotions.available_promotions}",
                    "fees": "${flight_pricing.fees}"
                },
                "result_variable": "final_pricing",
                "depends_on": ["get_pricing", "check_promotions"]
            }
        ]
    }
    
    # Booking subplan
    booking_plan = {
        "id": "flight_booking_001",
        "title": "Flight Booking Plan",
        "description": "Books the selected flight",
        "domains": ["travel", "flights", "booking"],
        "variables": {},
        "steps": [
            {
                "id": "validate_payment",
                "type": "tool",
                "title": "Validate payment information",
                "tool": "validate_payment",
                "args": {"payment_details": "${payment_details}"},
                "result_variable": "validated_payment"
            },
            {
                "id": "reserve_seats",
                "type": "tool",
                "title": "Reserve seats",
                "tool": "reserve_seats",
                "args": {
                    "flight": "${selected_flight}",
                    "passengers": "${passengers}"
                },
                "result_variable": "seat_reservation",
                "depends_on": ["validate_payment"]
            },
            {
                "id": "process_payment",
                "type": "tool",
                "title": "Process payment",
                "tool": "process_payment",
                "args": {
                    "payment_details": "${validated_payment.details}",
                    "amount": "${final_price}",
                    "reservation": "${seat_reservation.reservation_id}"
                },
                "result_variable": "payment_result",
                "depends_on": ["reserve_seats"]
            },
            {
                "id": "confirm_booking",
                "type": "tool",
                "title": "Confirm booking",
                "tool": "confirm_booking",
                "args": {
                    "reservation": "${seat_reservation.reservation_id}",
                    "payment": "${payment_result.transaction_id}"
                },
                "result_variable": "booking_confirmation",
                "depends_on": ["process_payment"]
            }
        ]
    }
    
    # Main flight booking process plan
    main_booking_plan = {
        "id": "flight_booking_process_001",
        "title": "Complete Flight Booking Process",
        "description": "End-to-end flight booking process",
        "domains": ["travel", "flights", "booking", "process"],
        "variables": {
            "preferences": {"max_stops": 1, "preferred_airlines": []},
            "passengers": 1
        },
        "steps": [
            {
                "id": "search_step",
                "type": "subplan",
                "title": "Search for flights",
                "plan_id": "flight_search_001",
                "args": {
                    "origin": "${origin}",
                    "destination": "${destination}",
                    "departure_date": "${departure_date}",
                    "return_date": "${return_date}",
                    "passengers": "${passengers}",
                    "preferences": "${preferences}"
                },
                "result_variable": "search_results"
            },
            {
                "id": "select_flight",
                "type": "logic",
                "title": "Select flight based on criteria",
                "metadata": {
                    "selection_method": "best_match",
                    "criteria": ["price", "duration", "stops"]
                },
                "depends_on": ["search_step"]
            },
            {
                "id": "pricing_step",
                "type": "subplan",
                "title": "Get pricing for selected flight",
                "plan_id": "flight_pricing_001",
                "args": {
                    "selected_flight": "${search_results.filtered_flights[0]}",
                    "user_profile": "${user_profile}"
                },
                "result_variable": "pricing_results",
                "depends_on": ["select_flight"]
            },
            {
                "id": "review_pricing",
                "type": "logic",
                "title": "Review pricing information",
                "metadata": {
                    "review_criteria": ["final_price", "available_promotions"]
                },
                "depends_on": ["pricing_step"]
            },
            {
                "id": "booking_step",
                "type": "subplan",
                "title": "Complete booking process",
                "plan_id": "flight_booking_001",
                "args": {
                    "selected_flight": "${search_results.filtered_flights[0]}",
                    "passengers": "${passengers}",
                    "payment_details": "${payment_details}",
                    "final_price": "${pricing_results.final_pricing.total}"
                },
                "result_variable": "booking_results",
                "depends_on": ["review_pricing"]
            },
            {
                "id": "generate_itinerary",
                "type": "tool",
                "title": "Generate travel itinerary",
                "tool": "generate_itinerary",
                "args": {
                    "booking": "${booking_results.booking_confirmation}",
                    "passenger_info": "${passenger_info}",
                    "flight_details": "${search_results.filtered_flights[0]}"
                },
                "result_variable": "itinerary",
                "depends_on": ["booking_step"]
            },
            {
                "id": "send_confirmation",
                "type": "tool",
                "title": "Send confirmation email",
                "tool": "send_email",
                "args": {
                    "to": "${user_profile.email}",
                    "subject": "Flight Booking Confirmation",
                    "content": "Your flight has been booked successfully.",
                    "attachments": ["${itinerary.pdf_url}"]
                },
                "result_variable": "email_sent",
                "depends_on": ["generate_itinerary"]
            }
        ]
    }
    
    return {
        "main": main_booking_plan,
        "search": search_plan,
        "pricing": pricing_plan,
        "booking": booking_plan
    }

def create_data_analysis_plan():
    """Create a data analysis plan"""
    
    # Data loading subplan
    data_loading_plan = {
        "id": "data_loading_001",
        "title": "Data Loading Plan",
        "description": "Loads and validates data from various sources",
        "domains": ["data", "analysis", "loading"],
        "variables": {},
        "steps": [
            {
                "id": "validate_source",
                "type": "tool",
                "title": "Validate data source",
                "tool": "validate_data_source",
                "args": {"source": "${data_source}"},
                "result_variable": "validated_source"
            },
            {
                "id": "load_data",
                "type": "tool",
                "title": "Load data from source",
                "tool": "load_data",
                "args": {
                    "source": "${validated_source.source}",
                    "parameters": "${load_parameters}"
                },
                "result_variable": "raw_data",
                "depends_on": ["validate_source"]
            },
            {
                "id": "validate_data",
                "type": "tool",
                "title": "Validate loaded data",
                "tool": "validate_data",
                "args": {"data": "${raw_data.data}"},
                "result_variable": "validated_data",
                "depends_on": ["load_data"]
            }
        ]
    }
    
    # Data cleaning subplan
    data_cleaning_plan = {
        "id": "data_cleaning_001",
        "title": "Data Cleaning Plan",
        "description": "Cleans and prepares data for analysis",
        "domains": ["data", "analysis", "cleaning"],
        "variables": {},
        "steps": [
            {
                "id": "handle_missing_values",
                "type": "tool",
                "title": "Handle missing values",
                "tool": "handle_missing",
                "args": {
                    "data": "${data}",
                    "strategy": "${missing_strategy}"
                },
                "result_variable": "missing_handled"
            },
            {
                "id": "normalize_data",
                "type": "tool",
                "title": "Normalize data",
                "tool": "normalize_data",
                "args": {
                    "data": "${missing_handled.data}",
                    "method": "${normalization_method}"
                },
                "result_variable": "normalized_data",
                "depends_on": ["handle_missing_values"]
            },
            {
                "id": "remove_outliers",
                "type": "tool",
                "title": "Remove outliers",
                "tool": "remove_outliers",
                "args": {
                    "data": "${normalized_data.data}",
                    "method": "${outlier_method}",
                    "threshold": "${outlier_threshold}"
                },
                "result_variable": "cleaned_data",
                "depends_on": ["normalize_data"]
            }
        ]
    }
    
    # Analysis subplan
    analysis_plan = {
        "id": "data_analysis_001",
        "title": "Data Analysis Plan",
        "description": "Performs statistical analysis on data",
        "domains": ["data", "analysis", "statistics"],
        "variables": {},
        "steps": [
            {
                "id": "descriptive_stats",
                "type": "tool",
                "title": "Calculate descriptive statistics",
                "tool": "descriptive_statistics",
                "args": {"data": "${data}"},
                "result_variable": "statistics"
            },
            {
                "id": "correlation_analysis",
                "type": "tool",
                "title": "Perform correlation analysis",
                "tool": "correlation_analysis",
                "args": {
                    "data": "${data}",
                    "method": "${correlation_method}"
                },
                "result_variable": "correlations",
                "depends_on": ["descriptive_stats"]
            },
            {
                "id": "regression_analysis",
                "type": "tool",
                "title": "Perform regression analysis",
                "tool": "regression_analysis",
                "args": {
                    "data": "${data}",
                    "dependent": "${dependent_variable}",
                    "independents": "${independent_variables}"
                },
                "result_variable": "regression_results",
                "depends_on": ["correlation_analysis"]
            }
        ]
    }
    
    # Visualization subplan
    visualization_plan = {
        "id": "data_visualization_001",
        "title": "Data Visualization Plan",
        "description": "Creates visualizations of data and analysis results",
        "domains": ["data", "visualization"],
        "variables": {},
        "steps": [
            {
                "id": "create_summary_charts",
                "type": "tool",
                "title": "Create summary charts",
                "tool": "create_charts",
                "args": {
                    "data": "${data}",
                    "statistics": "${statistics}",
                    "chart_types": ["histogram", "boxplot", "scatter"]
                },
                "result_variable": "summary_charts"
            },
            {
                "id": "create_correlation_heatmap",
                "type": "tool",
                "title": "Create correlation heatmap",
                "tool": "create_heatmap",
                "args": {"correlation_data": "${correlations.data}"},
                "result_variable": "correlation_heatmap",
                "depends_on": ["create_summary_charts"]
            },
            {
                "id": "create_regression_plots",
                "type": "tool",
                "title": "Create regression plots",
                "tool": "create_regression_plots",
                "args": {"regression_data": "${regression_results}"},
                "result_variable": "regression_plots",
                "depends_on": ["create_correlation_heatmap"]
            },
            {
                "id": "compile_visualization_report",
                "type": "tool",
                "title": "Compile visualization report",
                "tool": "compile_report",
                "args": {
                    "charts": "${summary_charts.charts}",
                    "heatmap": "${correlation_heatmap.chart}",
                    "regression_plots": "${regression_plots.plots}",
                    "format": "${report_format}"
                },
                "result_variable": "visualization_report",
                "depends_on": ["create_regression_plots"]
            }
        ]
    }
    
    # Main data analysis plan
    main_analysis_plan = {
        "id": "complete_data_analysis_001",
        "title": "Complete Data Analysis Workflow",
        "description": "End-to-end data analysis workflow",
        "domains": ["data", "analysis", "workflow"],
        "variables": {
            "missing_strategy": "mean",
            "normalization_method": "z-score",
            "outlier_method": "iqr",
            "outlier_threshold": 1.5,
            "correlation_method": "pearson",
            "report_format": "html"
        },
        "steps": [
            {
                "id": "loading_step",
                "type": "subplan",
                "title": "Load data",
                "plan_id": "data_loading_001",
                "args": {
                    "data_source": "${data_source}",
                    "load_parameters": "${load_parameters}"
                },
                "result_variable": "loading_results"
            },
            {
                "id": "cleaning_step",
                "type": "subplan",
                "title": "Clean data",
                "plan_id": "data_cleaning_001",
                "args": {
                    "data": "${loading_results.validated_data.data}",
                    "missing_strategy": "${missing_strategy}",
                    "normalization_method": "${normalization_method}",
                    "outlier_method": "${outlier_method}",
                    "outlier_threshold": "${outlier_threshold}"
                },
                "result_variable": "cleaning_results",
                "depends_on": ["loading_step"]
            },
            {
                "id": "analysis_step",
                "type": "subplan",
                "title": "Analyze data",
                "plan_id": "data_analysis_001",
                "args": {
                    "data": "${cleaning_results.cleaned_data.data}",
                    "correlation_method": "${correlation_method}",
                    "dependent_variable": "${dependent_variable}",
                    "independent_variables": "${independent_variables}"
                },
                "result_variable": "analysis_results",
                "depends_on": ["cleaning_step"]
            },
            {
                "id": "visualization_step",
                "type": "subplan",
                "title": "Create visualizations",
                "plan_id": "data_visualization_001",
                "args": {
                    "data": "${cleaning_results.cleaned_data.data}",
                    "statistics": "${analysis_results.statistics}",
                    "correlations": "${analysis_results.correlations}",
                    "regression_results": "${analysis_results.regression_results}",
                    "report_format": "${report_format}"
                },
                "result_variable": "visualization_results",
                "depends_on": ["analysis_step"]
            },
            {
                "id": "generate_final_report",
                "type": "tool",
                "title": "Generate final analysis report",
                "tool": "generate_analysis_report",
                "args": {
                    "data_summary": "${loading_results.validated_data.summary}",
                    "cleaning_summary": "${cleaning_results.cleaned_data.summary}",
                    "analysis_results": "${analysis_results}",
                    "visualizations": "${visualization_results.visualization_report}",
                    "format": "${report_format}"
                },
                "result_variable": "final_report",
                "depends_on": ["visualization_step"]
            }
        ]
    }
    
    return {
        "main": main_analysis_plan,
        "loading": data_loading_plan,
        "cleaning": data_cleaning_plan,
        "analysis": analysis_plan,
        "visualization": visualization_plan
    }

def store_plans_in_database(db):
    """Store all example plans in the database"""
    # Store Sudoku plans
    sudoku_plans = create_sudoku_plan()
    for plan_key, plan in sudoku_plans.items():
        db.store_plan(plan["id"], plan)
    
    # Store flight booking plans
    flight_plans = create_flight_booking_plan()
    for plan_key, plan in flight_plans.items():
        db.store_plan(plan["id"], plan)
    
    # Store data analysis plans
    data_plans = create_data_analysis_plan()
    for plan_key, plan in data_plans.items():
        db.store_plan(plan["id"], plan)
    
    return {
        "sudoku": sudoku_plans["main"]["id"],
        "flight": flight_plans["main"]["id"],
        "data_analysis": data_plans["main"]["id"]
    }