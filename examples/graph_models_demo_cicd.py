#!/usr/bin/env python3
# examples/graph_models_demo_e2e_cicd.py
"""
API & UI Testing Execution Demo: Graph Models for Test Automation
================================================================

This demo showcases how chuk_ai_planner graph models can represent
comprehensive test automation workflows including:

1. API testing (REST endpoints, authentication, data validation)
2. UI testing (browser automation, user interactions, assertions)
3. Test execution orchestration and reporting
4. Parallel test execution with dependencies
5. Test result aggregation and analysis

Scenario: E-commerce platform testing with API and UI test suites

Run with: python demo_testing_execution.py
"""

import json
from datetime import datetime, timezone, timedelta

# Core imports
from chuk_ai_planner.models import (
    NodeKind, SessionNode, PlanNode, PlanStep,
    UserMessage, AssistantMessage, ToolCall, TaskRun, Summary
)

from chuk_ai_planner.models.edges import (
    EdgeKind, ParentChildEdge, NextEdge, PlanEdge, StepEdge, GraphEdge
)

from chuk_ai_planner.store.memory import InMemoryGraphStore

# Updated import - use the corrected visualization module
try:
    from chuk_ai_planner.utils.visualization import print_graph_structure
except ImportError:
    print("⚠️ Visualization module not available - continuing without graph structure display")
    def print_graph_structure(graph):
        print("📊 Graph structure visualization not available")
        print(f"   Nodes: {len(graph.nodes)}")
        print(f"   Edges: {len(graph.edges)}")

def create_testing_session():
    """Create a comprehensive testing execution session."""
    print("🧪 Creating E-commerce Testing Session")
    print("=" * 60)
    
    graph = InMemoryGraphStore()
    
    # Testing session
    session = SessionNode(data={
        "session_type": "automated_testing",
        "target_application": "ecommerce_platform",
        "test_environment": "staging",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "test_runner": "chuk_test_framework_v1.0"
    })
    
    # Test request from QA team
    test_request = UserMessage(data={
        "content": "Please run the full test suite for the e-commerce platform before production deployment. Include API tests, UI tests, and performance validation.",
        "priority": "high",
        "requested_by": "qa_team",
        "deadline": "end_of_day"
    })
    
    # Test framework response
    test_response = AssistantMessage(data={
        "content": "I'll execute the comprehensive test suite including API tests, UI automation, and performance validation. Creating execution plan...",
        "estimated_duration": "45_minutes",
        "test_types": "api,ui,performance"
    })
    
    # Master test execution plan
    test_plan = PlanNode(data={
        "title": "E-commerce Platform Test Suite",
        "description": "Comprehensive automated testing including API, UI, and performance tests",
        "target_env": "https://staging.ecommerce.com",
        "parallel_execution": "true",
        "max_workers": "4"
    })
    
    # Test execution steps
    test_steps = [
        # Environment setup
        PlanStep(data={
            "index": "1",
            "description": "Setup test environment and validate connectivity",
            "step_type": "setup",
            "parallel_group": "setup",
            "timeout": "300"
        }),
        
        # API testing suite
        PlanStep(data={
            "index": "2",
            "description": "Execute API test suite",
            "step_type": "api_testing",
            "parallel_group": "api",
            "test_count": "25"
        }),
        
        # UI testing suite  
        PlanStep(data={
            "index": "3",
            "description": "Execute UI automation test suite",
            "step_type": "ui_testing",
            "parallel_group": "ui",
            "test_count": "15"
        }),
        
        # Performance testing
        PlanStep(data={
            "index": "4", 
            "description": "Run performance and load tests",
            "step_type": "performance_testing",
            "parallel_group": "performance",
            "test_count": "8"
        }),
        
        # Security testing
        PlanStep(data={
            "index": "5",
            "description": "Execute security vulnerability scans",
            "step_type": "security_testing",
            "parallel_group": "security",
            "test_count": "12"
        }),
        
        # Test result aggregation
        PlanStep(data={
            "index": "6",
            "description": "Aggregate results and generate test report",
            "step_type": "reporting",
            "parallel_group": "reporting",
            "depends_on": "2,3,4,5"
        })
    ]
    
    # Tool calls for each test step
    test_tools = [
        # Environment setup
        ToolCall(data={
            "name": "setup_test_environment",
            "args": '{"target_url": "https://staging.ecommerce.com", "browser": "chrome", "headless": true}',
            "result": '{"status": "ready", "browser_version": "119.0", "connectivity": "ok", "test_data_loaded": true}',
            "execution_time": "45.2",
            "cached": "false"
        }),
        
        # API testing
        ToolCall(data={
            "name": "run_api_tests",
            "args": '{"test_suite": "api_regression", "parallel": true, "workers": 4}',
            "result": '{"total_tests": 25, "passed": 23, "failed": 2, "skipped": 0, "duration": "127.8", "coverage": "94%"}',
            "execution_time": "127.8",
            "cached": "false"
        }),
        
        # UI testing
        ToolCall(data={
            "name": "run_ui_tests", 
            "args": '{"test_suite": "ui_regression", "browser": "chrome", "resolution": "1920x1080"}',
            "result": '{"total_tests": 15, "passed": 14, "failed": 1, "skipped": 0, "duration": "298.5", "screenshots": 45}',
            "execution_time": "298.5",
            "cached": "false"
        }),
        
        # Performance testing
        ToolCall(data={
            "name": "run_performance_tests",
            "args": '{"test_type": "load", "users": 100, "duration": "300s"}',
            "result": '{"total_tests": 8, "passed": 7, "failed": 1, "avg_response_time": "245ms", "max_response_time": "1.2s", "throughput": "150 req/s"}',
            "execution_time": "315.7",
            "cached": "false"
        }),
        
        # Security testing
        ToolCall(data={
            "name": "run_security_tests",
            "args": '{"scan_type": "comprehensive", "include_auth": true}',
            "result": '{"total_tests": 12, "passed": 10, "failed": 2, "vulnerabilities_found": 3, "severity": "medium"}',
            "execution_time": "89.3",
            "cached": "false"
        }),
        
        # Reporting
        ToolCall(data={
            "name": "generate_test_report",
            "args": '{"format": "html", "include_screenshots": true, "send_email": true}',
            "result": '{"report_generated": true, "report_size": "2.4MB", "charts": 8, "email_sent": true, "recipients": 5}',
            "execution_time": "23.1",
            "cached": "false"
        })
    ]
    
    # Task execution results
    task_runs = [
        TaskRun(data={
            "success": "true",
            "step_name": "environment_setup",
            "execution_time": "45.2",
            "memory_used": "128MB",
            "browser_instances": "1",
            "setup_errors": "0"
        }),
        
        TaskRun(data={
            "success": "true", 
            "step_name": "api_testing",
            "execution_time": "127.8",
            "memory_used": "256MB",
            "parallel_workers": "4",
            "api_calls_made": "147",
            "assertion_failures": "2"
        }),
        
        TaskRun(data={
            "success": "true",
            "step_name": "ui_testing",
            "execution_time": "298.5", 
            "memory_used": "512MB",
            "page_loads": "45",
            "element_interactions": "128",
            "screenshot_count": "45"
        }),
        
        TaskRun(data={
            "success": "true",
            "step_name": "performance_testing",
            "execution_time": "315.7",
            "memory_used": "384MB",
            "virtual_users": "100",
            "requests_sent": "45000",
            "errors": "23"
        }),
        
        TaskRun(data={
            "success": "true",
            "step_name": "security_testing",
            "execution_time": "89.3",
            "memory_used": "192MB",
            "scans_completed": "12",
            "vulnerabilities": "3",
            "false_positives": "1"
        }),
        
        TaskRun(data={
            "success": "true",
            "step_name": "report_generation",
            "execution_time": "23.1", 
            "memory_used": "96MB",
            "report_pages": "12",
            "charts_generated": "8",
            "email_notifications": "5"
        })
    ]
    
    # Test summaries for each step
    test_summaries = [
        Summary(data={
            "content": "Test environment setup completed successfully. Browser ready, connectivity verified.",
            "setup_time": "45.2s",
            "status": "ready"
        }),
        
        Summary(data={
            "content": "API tests: 23/25 passed (92% success rate). 2 failures in payment validation.",
            "test_success_rate": "92%",
            "critical_failures": "2"
        }),
        
        Summary(data={
            "content": "UI tests: 14/15 passed (93% success rate). 1 failure in checkout flow.",
            "test_success_rate": "93%",
            "screenshots_captured": "45"
        }),
        
        Summary(data={
            "content": "Performance tests: 7/8 passed. Average response time 245ms, one timeout issue.",
            "avg_response_time": "245ms",
            "throughput": "150 req/s"
        }),
        
        Summary(data={
            "content": "Security scan: 10/12 passed. Found 3 medium-severity vulnerabilities requiring attention.",
            "vulnerabilities": "3",
            "max_severity": "medium"
        }),
        
        Summary(data={
            "content": "Test report generated successfully. HTML report created and emailed to QA team.",
            "report_size": "2.4MB",
            "recipients": "5"
        })
    ]
    
    # Add all nodes to graph
    all_nodes = ([session, test_request, test_response, test_plan] + 
                test_steps + test_tools + task_runs + test_summaries)
    
    for node in all_nodes:
        graph.add_node(node)
    
    # Create comprehensive edge structure
    edges = [
        # Session structure
        ParentChildEdge(src=session.id, dst=test_request.id),
        ParentChildEdge(src=session.id, dst=test_response.id),
        ParentChildEdge(src=session.id, dst=test_plan.id),
        NextEdge(src=test_request.id, dst=test_response.id),
        
        # Plan structure
        *[ParentChildEdge(src=test_plan.id, dst=step.id) for step in test_steps],
        
        # Test dependencies (only setup → others, then others → reporting)
        StepEdge(src=test_steps[0].id, dst=test_steps[1].id),  # setup → api
        StepEdge(src=test_steps[0].id, dst=test_steps[2].id),  # setup → ui
        StepEdge(src=test_steps[0].id, dst=test_steps[3].id),  # setup → performance
        StepEdge(src=test_steps[0].id, dst=test_steps[4].id),  # setup → security
        StepEdge(src=test_steps[1].id, dst=test_steps[5].id),  # api → reporting
        StepEdge(src=test_steps[2].id, dst=test_steps[5].id),  # ui → reporting
        StepEdge(src=test_steps[3].id, dst=test_steps[5].id),  # performance → reporting
        StepEdge(src=test_steps[4].id, dst=test_steps[5].id),  # security → reporting
        
        # Tool execution chains
        *[PlanEdge(src=test_steps[i].id, dst=test_tools[i].id) 
          for i in range(len(test_steps))],
        *[ParentChildEdge(src=test_tools[i].id, dst=task_runs[i].id) 
          for i in range(len(test_tools))],
        *[ParentChildEdge(src=task_runs[i].id, dst=test_summaries[i].id) 
          for i in range(len(test_summaries))],
        
        # Parallel execution groups (custom edges)
        GraphEdge(kind=EdgeKind.CUSTOM, src=test_steps[1].id, dst=test_steps[2].id,
                 data={"execution_type": "parallel", "group": "test_execution"}),
        GraphEdge(kind=EdgeKind.CUSTOM, src=test_steps[1].id, dst=test_steps[3].id,
                 data={"execution_type": "parallel", "group": "test_execution"}),
        GraphEdge(kind=EdgeKind.CUSTOM, src=test_steps[1].id, dst=test_steps[4].id,
                 data={"execution_type": "parallel", "group": "test_execution"}),
    ]
    
    for edge in edges:
        graph.add_edge(edge)
    
    return graph, session, test_plan, test_steps, test_tools, task_runs, test_summaries

def analyze_test_execution(graph, session, test_plan, steps, tools, task_runs, summaries):
    """Analyze the test execution results."""
    print("\n📊 Test Execution Analysis")
    print("=" * 60)
    
    # Overall execution metrics
    total_execution_time = sum(float(tr.data.get('execution_time', '0')) for tr in task_runs)
    total_memory_used = sum(
        float(tr.data.get('memory_used', '0MB').replace('MB', '')) 
        for tr in task_runs
    )
    successful_steps = sum(1 for tr in task_runs if tr.data.get('success') == 'true')
    
    print(f"🎯 Execution Summary:")
    print(f"   Total execution time: {total_execution_time:.1f} seconds ({total_execution_time/60:.1f} minutes)")
    print(f"   Peak memory usage: {total_memory_used:.0f} MB")
    print(f"   Successful steps: {successful_steps}/{len(task_runs)}")
    print(f"   Overall success rate: {successful_steps/len(task_runs)*100:.1f}%")
    
    # Test results breakdown
    print(f"\n🧪 Test Results Breakdown:")
    
    # Extract test results from tool outputs
    test_results = []
    for tool in tools[1:5]:  # Skip setup, include test tools
        result = json.loads(tool.data['result'])
        test_results.append({
            'name': tool.data['name'],
            'total': result.get('total_tests', 0),
            'passed': result.get('passed', 0),
            'failed': result.get('failed', 0),
            'duration': float(tool.data['execution_time'])
        })
    
    total_tests = sum(tr['total'] for tr in test_results)
    total_passed = sum(tr['passed'] for tr in test_results)
    total_failed = sum(tr['failed'] for tr in test_results)
    
    for result in test_results:
        name = result['name'].replace('run_', '').replace('_tests', '').upper()
        success_rate = (result['passed'] / result['total'] * 100) if result['total'] > 0 else 0
        print(f"   {name:12} {result['passed']:2}/{result['total']:2} tests passed ({success_rate:5.1f}%) - {result['duration']:6.1f}s")
    
    print(f"   {'TOTAL':12} {total_passed:2}/{total_tests:2} tests passed ({total_passed/total_tests*100:5.1f}%)")
    
    # Performance analysis
    print(f"\n⚡ Performance Analysis:")
    slowest_step = max(task_runs, key=lambda tr: float(tr.data.get('execution_time', '0')))
    fastest_step = min(task_runs, key=lambda tr: float(tr.data.get('execution_time', '0')))
    
    print(f"   Slowest step: {slowest_step.data.get('step_name')} ({slowest_step.data.get('execution_time')}s)")
    print(f"   Fastest step: {fastest_step.data.get('step_name')} ({fastest_step.data.get('execution_time')}s)")
    
    # Parallel execution analysis
    api_time = float(task_runs[1].data['execution_time'])
    ui_time = float(task_runs[2].data['execution_time'])
    perf_time = float(task_runs[3].data['execution_time'])
    sec_time = float(task_runs[4].data['execution_time'])
    
    sequential_time = api_time + ui_time + perf_time + sec_time
    parallel_time = max(api_time, ui_time, perf_time, sec_time)
    time_saved = sequential_time - parallel_time
    
    print(f"   Sequential execution would take: {sequential_time:.1f}s")
    print(f"   Parallel execution took: {parallel_time:.1f}s")
    print(f"   Time saved by parallelization: {time_saved:.1f}s ({time_saved/sequential_time*100:.1f}%)")
    
    # Quality metrics
    print(f"\n🎯 Quality Metrics:")
    
    # API coverage (from tool results)
    api_result = json.loads(tools[1].data['result'])
    print(f"   API test coverage: {api_result.get('coverage', 'N/A')}")
    
    # Performance metrics
    perf_result = json.loads(tools[3].data['result'])
    print(f"   Average response time: {perf_result.get('avg_response_time', 'N/A')}")
    print(f"   Throughput: {perf_result.get('throughput', 'N/A')}")
    
    # Security issues
    sec_result = json.loads(tools[4].data['result'])
    print(f"   Security vulnerabilities: {sec_result.get('vulnerabilities_found', 0)} (severity: {sec_result.get('severity', 'N/A')})")

def demo_test_failure_analysis():
    """Demonstrate analysis of test failures."""
    print("\n🔍 Test Failure Analysis")
    print("=" * 60)
    
    print("🚨 Failed Test Details:")
    
    failures = [
        {
            "test_suite": "API Tests",
            "test_name": "Payment Validation",
            "failure_count": 2,
            "error_message": "HTTP 422: Invalid credit card format",
            "impact": "Critical - affects checkout flow"
        },
        {
            "test_suite": "UI Tests", 
            "test_name": "Checkout Flow",
            "failure_count": 1,
            "error_message": "Element not found: #submit-order-btn",
            "impact": "High - blocks order completion"
        },
        {
            "test_suite": "Performance Tests",
            "test_name": "Heavy Load Test",
            "failure_count": 1,
            "error_message": "Response timeout after 5 seconds",
            "impact": "Medium - affects user experience under load"
        },
        {
            "test_suite": "Security Tests",
            "test_name": "SQL Injection & XSS",
            "failure_count": 2,
            "error_message": "Potential vulnerabilities detected",
            "impact": "High - security risk"
        }
    ]
    
    for failure in failures:
        print(f"\n❌ {failure['test_suite']}: {failure['test_name']}")
        print(f"   Failures: {failure['failure_count']}")
        print(f"   Error: {failure['error_message']}")
        print(f"   Impact: {failure['impact']}")
    
    print(f"\n📋 Recommended Actions:")
    print(f"   1. Fix payment validation regex in API layer")
    print(f"   2. Update UI element selectors for checkout button")
    print(f"   3. Investigate database performance under load")
    print(f"   4. Implement input sanitization for security issues")
    print(f"   5. Re-run affected test suites after fixes")

def demo_parallel_execution_visualization():
    """Visualize the parallel test execution timeline."""
    print("\n⏱️ Parallel Execution Timeline")
    print("=" * 60)
    
    # Simulate execution timeline
    timeline = [
        {"step": "Environment Setup", "start": 0, "duration": 45.2, "status": "✅", "parallel_group": "setup"},
        {"step": "API Tests", "start": 45.2, "duration": 127.8, "status": "⚠️", "parallel_group": "tests"},
        {"step": "UI Tests", "start": 45.2, "duration": 298.5, "status": "⚠️", "parallel_group": "tests"},
        {"step": "Performance Tests", "start": 45.2, "duration": 315.7, "status": "⚠️", "parallel_group": "tests"},
        {"step": "Security Tests", "start": 45.2, "duration": 89.3, "status": "⚠️", "parallel_group": "tests"},
        {"step": "Report Generation", "start": 360.9, "duration": 23.1, "status": "✅", "parallel_group": "reporting"}
    ]
    
    print("📊 Execution Timeline (seconds):")
    print("   0    50   100  150  200  250  300  350  400")
    print("   |    |    |    |    |    |    |    |    |")
    
    for item in timeline:
        start_pos = int(item['start'] / 10)
        duration_pos = int(item['duration'] / 10)
        
        # Create visual timeline
        timeline_bar = " " * start_pos + "█" * max(1, duration_pos)
        status = item['status']
        
        print(f"{status} {item['step']:<20} {timeline_bar}")
        print(f"   {'':<20} {item['start']:>6.1f}s → {item['start'] + item['duration']:>6.1f}s ({item['duration']:>5.1f}s)")

def main():
    """Run the comprehensive testing demo."""
    print("🧪 API & UI Testing Execution - Graph Models Demo")
    print("=" * 80)
    
    try:
        # Create the testing workflow
        graph, session, plan, steps, tools, task_runs, summaries = create_testing_session()
        
        # Show graph structure
        print("\n🏗️ Test Execution Graph Structure")
        print("=" * 60)
        print_graph_structure(graph)
        
        # Analyze test execution
        analyze_test_execution(graph, session, plan, steps, tools, task_runs, summaries)
        
        # Demonstrate failure analysis
        demo_test_failure_analysis()
        
        # Show parallel execution timeline
        demo_parallel_execution_visualization()
        
        # Final summary
        print("\n" + "=" * 80)
        print("✅ Testing Execution Demo Complete!")
        print("\n🎯 Key Demonstrations:")
        print("   • Complex multi-stage test execution workflow")
        print("   • Parallel test execution with proper dependencies")
        print("   • Rich test result tracking and analysis")
        print("   • Test failure categorization and impact assessment")
        print("   • Performance metrics and optimization insights")
        print("   • Comprehensive test reporting and notifications")
        print("\n📊 Test Results Summary:")
        print("   • Total tests executed: 60 across 4 test suites")
        print("   • Overall success rate: 90% (54/60 tests passed)")
        print("   • Execution time: 6.4 minutes (with parallelization)")
        print("   • Time saved: 67% compared to sequential execution")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()