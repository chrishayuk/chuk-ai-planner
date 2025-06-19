#!/usr/bin/env python3
# examples/plan_agent_demo.py
"""
Final Plan Agent Demo
========================================
The four key providers showcased:
- OpenAI GPT-4o Mini (reliable baseline)
- Anthropic Claude 4 Sonnet (advanced reasoning) 
- Mistral Medium 2505 (multimodal + tools)
- IBM WatsonX Granite 3.3 (enterprise)
"""

import asyncio
import json
import os
import time
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

# Import the plan agent components
from chuk_ai_planner.agents.plan_agent import PlanAgent


# ============================================================================
# Custom Validator - More Flexible Than Coffee/Weather Only
# ============================================================================

def create_flexible_validator():
    """Create a flexible validator that doesn't restrict to just coffee/weather tools."""
    
    def validate_step(step: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate a single step in the plan - much more flexible."""
        
        # Check required fields
        if "title" not in step:
            return False, "Missing required field 'title'"
        if "tool" not in step:
            return False, "Missing required field 'tool'"
        
        # Check field types
        if not isinstance(step["title"], str):
            return False, "'title' must be a string"
        if not isinstance(step["tool"], str):
            return False, "'tool' must be a string"
        
        # Check optional fields
        if "args" in step and not isinstance(step["args"], dict):
            return False, "'args' must be an object"
        
        if "depends_on" in step:
            if not isinstance(step["depends_on"], list):
                return False, "'depends_on' must be an array"
            for dep in step["depends_on"]:
                if not isinstance(dep, int) or dep < 1:
                    return False, "'depends_on' must contain positive integers"
        
        # Accept ANY tool name - don't restrict to specific tools
        # This allows the model to be creative with tool selection
        
        return True, ""
    
    return validate_step


# ============================================================================
# Completely New System Prompt - No Examples That Could Confuse
# ============================================================================

CLEAN_SYSTEM_PROMPT = """
You are a professional AI assistant that creates structured plans from natural language requests.

Your task is to convert the user's specific request into a JSON plan that directly addresses what they asked for.

RESPONSE FORMAT (JSON only, no other text):
{
  "title": "Clear title that reflects the user's actual request",
  "description": "Brief description of what this plan accomplishes", 
  "steps": [
    {
      "title": "Specific step description",
      "tool": "appropriate_tool_name",
      "args": {"parameter": "value"},
      "depends_on": [step_numbers_if_needed],
      "rationale": "Why this step is necessary"
    }
  ],
  "estimated_duration": "realistic time estimate",
  "complexity": "simple|moderate|complex"
}

AVAILABLE TOOLS:
- weather: Get weather information
- calculator: Perform mathematical calculations
- search: Search for information online
- grind_beans: Grind coffee beans
- boil_water: Heat water
- brew_coffee: Brew coffee
- clean_station: Clean equipment
- send_email: Send email notifications

CRITICAL INSTRUCTIONS:
1. READ the user's request carefully
2. CREATE a plan that directly addresses their specific request
3. USE appropriate tools for their actual needs
4. DO NOT create generic plans unless specifically requested
5. MATCH the complexity to the actual task difficulty
6. PROVIDE realistic, actionable steps

Your response must be valid JSON that addresses the user's exact request.
"""


# ============================================================================
# Provider Configurations
# ============================================================================

@dataclass
class ProviderConfig:
    """Configuration for a specific provider."""
    name: str
    provider: str
    model: str
    display_name: str
    strengths: List[str]
    max_tokens: int

KEY_PROVIDERS = [
    ProviderConfig(
        name="openai_gpt4o_mini",
        provider="openai", 
        model="gpt-4o-mini",
        display_name="OpenAI GPT-4o Mini",
        strengths=["Reliable", "Fast", "Cost-effective"],
        max_tokens=3000
    ),
    ProviderConfig(
        name="claude4_sonnet", 
        provider="anthropic",
        model="claude-sonnet-4-20250514", 
        display_name="Claude 4 Sonnet",
        strengths=["Advanced reasoning", "Long context", "Safety"],
        max_tokens=4000
    ),
    ProviderConfig(
        name="mistral_medium",
        provider="mistral",
        model="mistral-medium-2505",
        display_name="Mistral Medium 2505", 
        strengths=["Multimodal", "Tool calling", "European"],
        max_tokens=3500
    ),
    ProviderConfig(
        name="watsonx_granite",
        provider="watsonx", 
        model="ibm/granite-3-3-8b-instruct",
        display_name="IBM WatsonX Granite 3.3",
        strengths=["Enterprise grade", "Transparent", "Compliance"],
        max_tokens=3000
    )
]


# ============================================================================
# Clean Test Scenarios
# ============================================================================

TEST_SCENARIOS = [
    {
        "name": "Simple Weather Request",
        "prompt": "What's the weather in London?",
        "expected_keywords": ["London", "weather"],
        "expected_tools": ["weather"]
    },
    {
        "name": "Math Problem",
        "prompt": "Calculate 15% tip on a $85 restaurant bill",
        "expected_keywords": ["15%", "tip", "85"],
        "expected_tools": ["calculator"]
    },
    {
        "name": "Research Task",
        "prompt": "Research current AI trends and summarize findings",
        "expected_keywords": ["AI", "research", "trends"],
        "expected_tools": ["search"]
    },
    {
        "name": "Multi-step Process",
        "prompt": "Check weather in Paris, search for local news, then email summary to manager@example.com",
        "expected_keywords": ["Paris", "weather", "news", "email", "manager@example.com"],
        "expected_tools": ["weather", "search", "send_email"]
    }
]


# ============================================================================
# Demo Functions
# ============================================================================

async def test_single_provider(provider_config: ProviderConfig, scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Test a single provider with a scenario and return detailed results."""
    
    try:
        agent = PlanAgent(
            system_prompt=CLEAN_SYSTEM_PROMPT,
            validate_step=create_flexible_validator(),
            provider=provider_config.provider,
            model=provider_config.model,
            max_tokens=provider_config.max_tokens,
            max_retries=3,
            temperature=0.3
        )
        
        print(f"   🤖 Testing {provider_config.display_name}...")
        
        start_time = time.time()
        plan = await agent.plan(scenario['prompt'])
        duration = time.time() - start_time
        
        # Analyze the plan
        title = plan.get('title', '')
        description = plan.get('description', '')
        steps = plan.get('steps', [])
        complexity = plan.get('complexity', 'unknown')
        
        # Check if it addresses the request
        content_to_check = f"{title} {description}".lower()
        addresses_request = any(
            keyword.lower() in content_to_check 
            for keyword in scenario['expected_keywords']
        )
        
        # Check tools used
        tools_used = [step.get('tool') for step in steps if step.get('tool')]
        expected_tools_found = any(
            tool in tools_used 
            for tool in scenario['expected_tools']
        )
        
        result = {
            'provider': provider_config.display_name,
            'success': True,
            'duration': duration,
            'title': title,
            'description': description,
            'steps_count': len(steps),
            'complexity': complexity,
            'addresses_request': addresses_request,
            'uses_expected_tools': expected_tools_found,
            'tools_used': tools_used,
            'steps': steps,
            'attempts': len(agent.get_history())
        }
        
        # Print results
        relevance_icon = "✅" if addresses_request else "❌"
        tools_icon = "✅" if expected_tools_found else "❌"
        
        print(f"      ✅ Success in {duration:.2f}s")
        print(f"      📋 Title: {title}")
        print(f"      🎯 Addresses request: {relevance_icon}")
        print(f"      🔧 Uses expected tools: {tools_icon}")
        print(f"      📊 {len(steps)} steps, complexity: {complexity}")
        
        # Show the steps
        if steps:
            print(f"      📝 Generated Steps:")
            for i, step in enumerate(steps, 1):
                step_title = step.get('title', 'Untitled')
                tool = step.get('tool', 'unknown')
                args = step.get('args', {})
                print(f"         {i}. {step_title}")
                print(f"            🔧 Tool: {tool}")
                if args:
                    args_str = ", ".join(f"{k}={v}" for k, v in args.items())
                    print(f"            ⚙️ Args: {args_str}")
        
        print()
        return result
        
    except Exception as e:
        print(f"      ❌ Failed: {str(e)[:100]}...")
        return {
            'provider': provider_config.display_name,
            'success': False,
            'error': str(e)[:200],
            'duration': 0,
            'addresses_request': False,
            'uses_expected_tools': False
        }


async def run_comprehensive_test():
    """Run comprehensive test across all providers and scenarios."""
    print("🎯 Comprehensive Provider Test - Fixed Version")
    print("=" * 60)
    
    all_results = []
    
    for scenario in TEST_SCENARIOS:
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"   Request: {scenario['prompt']}")
        print(f"   Expected keywords: {', '.join(scenario['expected_keywords'])}")
        print(f"   Expected tools: {', '.join(scenario['expected_tools'])}")
        
        scenario_results = []
        
        for provider_config in KEY_PROVIDERS:
            result = await test_single_provider(provider_config, scenario)
            scenario_results.append(result)
        
        all_results.append({
            'scenario': scenario['name'],
            'results': scenario_results
        })
    
    # Print summary
    print_final_summary(all_results)


def print_final_summary(all_results: List[Dict[str, Any]]):
    """Print a comprehensive summary."""
    print("\n📊 Final Test Results Summary")
    print("=" * 50)
    
    # Calculate provider statistics
    provider_stats = {}
    for provider_config in KEY_PROVIDERS:
        provider_stats[provider_config.display_name] = {
            'total_tests': 0,
            'successes': 0,
            'addresses_request': 0,
            'uses_expected_tools': 0,
            'total_duration': 0,
            'avg_duration': 0
        }
    
    # Collect stats
    for scenario_result in all_results:
        for result in scenario_result['results']:
            provider = result['provider']
            if provider in provider_stats:
                stats = provider_stats[provider]
                stats['total_tests'] += 1
                
                if result['success']:
                    stats['successes'] += 1
                    stats['total_duration'] += result['duration']
                    
                    if result.get('addresses_request', False):
                        stats['addresses_request'] += 1
                    
                    if result.get('uses_expected_tools', False):
                        stats['uses_expected_tools'] += 1
    
    # Calculate averages
    for stats in provider_stats.values():
        if stats['successes'] > 0:
            stats['avg_duration'] = stats['total_duration'] / stats['successes']
    
    # Print table
    print(f"\n{'Provider':<25} {'Success':<8} {'Relevance':<10} {'Tools':<8} {'Speed':<8}")
    print("-" * 65)
    
    for provider, stats in provider_stats.items():
        success_rate = (stats['successes'] / stats['total_tests'] * 100) if stats['total_tests'] > 0 else 0
        relevance_rate = (stats['addresses_request'] / stats['successes'] * 100) if stats['successes'] > 0 else 0
        tools_rate = (stats['uses_expected_tools'] / stats['successes'] * 100) if stats['successes'] > 0 else 0
        avg_speed = stats['avg_duration']
        
        print(f"{provider:<25} {success_rate:>5.0f}%    {relevance_rate:>7.0f}%    {tools_rate:>5.0f}%   {avg_speed:>5.1f}s")
    
    # Show best performers
    print(f"\n🏆 Best Performers by Scenario:")
    for scenario_result in all_results:
        successful_results = [r for r in scenario_result['results'] if r['success']]
        if successful_results:
            # Score based on relevance, tools, and speed
            def score_result(r):
                relevance_score = 100 if r.get('addresses_request', False) else 0
                tools_score = 50 if r.get('uses_expected_tools', False) else 0
                speed_score = max(0, 10 - r['duration'])  # Faster is better
                return relevance_score + tools_score + speed_score
            
            best = max(successful_results, key=score_result)
            relevance_icon = "✅" if best.get('addresses_request', False) else "❌"
            tools_icon = "✅" if best.get('uses_expected_tools', False) else "❌"
            
            print(f"   {scenario_result['scenario']:<25} → {best['provider']} {relevance_icon}{tools_icon}")
    
    # Final recommendations
    print(f"\n💡 Key Findings:")
    
    # Find most reliable
    most_reliable = max(provider_stats.items(), key=lambda x: x[1]['successes'] / x[1]['total_tests'] if x[1]['total_tests'] > 0 else 0)
    print(f"   🎯 Most Reliable: {most_reliable[0]}")
    
    # Find most relevant
    most_relevant = max(provider_stats.items(), key=lambda x: x[1]['addresses_request'] / x[1]['successes'] if x[1]['successes'] > 0 else 0)
    print(f"   ✅ Most Relevant: {most_relevant[0]}")
    
    # Find fastest
    fastest = min(provider_stats.items(), key=lambda x: x[1]['avg_duration'] if x[1]['successes'] > 0 else float('inf'))
    print(f"   ⚡ Fastest: {fastest[0]}")


async def test_specific_strengths():
    """Test each provider with a task designed for their strengths."""
    print(f"\n💪 Provider-Specific Strength Tests")
    print("=" * 50)
    
    strength_tests = [
        {
            "provider": "openai_gpt4o_mini",
            "task": "Calculate compound interest: $1000 principal, 5% annual rate, 3 years",
            "why": "Testing mathematical reliability"
        },
        {
            "provider": "claude4_sonnet",
            "task": "Create a detailed analysis plan for comparing three marketing strategies",
            "why": "Testing complex reasoning and planning"
        },
        {
            "provider": "mistral_medium",
            "task": "Set up automated workflow: get weather, search news, send daily briefing email",
            "why": "Testing tool integration capabilities"
        },
        {
            "provider": "watsonx_granite",
            "task": "Design audit checklist for regulatory compliance review process",
            "why": "Testing enterprise systematic approach"
        }
    ]
    
    for test in strength_tests:
        provider_config = next(p for p in KEY_PROVIDERS if p.name == test["provider"])
        
        print(f"\n🔹 {provider_config.display_name}")
        print(f"   Focus: {test['why']}")
        print(f"   Task: {test['task']}")
        
        scenario = {
            'name': 'Strength Test',
            'prompt': test['task'],
            'expected_keywords': test['task'].split()[:3],  # First 3 words as keywords
            'expected_tools': ['calculator', 'search', 'send_email']  # Accept any of these
        }
        
        await test_single_provider(provider_config, scenario)


async def main():
    """Run the complete fixed demo."""
    print("🎯 Final Plan Agent Demo - Completely Fixed")
    print("Addressing models falling back to coffee/weather examples")
    print()
    
    await run_comprehensive_test()
    await test_specific_strengths()
    
    print("\n🎉 Complete test finished!")
    print("\n🔧 Technical Insights:")
    print("   • Clean system prompts without examples reduce confusion")
    print("   • Flexible validators allow for more creative tool usage")
    print("   • Higher token limits enable more complex responses")
    print("   • Multiple retries help with JSON parsing issues")
    print("   • Provider selection should match task characteristics")


if __name__ == "__main__":
    asyncio.run(main())