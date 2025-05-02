# chuk_ai_planner/demo/llm_simulator.py
"""
LLM simulation for testing and demonstration purposes.

This module provides simulated LLM responses that can be used in place
of actual LLM calls during testing and demonstrations.
"""

import asyncio
import json
from typing import Dict, Any

async def simulate_llm_call(prompt: str) -> Dict[str, Any]:
    """
    Simulate an LLM call with responses based on prompt content.
    
    Parameters
    ----------
    prompt : str
        The prompt text to analyze
        
    Returns
    -------
    Dict[str, Any]
        Simulated LLM response with content and tool_calls
    """
    print(f"🤖 LLM prompted with: {prompt[:50]}...")
    
    # Simulate LLM thinking
    await asyncio.sleep(0.5)
    
    # For demo purposes, return a response based on keywords in the prompt
    if "weather" in prompt.lower():
        return {
            "content": "I'll check the weather for you.",
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": json.dumps({"location": "New York"})
                    }
                }
            ]
        }
    elif any(x in prompt.lower() for x in ["calculate", "math", "multiply"]):
        return {
            "content": "I'll calculate that for you.",
            "tool_calls": [
                {
                    "id": "call_calc",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": json.dumps({
                            "operation": "multiply", 
                            "a": 235.5, 
                            "b": 18.75
                        })
                    }
                }
            ]
        }
    elif "search" in prompt.lower():
        return {
            "content": "Let me search for that information.",
            "tool_calls": [
                {
                    "id": "call_search",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": json.dumps({
                            "query": "climate change adaptation"
                        })
                    }
                }
            ]
        }
    else:
        # Default response with multiple tool calls
        return {
            "content": "I'll help you with that comprehensive request.",
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": json.dumps({"location": "New York"})
                    }
                },
                {
                    "id": "call_calc",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": json.dumps({
                            "operation": "multiply", 
                            "a": 235.5, 
                            "b": 18.75
                        })
                    }
                }
            ]
        }