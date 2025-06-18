# chuk_ai_planner/agents/plan_agent.py
"""
chuk_ai_planner.agents.plan_agent
==========================================

Flexible plan agent that uses chuk_llm's get_client approach for maximum
configurability and supports environment variable configuration.

Environment Variables:
- PLANNER_LLM_PROVIDER: Provider to use (default: openai)
- PLANNER_LLM_MODEL: Model to use (default: gpt-4o-mini)
- PLANNER_LLM_TEMPERATURE: Temperature (default: 0.3)
- PLANNER_LLM_MAX_TOKENS: Max tokens (default: 1000)
- PLANNER_LLM_MAX_RETRIES: Max retries (default: 3)

Typical usage
-------------
    from chuk_ai_planner.agents.plan_agent import FlexiblePlanAgent

    async def my_validator(step: dict[str, any]) -> tuple[bool, str]:
        ...

    # Uses environment variables or defaults
    agent = FlexiblePlanAgent(
        system_prompt=SYS_MSG,
        validate_step=my_validator,
    )
    
    # Or override specific settings
    agent = FlexiblePlanAgent(
        system_prompt=SYS_MSG,
        validate_step=my_validator,
        provider="anthropic",
        model="claude-3-sonnet-20240229"
    )
    
    plan_dict = await agent.plan("user request")
"""

from __future__ import annotations
import json
import os
import textwrap
from typing import Any, Callable, Dict, List, Tuple, Optional

from dotenv import load_dotenv

load_dotenv()  # Load environment variables

# chuk llm
from chuk_llm.llm.client import get_client
from chuk_llm.llm.system_prompt_generator import SystemPromptGenerator

_Validate = Callable[[Dict[str, Any]], Tuple[bool, str]]


class PlanAgent:
    """
    Flexible plan generator using chuk_llm's get_client approach.
    
    This agent is highly configurable through environment variables
    and supports all providers/models that chuk_llm supports.
    """

    def __init__(
        self,
        *,
        system_prompt: str,
        validate_step: _Validate,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        use_system_prompt_generator: bool = False,
    ):
        """
        Initialize the flexible plan agent.
        
        Parameters
        ----------
        system_prompt : str
            The system prompt for plan generation
        validate_step : Callable
            Function to validate individual plan steps
        provider : Optional[str]
            LLM provider (overrides PLANNER_LLM_PROVIDER env var)
        model : Optional[str]
            Model name (overrides PLANNER_LLM_MODEL env var)
        temperature : Optional[float]
            Sampling temperature (overrides PLANNER_LLM_TEMPERATURE env var)
        max_tokens : Optional[int]
            Maximum tokens (overrides PLANNER_LLM_MAX_TOKENS env var)
        max_retries : Optional[int]
            Maximum retry attempts (overrides PLANNER_LLM_MAX_RETRIES env var)
        use_system_prompt_generator : bool
            Whether to use chuk_llm's SystemPromptGenerator
        """
        # Configuration with environment variable fallbacks
        self.provider = provider or os.getenv("PLANNER_LLM_PROVIDER", "openai")
        self.model = model or os.getenv("PLANNER_LLM_MODEL", "gpt-4o-mini")
        self.temperature = temperature or float(os.getenv("PLANNER_LLM_TEMPERATURE", "0.3"))
        self.max_tokens = max_tokens or (int(os.getenv("PLANNER_LLM_MAX_TOKENS")) if os.getenv("PLANNER_LLM_MAX_TOKENS") else 1000)
        self.max_retries = max_retries or int(os.getenv("PLANNER_LLM_MAX_RETRIES", "3"))
        
        # System prompt handling
        if use_system_prompt_generator:
            self.system_prompt = SystemPromptGenerator().generate_prompt({})
        else:
            self.system_prompt = textwrap.dedent(system_prompt).strip()
        
        self.validate_step = validate_step
        self.history: List[Dict[str, Any]] = []
        
        # Initialize client
        self.client = None
        
        # Display configuration
        print(f"🔧 Plan Agent Configuration:")
        print(f"   Provider: {self.provider}")
        print(f"   Model: {self.model}")
        print(f"   Temperature: {self.temperature}")
        print(f"   Max Tokens: {self.max_tokens}")
        print(f"   Max Retries: {self.max_retries}")

    def _get_client(self):
        """Get or create the LLM client."""
        if self.client is None:
            self.client = get_client(provider=self.provider, model=self.model)
        return self.client

    async def _chat(self, prompt: str) -> str:
        """Make a chat request using chuk_llm's client approach."""
        client = self._get_client()
        
        # Prepare messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Use the client's create_completion method
            completion = await client.create_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract response content
            if isinstance(completion, dict):
                return completion.get("response", str(completion))
            else:
                return str(completion)
                
        except Exception as e:
            raise RuntimeError(f"LLM call failed for {self.provider}/{self.model}: {str(e)}")

    async def plan(self, user_prompt: str) -> Dict[str, Any]:
        """Return a syntactically and semantically valid JSON plan."""
        prompt = user_prompt

        for attempt in range(1, self.max_retries + 1):
            try:
                raw = await self._chat(prompt)
                
                record = {
                    "attempt": attempt, 
                    "raw": raw, 
                    "provider": self.provider, 
                    "model": self.model,
                    "prompt_length": len(prompt)
                }
                
                # Try to extract and parse JSON
                plan = self._extract_and_parse_json(raw)
                
                # Validate the plan
                errors = self._validate_plan(plan)
                
                record["errors"] = errors
                record["plan"] = plan if not errors else None
                self.history.append(record)
                
                if not errors:
                    print(f"✅ Plan generated successfully on attempt {attempt}")
                    return plan
                
                # Prepare corrective prompt for next attempt
                error_summary = "\n".join(f"- {e}" for e in errors)
                prompt = (
                    f"Your previous response had these issues:\n{error_summary}\n\n"
                    f"Please return a *complete* corrected JSON plan that addresses all these issues. "
                    f"Make sure to return ONLY valid JSON with no additional text or formatting."
                )
                
                print(f"⚠️ Attempt {attempt} failed: {len(errors)} errors. Retrying...")
                
            except Exception as e:
                record = {
                    "attempt": attempt,
                    "error": str(e),
                    "provider": self.provider,
                    "model": self.model
                }
                self.history.append(record)
                
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"Plan generation failed after {self.max_retries} attempts. "
                        f"Last error: {str(e)}"
                    )

        # If we get here, all attempts failed
        raise RuntimeError(
            f"Plan generation failed after {self.max_retries} attempts. "
            f"Debug trace:\n{json.dumps(self.history, indent=2)[:1500]}"
        )

    def _extract_and_parse_json(self, raw_response: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response."""
        # First try to parse the raw response
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from code blocks
        import re
        
        # Look for JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # Look for JSON-like structures
        json_match = re.search(r'(\{.*\})', raw_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        raise json.JSONDecodeError("No valid JSON found in response", raw_response, 0)

    def _validate_plan(self, plan: Dict[str, Any]) -> List[str]:
        """Validate the plan structure and steps."""
        errors = []
        
        # Check basic structure
        if not isinstance(plan, dict):
            errors.append("Plan must be a JSON object")
            return errors
        
        if "steps" not in plan:
            errors.append("Plan must have a 'steps' field")
            return errors
        
        if not isinstance(plan["steps"], list):
            errors.append("'steps' must be an array")
            return errors
        
        # Validate each step
        for i, step in enumerate(plan["steps"], 1):
            if not isinstance(step, dict):
                errors.append(f"Step {i} must be an object")
                continue
            
            # Use the custom validator
            try:
                is_valid, error_msg = self.validate_step(step)
                if not is_valid:
                    errors.append(f"Step {i}: {error_msg}")
            except Exception as e:
                errors.append(f"Step {i}: Validation error - {str(e)}")
        
        return errors

    def get_configuration(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
            "attempts_made": len(self.history)
        }

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the generation history."""
        return self.history.copy()


# ---------------------------------------------------------------- Validator functions

def create_basic_step_validator() -> _Validate:
    """Create a basic step validator for common tool-based plans."""
    
    def validate_step(step: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate a single step in the plan."""
        
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
        
        return True, ""
    
    return validate_step


def create_tool_specific_validator(valid_tools: Dict[str, Dict[str, Any]]) -> _Validate:
    """
    Create a validator for specific tools with their requirements.
    
    Parameters
    ----------
    valid_tools : Dict[str, Dict[str, Any]]
        Tool definitions with requirements. Example:
        {
            "weather": {"required_args": ["location"], "arg_types": {"location": str}},
            "calculator": {"required_args": ["operation", "a", "b"], "arg_types": {"operation": str, "a": (int, float), "b": (int, float)}}
        }
    """
    
    def validate_step(step: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate a step with tool-specific rules."""
        
        # Basic validation first
        basic_validator = create_basic_step_validator()
        is_valid, error = basic_validator(step)
        if not is_valid:
            return is_valid, error
        
        # Tool-specific validation
        tool = step["tool"]
        args = step.get("args", {})
        
        if tool not in valid_tools:
            available_tools = ", ".join(sorted(valid_tools.keys()))
            return False, f"Unknown tool '{tool}'. Valid tools: {available_tools}"
        
        tool_spec = valid_tools[tool]
        
        # Check required arguments
        required_args = tool_spec.get("required_args", [])
        for arg in required_args:
            if arg not in args:
                return False, f"Tool '{tool}' requires argument '{arg}'"
        
        # Check argument types
        arg_types = tool_spec.get("arg_types", {})
        for arg, expected_type in arg_types.items():
            if arg in args:
                if isinstance(expected_type, tuple):
                    # Multiple allowed types
                    if not isinstance(args[arg], expected_type):
                        type_names = " or ".join(t.__name__ for t in expected_type)
                        return False, f"Argument '{arg}' must be {type_names}"
                else:
                    # Single type
                    if not isinstance(args[arg], expected_type):
                        return False, f"Argument '{arg}' must be {expected_type.__name__}"
        
        # Check specific validations
        validations = tool_spec.get("validations", {})
        for arg, validation_func in validations.items():
            if arg in args:
                try:
                    is_valid, error_msg = validation_func(args[arg])
                    if not is_valid:
                        return False, f"Argument '{arg}': {error_msg}"
                except Exception as e:
                    return False, f"Validation error for '{arg}': {str(e)}"
        
        return True, ""
    
    return validate_step


def create_weather_calculator_validator() -> _Validate:
    """Create a validator specifically for weather and calculator tools."""
    
    def validate_operation(operation: str) -> Tuple[bool, str]:
        """Validate calculator operation."""
        valid_ops = ["add", "subtract", "multiply", "divide"]
        if operation not in valid_ops:
            return False, f"must be one of: {', '.join(valid_ops)}"
        return True, ""
    
    valid_tools = {
        "weather": {
            "required_args": ["location"],
            "arg_types": {"location": str}
        },
        "calculator": {
            "required_args": ["operation", "a", "b"],
            "arg_types": {"operation": str, "a": (int, float), "b": (int, float)},
            "validations": {"operation": validate_operation}
        },
        "search": {
            "required_args": ["query"],
            "arg_types": {"query": str}
        },
        "grind_beans": {},
        "boil_water": {},
        "brew_coffee": {},
        "clean_station": {}
    }
    
    return create_tool_specific_validator(valid_tools)


# ---------------------------------------------------------------- Configuration helpers

def get_plan_agent_config_from_env() -> Dict[str, Any]:
    """Get plan agent configuration from environment variables."""
    return {
        "provider": os.getenv("PLANNER_LLM_PROVIDER", "openai"),
        "model": os.getenv("PLANNER_LLM_MODEL", "gpt-4o-mini"),
        "temperature": float(os.getenv("PLANNER_LLM_TEMPERATURE", "0.3")),
        "max_tokens": int(os.getenv("PLANNER_LLM_MAX_TOKENS", "1000")) if os.getenv("PLANNER_LLM_MAX_TOKENS") else 1000,
        "max_retries": int(os.getenv("PLANNER_LLM_MAX_RETRIES", "3"))
    }


def create_configured_plan_agent(system_prompt: str, validate_step: _Validate, **overrides) -> FlexiblePlanAgent:
    """Create a plan agent with environment configuration and optional overrides."""
    config = get_plan_agent_config_from_env()
    config.update(overrides)
    
    return FlexiblePlanAgent(
        system_prompt=system_prompt,
        validate_step=validate_step,
        **config
    )


# ---------------------------------------------------------------- Demo/Example

async def demo_flexible_plan_agent():
    """Demonstrate the FlexiblePlanAgent."""
    
    system_prompt = """
    You are an assistant that converts a natural-language task into a JSON plan.
    Return ONLY valid JSON!
    
    Schema:
    {
      "title": str,
      "steps": [
        {"title": str, "tool": str, "args": {}, "depends_on": [indices]},
        ...
      ]
    }
    
    Indices start at 1 in the final output.
    Available tools: weather, calculator, search, grind_beans, boil_water, brew_coffee, clean_station
    
    Rules:
    - Each step must have "title" and "tool" fields
    - "args" is optional but must be an object if present
    - "depends_on" is optional but must be an array of step indices if present
    - Weather tool needs {"location": "city name"}
    - Calculator tool needs {"operation": "add|subtract|multiply|divide", "a": number, "b": number}
    - Search tool needs {"query": "search terms"}
    """
    
    print("🚀 Flexible Plan Agent Demo")
    print("=" * 50)
    
    # Test 1: Using environment variables (defaults)
    print("\n1️⃣ Using environment configuration:")
    agent1 = FlexiblePlanAgent(
        system_prompt=system_prompt,
        validate_step=create_weather_calculator_validator(),
    )
    
    try:
        plan1 = await agent1.plan(
            "Check weather in Tokyo, then calculate 15 * 23, "
            "and search for machine learning tutorials."
        )
        print("✅ Plan generated successfully!")
        print(json.dumps(plan1, indent=2))
        config1 = agent1.get_configuration()
        print(f"📊 Configuration: {config1}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Test 2: Override with Anthropic
    print("2️⃣ Override with Anthropic:")
    agent2 = FlexiblePlanAgent(
        system_prompt=system_prompt,
        validate_step=create_weather_calculator_validator(),
        provider="anthropic",
        model="claude-3-sonnet-20240229",
        temperature=0.2
    )
    
    try:
        plan2 = await agent2.plan(
            "Get weather for London, multiply 42 by 7, "
            "and search for 'async programming Python'."
        )
        print("✅ Anthropic plan generated successfully!")
        print(json.dumps(plan2, indent=2))
        config2 = agent2.get_configuration()
        print(f"📊 Configuration: {config2}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Test 3: Using helper function
    print("3️⃣ Using configuration helper:")
    agent3 = create_configured_plan_agent(
        system_prompt=system_prompt,
        validate_step=create_weather_calculator_validator(),
        provider="openai",  # Override just the provider
        max_retries=2
    )
    
    try:
        plan3 = await agent3.plan(
            "Plan a morning routine: check weather, calculate my budget (income 5000 - expenses 3500), "
            "and search for productivity tips."
        )
        print("✅ Helper-configured plan generated successfully!")
        print(json.dumps(plan3, indent=2))
        config3 = agent3.get_configuration()
        print(f"📊 Configuration: {config3}")
    except Exception as e:
        print(f"❌ Failed: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_flexible_plan_agent())