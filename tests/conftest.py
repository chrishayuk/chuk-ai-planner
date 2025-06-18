"""
Root conftest.py to configure pytest for all tests.
"""
import sys
import os
from pathlib import Path
from enum import Enum

# Setup mocks FIRST before any imports
class EventType(Enum):
    MESSAGE = "message"
    TOOL_CALL = "tool_call" 
    SUMMARY = "summary"
    ERROR = "error"

class EventSource(str, Enum):
    """Source of the session event."""
    USER = "user"
    LLM = "llm"
    SYSTEM = "system"

class MockSessionEvent:
    def __init__(self, message=None, type=None, source=None, metadata=None, **kwargs):
        import uuid
        self.id = str(uuid.uuid4())
        self.message = message or {}
        self.type = type or EventType.MESSAGE
        self.source = source or EventSource.SYSTEM
        self.metadata = metadata or {}

class MockSessionRun:
    def __init__(self):
        import uuid
        self.id = str(uuid.uuid4())
        self.status = "pending"
    def mark_running(self): self.status = "running"
    def mark_completed(self): self.status = "completed"
    def mark_failed(self, error=None): self.status = "failed"

class MockSession:
    def __init__(self):
        import uuid
        self.id = str(uuid.uuid4())
        self.events = []
        self.runs = []

class MockSessionStore:
    def __init__(self):
        self.sessions = {}
    
    async def get(self, session_id): 
        return self.sessions.get(session_id)
    
    async def save(self, session): 
        """Properly async save method that is awaitable"""
        self.sessions[session.id] = session
        # Return None but make it awaitable

class MockSessionStoreProvider:
    _store = None
    @classmethod
    def get_store(cls):
        if cls._store is None:
            cls._store = MockSessionStore()
        return cls._store
    @classmethod 
    def set_store(cls, store): 
        cls._store = store

# Mock ToolResult class for chuk_tool_processor
class MockToolResult:
    def __init__(self, id="", tool="", args=None, result=None, error=None):
        self.id = id
        self.tool = tool
        self.args = args or {}
        self.arguments = args or {}  # Alias for compatibility
        self.result = result
        self.error = error

# Mock ToolCall class for chuk_tool_processor
class MockToolCall:
    def __init__(self, id=None, tool="", arguments=None):
        import uuid
        self.id = id or str(uuid.uuid4())
        self.tool = tool
        self.arguments = arguments or {}

# Mock classes for tool processor execution
class MockRegistry:
    def __init__(self):
        self.tools = {}

class MockInProcessStrategy:
    def __init__(self, registry=None):
        self.registry = registry

class MockToolExecutor:
    def __init__(self, registry=None, strategy=None):
        self.registry = registry
        self.strategy = strategy

# Mock function for getting default registry
async def mock_get_default_registry():
    return MockRegistry()

# Create simple module objects (not MagicMock to avoid recursion)
class SimpleModule:
    def __init__(self):
        pass

# Build the session manager module structure manually
session_mgr = SimpleModule()
session_mgr.models = SimpleModule()
session_mgr.models.event_type = SimpleModule()
session_mgr.models.event_source = SimpleModule()  
session_mgr.models.session_event = SimpleModule()
session_mgr.models.session_run = SimpleModule()
session_mgr.models.session = SimpleModule()
session_mgr.storage = SimpleModule()
session_mgr.storage.providers = SimpleModule()
session_mgr.storage.providers.memory = SimpleModule()

# Assign the session manager classes
session_mgr.models.event_type.EventType = EventType
session_mgr.models.event_source.EventSource = EventSource
session_mgr.models.session_event.SessionEvent = MockSessionEvent
session_mgr.models.session_run.SessionRun = MockSessionRun
session_mgr.models.session.Session = MockSession
session_mgr.models.session.SessionEvent = MockSessionEvent  # Make SessionEvent available in session module too
session_mgr.storage.SessionStoreProvider = MockSessionStoreProvider
session_mgr.storage.InMemorySessionStore = MockSessionStore  # Fix for import error
session_mgr.storage.providers.memory.InMemorySessionStore = MockSessionStore

# Build the tool processor module structure manually
tool_proc = SimpleModule()
tool_proc.models = SimpleModule()
tool_proc.models.tool_result = SimpleModule()
tool_proc.models.tool_call = SimpleModule()
tool_proc.registry = SimpleModule()
tool_proc.execution = SimpleModule()
tool_proc.execution.strategies = SimpleModule()
tool_proc.execution.strategies.inprocess_strategy = SimpleModule()
tool_proc.execution.tool_executor = SimpleModule()

# Assign the tool processor classes
tool_proc.models.tool_result.ToolResult = MockToolResult
tool_proc.models.tool_call.ToolCall = MockToolCall
tool_proc.registry.get_default_registry = mock_get_default_registry
tool_proc.execution.strategies.inprocess_strategy.InProcessStrategy = MockInProcessStrategy
tool_proc.execution.tool_executor.ToolExecutor = MockToolExecutor

# Add to sys.modules BEFORE any other imports
sys.modules['chuk_ai_session_manager'] = session_mgr
sys.modules['chuk_ai_session_manager.models'] = session_mgr.models
sys.modules['chuk_ai_session_manager.models.event_type'] = session_mgr.models.event_type
sys.modules['chuk_ai_session_manager.models.event_source'] = session_mgr.models.event_source
sys.modules['chuk_ai_session_manager.models.session_event'] = session_mgr.models.session_event
sys.modules['chuk_ai_session_manager.models.session_run'] = session_mgr.models.session_run
sys.modules['chuk_ai_session_manager.models.session'] = session_mgr.models.session
sys.modules['chuk_ai_session_manager.storage'] = session_mgr.storage
sys.modules['chuk_ai_session_manager.storage.providers'] = session_mgr.storage.providers
sys.modules['chuk_ai_session_manager.storage.providers.memory'] = session_mgr.storage.providers.memory

sys.modules['chuk_tool_processor'] = tool_proc
sys.modules['chuk_tool_processor.models'] = tool_proc.models
sys.modules['chuk_tool_processor.models.tool_result'] = tool_proc.models.tool_result
sys.modules['chuk_tool_processor.models.tool_call'] = tool_proc.models.tool_call
sys.modules['chuk_tool_processor.registry'] = tool_proc.registry
sys.modules['chuk_tool_processor.execution'] = tool_proc.execution
sys.modules['chuk_tool_processor.execution.strategies'] = tool_proc.execution.strategies
sys.modules['chuk_tool_processor.execution.strategies.inprocess_strategy'] = tool_proc.execution.strategies.inprocess_strategy
sys.modules['chuk_tool_processor.execution.tool_executor'] = tool_proc.execution.tool_executor

# NOW add the src directory to Python's module search path
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
    print(f"Added {src_dir} to Python path")

# Print debugging info
print(f"Project root: {project_root}")
print(f"Source directory: {src_dir}")

# Check for src/chuk_ai_planner directory
package_dir = src_dir / "chuk_ai_planner"
if not package_dir.exists():
    print(f"WARNING: Package directory not found: {package_dir}")
else:
    print(f"Package directory found: {package_dir}")
    
    # Check for key modules
    models_dir = package_dir / "models"
    storage_dir = package_dir / "storage"
    
    if models_dir.exists():
        print(f"Models directory found: {models_dir}")
        model_files = list(models_dir.glob("*.py"))
        print(f"Model files: {[f.name for f in model_files]}")
    
    if storage_dir.exists():
        print(f"Storage directory found: {storage_dir}")
        storage_files = list(storage_dir.glob("*.py"))
        print(f"Storage files: {[f.name for f in storage_files]}")

import pytest

@pytest.fixture
def sample_session():
    """Create a sample session for testing"""
    # Setup store
    store = MockSessionStore()
    MockSessionStoreProvider.set_store(store)
    
    # Create session
    session = MockSession()
    # Use asyncio.run to handle the async save
    import asyncio
    asyncio.run(store.save(session))
    
    return session

@pytest.fixture
def graph_store():
    """Create an in-memory graph store for testing"""
    from chuk_ai_planner.store.memory import InMemoryGraphStore
    return InMemoryGraphStore()

@pytest.fixture
def mock_tool_registry():
    """Create a mock tool registry with sample tools"""
    registry = {}
    
    # Add a simple echo tool
    def echo_tool(**kwargs):
        return {"echo": kwargs}
    
    # Add an async tool
    async def async_echo_tool(**kwargs):
        return {"async_echo": kwargs}
    
    registry["echo"] = echo_tool
    registry["async_echo"] = async_echo_tool
    
    return registry