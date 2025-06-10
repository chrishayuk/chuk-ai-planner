# tests/models/test_edges.py - Fixed for ReadOnlyList implementation
import pytest
import re
from datetime import datetime, timezone
from uuid import uuid4

from chuk_ai_planner.models.edges import (
    ParentChildEdge,
    NextEdge,
    PlanEdge,
    StepEdge,
    EdgeKind,
    GraphEdge,
)

UUID_V4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.I,
)


@pytest.mark.parametrize(
    "cls, expected_kind",
    [
        (ParentChildEdge, EdgeKind.PARENT_CHILD),
        (NextEdge, EdgeKind.NEXT),
        (PlanEdge, EdgeKind.PLAN_LINK),
        (StepEdge, EdgeKind.STEP_ORDER),
    ],
)
def test_edge_defaults(cls, expected_kind):
    edge: GraphEdge = cls(src="A", dst="B")  # type: ignore[call-arg]

    assert edge.kind == expected_kind
    assert UUID_V4_RE.match(edge.id)
    assert edge.src == "A" and edge.dst == "B"
    assert edge.data == {}


def test_edge_immutable():
    e = NextEdge(src="n1", dst="n2")
    with pytest.raises(TypeError):
        e.src = "XXX"  # frozen


def test_edge_repr():
    e = ParentChildEdge(src="abcdef00", dst="deadbeef")
    text = repr(e)
    # Shape: <parent_child:abcdef→deadbe>
    assert text.startswith(f"<{e.kind.value}:")
    assert "abcdef" in text and "deadbe" in text and text.endswith(">")


# ============================================================================
# IMMUTABILITY TESTS - UPDATED FOR READONLYLIST IMPLEMENTATION
# ============================================================================

class TestEdgeCreation:
    """Test edge creation with various parameters."""
    
    def test_custom_id(self):
        """Test creating edge with custom ID."""
        custom_id = str(uuid4())
        edge = GraphEdge(id=custom_id, kind=EdgeKind.CUSTOM, src="src1", dst="dst1")
        assert edge.id == custom_id
        assert edge.kind == EdgeKind.CUSTOM
        assert edge.src == "src1"
        assert edge.dst == "dst1"
    
    def test_with_data(self):
        """Test creating edge with custom data."""
        data = {"weight": 1.0, "label": "test", "metadata": {"created_by": "test"}}
        edge = ParentChildEdge(src="parent", dst="child", data=data)
        assert edge.data["weight"] == 1.0
        assert edge.data["metadata"]["created_by"] == "test"
    
    def test_nested_data_immutability(self):
        """Test that nested data structures are deeply frozen."""
        nested_data = {"config": {"timeout": 30, "retries": 3}, "items": [1, 2, 3]}
        edge = StepEdge(src="step1", dst="step2", data=nested_data)
        
        # Should be able to read nested values
        assert edge.data["config"]["timeout"] == 30
        assert edge.data["items"] == [1, 2, 3]  # ReadOnlyList compares equal to lists
        
        # Nested structures should be immutable
        from types import MappingProxyType
        from chuk_ai_planner.models.base import _ReadOnlyList
        
        assert isinstance(edge.data, MappingProxyType)
        assert isinstance(edge.data["config"], MappingProxyType)
        assert isinstance(edge.data["items"], _ReadOnlyList)  # Lists become _ReadOnlyList


class TestEdgeValidation:
    """Test edge validation and error cases."""
    
    def test_required_fields(self):
        """Test that required fields are enforced."""
        from pydantic_core import ValidationError
        
        with pytest.raises(ValidationError):
            GraphEdge()  # Missing required fields
        
        with pytest.raises(ValidationError):
            GraphEdge(kind=EdgeKind.CUSTOM)  # Missing src and dst
        
        with pytest.raises(ValidationError):
            GraphEdge(kind=EdgeKind.CUSTOM, src="test")  # Missing dst
    
    def test_invalid_node_ids(self):
        """Test validation of src/dst node IDs."""
        # Current implementation raises ValidationError, not TypeError
        from pydantic_core import ValidationError
        
        with pytest.raises(ValidationError):
            GraphEdge(kind=EdgeKind.CUSTOM, src=123, dst="dst")  # Non-string src
        
        with pytest.raises(ValidationError):
            GraphEdge(kind=EdgeKind.CUSTOM, src="src", dst=None)  # None dst
        
        # Empty strings are allowed
        edge = GraphEdge(kind=EdgeKind.CUSTOM, src="", dst="dst")
        assert edge.src == ""
        
        edge = GraphEdge(kind=EdgeKind.CUSTOM, src="src", dst="")
        assert edge.dst == ""
    
    def test_invalid_data_type(self):
        """Test validation of data field type."""
        # Current implementation raises ValidationError, not TypeError
        from pydantic_core import ValidationError
        
        with pytest.raises(ValidationError):
            GraphEdge(kind=EdgeKind.CUSTOM, src="a", dst="b", data="not a dict")
        
        with pytest.raises(ValidationError):
            GraphEdge(kind=EdgeKind.CUSTOM, src="a", dst="b", data=[1, 2, 3])


class TestEdgeImmutability:
    """Test comprehensive immutability enforcement."""
    
    def test_all_fields_immutable(self):
        """Test that all edge fields are immutable."""
        edge = PlanEdge(src="plan", dst="task", data={"priority": "high"})
        
        # Test each field
        with pytest.raises(TypeError):
            edge.id = "new_id"
        
        with pytest.raises(TypeError):
            edge.kind = EdgeKind.NEXT
        
        with pytest.raises(TypeError):
            edge.src = "new_src"
        
        with pytest.raises(TypeError):
            edge.dst = "new_dst"
        
        with pytest.raises(TypeError):
            edge.data = {"new": "data"}
    
    def test_data_dict_immutable(self):
        """Test that the data dictionary is immutable."""
        edge = StepEdge(src="step1", dst="step2", data={"order": 1, "config": {"timeout": 30}})
        
        # Direct modification should fail
        with pytest.raises(TypeError):
            edge.data["new_key"] = "new_value"
        
        # Deletion should fail
        with pytest.raises(TypeError):
            del edge.data["order"]
    
    def test_nested_data_immutable(self):
        """Test that nested data structures are also immutable."""
        nested_data = {"config": {"timeout": 30, "retries": 3}, "items": [1, 2, 3]}
        edge = ParentChildEdge(src="parent", dst="child", data=nested_data)
        
        # Should be able to read nested values
        assert edge.data["config"]["timeout"] == 30
        assert edge.data["items"] == [1, 2, 3]  # ReadOnlyList compares equal to lists
        
        # But not modify them
        with pytest.raises(TypeError):
            edge.data["config"]["timeout"] = 60
        
        # Lists are converted to ReadOnlyList (immutable)
        from chuk_ai_planner.models.base import _ReadOnlyList
        assert isinstance(edge.data["items"], _ReadOnlyList)
        
        # Try to modify the ReadOnlyList - should fail
        with pytest.raises(TypeError):
            edge.data["items"].append(4)
    
    def test_data_isolation(self):
        """Test that original data cannot affect edge after creation."""
        mutable_data = {"items": [1, 2, 3], "config": {"debug": True}}
        edge = StepEdge(src="step1", dst="step2", data=mutable_data)
        
        # Modify original data
        mutable_data["items"].append(4)
        mutable_data["new_key"] = "new_value"
        mutable_data["config"]["debug"] = False
        
        # Edge data should be unchanged (deep copy + freeze)
        assert edge.data["items"] == [1, 2, 3]  # ReadOnlyList still compares equal
        assert "new_key" not in edge.data
        assert edge.data["config"]["debug"] is True


class TestEdgeEquality:
    """Test edge equality and hashing."""
    
    def test_edge_equality_by_id(self):
        """Test that edges are equal if they have the same ID."""
        edge_id = str(uuid4())
        edge1 = GraphEdge(id=edge_id, kind=EdgeKind.CUSTOM, src="a", dst="b")
        edge2 = GraphEdge(id=edge_id, kind=EdgeKind.NEXT, src="c", dst="d")  # Different fields
        
        # Should be equal based on ID
        assert edge1 == edge2
        assert hash(edge1) == hash(edge2)
    
    def test_edge_hashing(self):
        """Test that edges can be used as dictionary keys and in sets."""
        edge1 = NextEdge(src="a", dst="b")
        edge2 = StepEdge(src="c", dst="d")
        
        # Should be hashable
        edge_set = {edge1, edge2}
        assert len(edge_set) == 2
        
        edge_dict = {edge1: "value1", edge2: "value2"}
        assert edge_dict[edge1] == "value1"
        assert edge_dict[edge2] == "value2"
    
    def test_edge_uniqueness(self):
        """Test that different edge instances have different IDs."""
        edge1 = ParentChildEdge(src="a", dst="b")
        edge2 = ParentChildEdge(src="a", dst="b")  # Same src/dst
        
        assert edge1.id != edge2.id
        assert edge1 != edge2


class TestEdgeDataTypes:
    """Test various data types in edge data."""
    
    def test_complex_data_types(self):
        """Test that complex data types are properly frozen."""
        complex_data = {
            "string": "test",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none": None,
            "list": [1, 2, {"nested": "value"}],
            "dict": {"nested": "value"},
            "set": {1, 2, 3},
            "datetime": datetime.now(timezone.utc).isoformat()
        }
        
        edge = ParentChildEdge(src="a", dst="b", data=complex_data)
        
        # All data should be preserved but frozen - check for current behavior
        assert edge.data["string"] == "test"
        assert edge.data["integer"] == 42
        assert edge.data["float"] == 3.14
        assert edge.data["boolean"] is True
        assert edge.data["none"] is None
        assert edge.data["list"] == [1, 2, {"nested": "value"}]  # ReadOnlyList compares equal to lists
        assert edge.data["dict"]["nested"] == "value"
        assert edge.data["set"] == frozenset({1, 2, 3})  # Set → frozenset
        
        # Verify immutability
        from types import MappingProxyType
        from chuk_ai_planner.models.base import _ReadOnlyList
        
        assert isinstance(edge.data, MappingProxyType)
        assert isinstance(edge.data["dict"], MappingProxyType)
        assert isinstance(edge.data["list"], _ReadOnlyList)
        assert isinstance(edge.data["set"], frozenset)


class TestSpecificEdgeTypes:
    """Test specific edge type behaviors."""
    
    def test_parent_child_edge(self):
        """Test ParentChildEdge specific functionality."""
        edge = ParentChildEdge(
            src="session_123", 
            dst="message_456",
            data={"relationship": "contains", "created_at": "2023-01-01"}
        )
        assert edge.kind == EdgeKind.PARENT_CHILD
        assert edge.data["relationship"] == "contains"
    
    def test_next_edge(self):
        """Test NextEdge specific functionality."""
        edge = NextEdge(
            src="msg1", 
            dst="msg2",
            data={"sequence": 1, "time_gap": "5s"}
        )
        assert edge.kind == EdgeKind.NEXT
        assert edge.data["sequence"] == 1
    
    def test_plan_edge(self):
        """Test PlanEdge specific functionality."""
        edge = PlanEdge(
            src="plan_step", 
            dst="tool_call",
            data={"execution_order": 1, "timeout": 300}
        )
        assert edge.kind == EdgeKind.PLAN_LINK
        assert edge.data["execution_order"] == 1
    
    def test_step_edge(self):
        """Test StepEdge specific functionality."""
        edge = StepEdge(
            src="step1", 
            dst="step2",
            data={"dependency_type": "sequential", "blocking": True}
        )
        assert edge.kind == EdgeKind.STEP_ORDER
        assert edge.data["dependency_type"] == "sequential"
        assert edge.data["blocking"] is True


class TestErrorHandling:
    """Test error handling and comprehensive validation."""
    
    def test_invalid_data_modification_attempts(self):
        """Test that all data modification attempts are properly blocked."""
        edge = PlanEdge(src="plan", dst="tool", data={"config": {"timeout": 30}, "items": [1, 2, 3]})
        
        # Direct assignment should fail
        with pytest.raises(TypeError):
            edge.data["new"] = "value"
        
        # Deletion should fail  
        with pytest.raises(TypeError):
            del edge.data["config"]
        
        # Nested modification should fail
        with pytest.raises(TypeError):
            edge.data["config"]["timeout"] = 60
        
        # List operations should fail (lists are converted to ReadOnlyList)
        from chuk_ai_planner.models.base import _ReadOnlyList
        assert isinstance(edge.data["items"], _ReadOnlyList)
        
        # ReadOnlyList should prevent append
        with pytest.raises(TypeError):
            edge.data["items"].append(4)
    
    def test_comprehensive_immutability(self):
        """Test comprehensive immutability across all data types."""
        data = {
            "simple": "value",
            "nested_dict": {"inner": {"deep": "value"}},
            "nested_list": [1, [2, 3], {"key": "value"}],
            "mixed_set": {1, "two", 3.0},
            "complex": {
                "lists": [[1, 2], [3, 4]],
                "dicts": {"a": {"b": {"c": "deep"}}},
                "sets": {frozenset({1, 2}), frozenset({3, 4})}
            }
        }
        
        edge = NextEdge(src="a", dst="b", data=data)
        
        # All structures should be deeply frozen
        from types import MappingProxyType
        from chuk_ai_planner.models.base import _ReadOnlyList
        
        # Top level
        assert isinstance(edge.data, MappingProxyType)
        
        # Nested dicts
        assert isinstance(edge.data["nested_dict"], MappingProxyType)
        assert isinstance(edge.data["nested_dict"]["inner"], MappingProxyType)
        
        # Lists converted to ReadOnlyList in current implementation
        assert isinstance(edge.data["nested_list"], _ReadOnlyList)
        assert isinstance(edge.data["nested_list"][1], _ReadOnlyList)  # Nested list → ReadOnlyList
        assert isinstance(edge.data["nested_list"][2], MappingProxyType)  # Dict in list
        
        # Sets converted to frozensets
        assert isinstance(edge.data["mixed_set"], frozenset)
        
        # Complex nested structures
        assert isinstance(edge.data["complex"]["lists"], _ReadOnlyList)
        assert isinstance(edge.data["complex"]["lists"][0], _ReadOnlyList)
        assert isinstance(edge.data["complex"]["dicts"], MappingProxyType)
        assert isinstance(edge.data["complex"]["dicts"]["a"]["b"], MappingProxyType)
        assert isinstance(edge.data["complex"]["sets"], frozenset)


class TestJSONSerialization:
    """Test JSON serialization capabilities."""
    
    def test_to_dict_method(self):
        """Test that edges can be converted to JSON-serializable dictionaries."""
        data = {
            "weight": 1.5,
            "metadata": {"created_by": "system"},
            "tags": ["important", "urgent"],
            "config": {"timeout": 30}
        }
        edge = PlanEdge(src="step1", dst="tool1", data=data)
        
        # to_dict should produce JSON-serializable output
        edge_dict = edge.to_dict()
        
        assert edge_dict["id"] == edge.id
        assert edge_dict["kind"] == "plan_link"
        assert edge_dict["src"] == "step1"
        assert edge_dict["dst"] == "tool1"
        
        # Data should be unfrozen for JSON compatibility
        data_dict = edge_dict["data"]
        assert data_dict["weight"] == 1.5
        assert data_dict["metadata"]["created_by"] == "system"
        assert data_dict["tags"] == ["important", "urgent"]  # Should be regular list
        assert data_dict["config"]["timeout"] == 30
        
        # Should be JSON serializable
        import json
        json_str = json.dumps(edge_dict)
        assert isinstance(json_str, str)
        assert len(json_str) > 0
    
    def test_get_json_serializable_data_method(self):
        """Test the get_json_serializable_data method specifically."""
        data = {
            "lists": [[1, 2], [3, 4]],
            "dicts": {"nested": {"deep": "value"}},
            "mixed": [{"key": "value"}, [5, 6]]
        }
        edge = NextEdge(src="a", dst="b", data=data)
        
        # Get JSON-serializable version of just the data
        json_data = edge.get_json_serializable_data()
        
        # Should be regular Python types
        assert isinstance(json_data["lists"], list)
        assert isinstance(json_data["lists"][0], list)
        assert isinstance(json_data["dicts"], dict)
        assert isinstance(json_data["dicts"]["nested"], dict)
        assert isinstance(json_data["mixed"][0], dict)
        assert isinstance(json_data["mixed"][1], list)
        
        # Should be JSON serializable
        import json
        json_str = json.dumps(json_data)
        assert isinstance(json_str, str)
        assert len(json_str) > 0