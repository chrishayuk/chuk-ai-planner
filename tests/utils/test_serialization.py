"""Tests for utils/serialization.py"""

import json
from types import MappingProxyType

from chuk_ai_planner.utils.serialization import (
    unfreeze_data,
    serialize_node_data,
    serialize_tool_args,
)


class TestUnfreezeData:
    """Test the unfreeze_data function."""

    def test_unfreeze_mapping_proxy(self):
        """Test unfreezing a MappingProxyType."""
        frozen_dict = MappingProxyType({"key": "value", "number": 42})
        result = unfreeze_data(frozen_dict)

        assert isinstance(result, dict)
        assert result == {"key": "value", "number": 42}

    def test_unfreeze_nested_mapping_proxy(self):
        """Test unfreezing nested MappingProxyType."""
        inner = MappingProxyType({"inner_key": "inner_value"})
        outer = MappingProxyType({"outer_key": inner})
        result = unfreeze_data(outer)

        assert isinstance(result, dict)
        assert isinstance(result["outer_key"], dict)
        assert result == {"outer_key": {"inner_key": "inner_value"}}

    def test_unfreeze_tuple(self):
        """Test unfreezing a tuple to list."""
        frozen_tuple = (1, 2, 3, 4)
        result = unfreeze_data(frozen_tuple)

        assert isinstance(result, list)
        assert result == [1, 2, 3, 4]

    def test_unfreeze_nested_tuple(self):
        """Test unfreezing nested tuples."""
        nested = ((1, 2), (3, 4))
        result = unfreeze_data(nested)

        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert result == [[1, 2], [3, 4]]

    def test_unfreeze_frozenset(self):
        """Test unfreezing a frozenset to list."""
        frozen = frozenset([1, 2, 3, 4])
        result = unfreeze_data(frozen)

        assert isinstance(result, list)
        assert set(result) == {1, 2, 3, 4}  # Order may vary

    def test_unfreeze_nested_frozenset(self):
        """Test unfreezing nested frozensets."""
        inner = frozenset([1, 2])
        outer = frozenset([inner, 3])
        result = unfreeze_data(outer)

        assert isinstance(result, list)
        # The inner frozenset should also be converted to a list
        assert 3 in result
        # One of the items should be a list containing 1 and 2
        list_items = [item for item in result if isinstance(item, list)]
        assert len(list_items) == 1
        assert set(list_items[0]) == {1, 2}

    def test_unfreeze_regular_dict(self):
        """Test that regular dicts are processed (recursion for nested structures)."""
        regular_dict = {"key": "value"}
        result = unfreeze_data(regular_dict)

        # Now we recurse into dicts to handle nested frozen structures
        assert result == regular_dict
        assert isinstance(result, dict)

    def test_unfreeze_regular_list(self):
        """Test that regular lists are processed (recursion for nested structures)."""
        regular_list = [1, 2, 3]
        result = unfreeze_data(regular_list)

        # Now we recurse into lists to handle nested frozen structures
        assert result == regular_list
        assert isinstance(result, list)

    def test_unfreeze_primitive_types(self):
        """Test that primitive types pass through unchanged."""
        assert unfreeze_data(42) == 42
        assert unfreeze_data("string") == "string"
        assert unfreeze_data(3.14) == 3.14
        assert unfreeze_data(True) is True
        assert unfreeze_data(None) is None

    def test_unfreeze_complex_nested_structure(self):
        """Test unfreezing a complex nested structure."""
        complex_structure = MappingProxyType(
            {
                "frozen_dict": MappingProxyType({"inner": "value"}),
                "tuple": (1, 2, 3),
                "frozenset": frozenset([4, 5, 6]),
                "regular": {"normal": "dict"},
                "nested_tuple": ((1, 2), (3, 4)),
            }
        )
        result = unfreeze_data(complex_structure)

        assert isinstance(result, dict)
        assert isinstance(result["frozen_dict"], dict)
        assert isinstance(result["tuple"], list)
        assert isinstance(result["frozenset"], list)
        assert isinstance(result["regular"], dict)
        assert isinstance(result["nested_tuple"], list)
        assert isinstance(result["nested_tuple"][0], list)


class TestSerializeNodeData:
    """Test the serialize_node_data function."""

    def test_serialize_simple_dict(self):
        """Test serializing a simple dictionary."""
        data = {"key": "value", "number": 42}
        result = serialize_node_data(data)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == data

    def test_serialize_mapping_proxy(self):
        """Test serializing a MappingProxyType."""
        data = MappingProxyType({"frozen": "data", "count": 10})
        result = serialize_node_data(data)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == {"frozen": "data", "count": 10}

    def test_serialize_with_tuple(self):
        """Test serializing data with tuples."""
        data = {"tuple": (1, 2, 3)}
        result = serialize_node_data(data)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == {"tuple": [1, 2, 3]}

    def test_serialize_with_frozenset(self):
        """Test serializing data with frozensets."""
        data = {"frozenset": frozenset([1, 2, 3])}
        result = serialize_node_data(data)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert set(parsed["frozenset"]) == {1, 2, 3}

    def test_serialize_complex_node_data(self):
        """Test serializing complex node data."""
        data = MappingProxyType(
            {
                "id": "node_1",
                "type": "plan",
                "metadata": MappingProxyType({"tags": ("tag1", "tag2")}),
                "frozen_set": frozenset(["a", "b"]),
            }
        )
        result = serialize_node_data(data)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["id"] == "node_1"
        assert parsed["type"] == "plan"
        assert parsed["metadata"]["tags"] == ["tag1", "tag2"]


class TestSerializeToolArgs:
    """Test the serialize_tool_args function."""

    def test_serialize_simple_args(self):
        """Test serializing simple tool arguments."""
        args = {"param1": "value1", "param2": 42}
        result = serialize_tool_args(args)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == args

    def test_serialize_frozen_args(self):
        """Test serializing frozen tool arguments."""
        args = MappingProxyType({"frozen_param": "frozen_value"})
        result = serialize_tool_args(args)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == {"frozen_param": "frozen_value"}

    def test_serialize_args_with_tuple(self):
        """Test serializing args with tuples."""
        args = {"coordinates": (10, 20, 30)}
        result = serialize_tool_args(args)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == {"coordinates": [10, 20, 30]}

    def test_serialize_args_with_frozenset(self):
        """Test serializing args with frozensets."""
        args = {"tags": frozenset(["tag1", "tag2", "tag3"])}
        result = serialize_tool_args(args)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert set(parsed["tags"]) == {"tag1", "tag2", "tag3"}

    def test_serialize_complex_args(self):
        """Test serializing complex tool arguments."""
        args = MappingProxyType(
            {
                "nested": MappingProxyType({"inner": (1, 2, 3)}),
                "set": frozenset([4, 5]),
                "regular": {"key": "value"},
            }
        )
        result = serialize_tool_args(args)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["nested"]["inner"] == [1, 2, 3]
        assert set(parsed["set"]) == {4, 5}
        assert parsed["regular"] == {"key": "value"}
