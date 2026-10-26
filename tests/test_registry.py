"""Unit tests for the Tool Registry."""

import json
import tempfile
from pathlib import Path

import pytest

from conversation_factory.registry.models import (
    Endpoint,
    Parameter,
    ParamType,
    Tool,
    ToolCategory,
)
from conversation_factory.registry.loader import ToolBenchLoader
from conversation_factory.registry.registry import ToolRegistry
from tests.fixtures import (
    SAMPLE_TOOLBENCH_TOOLS,
    create_sample_toolbench_dir,
    create_category_layout_dir,
)


class TestParameter:
    def test_normalize_type_string(self):
        p = Parameter(name="test", param_type="str")
        assert p.param_type == ParamType.STRING

    def test_normalize_type_int(self):
        p = Parameter(name="test", param_type="int")
        assert p.param_type == ParamType.INTEGER

    def test_normalize_type_unknown(self):
        p = Parameter(name="test", param_type="unknown_type")
        assert p.param_type == ParamType.STRING

    def test_normalize_type_enum(self):
        p = Parameter(name="test", param_type=ParamType.BOOLEAN)
        assert p.param_type == ParamType.BOOLEAN


class TestEndpoint:
    def test_required_parameters(self):
        ep = Endpoint(
            name="test",
            parameters=[
                Parameter(name="a", required=True),
                Parameter(name="b", required=False),
                Parameter(name="c", required=True),
            ],
        )
        assert len(ep.required_parameters) == 2
        assert len(ep.optional_parameters) == 1


class TestTool:
    def test_auto_id(self):
        tool = Tool(tool_name="MyTool")
        assert tool.tool_id != ""
        assert len(tool.tool_id) == 12

    def test_full_endpoint_ids(self):
        tool = Tool(
            tool_name="MyTool",
            endpoints=[Endpoint(name="ep1"), Endpoint(name="ep2")],
        )
        assert tool.full_endpoint_ids == ["MyTool/ep1", "MyTool/ep2"]

    def test_get_endpoint(self):
        tool = Tool(
            tool_name="MyTool",
            endpoints=[Endpoint(name="ep1"), Endpoint(name="ep2")],
        )
        assert tool.get_endpoint("ep1") is not None
        assert tool.get_endpoint("ep3") is None


class TestToolBenchLoader:
    def test_load_single_file(self):
        tmpdir = create_sample_toolbench_dir()
        loader = ToolBenchLoader(tmpdir)
        tools = loader.load_all()
        assert len(tools) == 6
        assert any(t.tool_name == "Hotels_API" for t in tools)

    def test_load_category_layout(self):
        tmpdir = create_category_layout_dir()
        loader = ToolBenchLoader(tmpdir)
        tools = loader.load_all()
        assert len(tools) == 6

    def test_load_with_max_tools(self):
        tmpdir = create_sample_toolbench_dir()
        loader = ToolBenchLoader(tmpdir)
        tools = loader.load_all(max_tools=3)
        assert len(tools) == 3

    def test_handles_missing_fields(self):
        tmpdir = tempfile.mkdtemp()
        tool_data = [
            {
                "tool_name": "Minimal",
                "api_list": [
                    {"name": "do_thing"},  # Missing description, params, etc.
                ],
            }
        ]
        with open(Path(tmpdir) / "tools.json", "w") as f:
            json.dump(tool_data, f)

        loader = ToolBenchLoader(tmpdir)
        tools = loader.load_all()
        assert len(tools) == 1
        assert tools[0].tool_name == "Minimal"
        assert len(tools[0].endpoints) == 1

    def test_handles_invalid_json(self):
        tmpdir = tempfile.mkdtemp()
        with open(Path(tmpdir) / "bad.json", "w") as f:
            f.write("not json{{{")

        loader = ToolBenchLoader(tmpdir)
        tools = loader.load_all()
        assert len(tools) == 0

    def test_parameter_parsing(self):
        tmpdir = create_sample_toolbench_dir()
        loader = ToolBenchLoader(tmpdir)
        tools = loader.load_all()

        hotel_tool = next(t for t in tools if t.tool_name == "Hotels_API")
        search_ep = hotel_tool.get_endpoint("search_hotels")
        assert search_ep is not None
        assert len(search_ep.required_parameters) == 1
        assert search_ep.required_parameters[0].name == "city"
        assert len(search_ep.optional_parameters) >= 3


class TestToolRegistry:
    def test_add_and_get_tool(self):
        registry = ToolRegistry()
        tool = Tool(tool_name="Test", endpoints=[Endpoint(name="ep1")])
        registry.add_tool(tool)
        assert registry.get_tool("Test") is not None
        assert registry.num_tools == 1

    def test_get_endpoint(self):
        registry = ToolRegistry()
        tool = Tool(tool_name="Test", endpoints=[Endpoint(name="ep1")])
        registry.add_tool(tool)
        result = registry.get_endpoint("Test/ep1")
        assert result is not None
        assert result[0].tool_name == "Test"
        assert result[1].name == "ep1"

    def test_categories(self):
        registry = ToolRegistry()
        registry.add_tool(
            Tool(tool_name="A", category=ToolCategory(name="Cat1"), endpoints=[Endpoint(name="e")])
        )
        registry.add_tool(
            Tool(tool_name="B", category=ToolCategory(name="Cat2"), endpoints=[Endpoint(name="e")])
        )
        assert "Cat1" in registry.categories
        assert "Cat2" in registry.categories

    def test_save_and_load(self):
        registry = ToolRegistry()
        tool = Tool(
            tool_name="Test",
            description="A test tool",
            endpoints=[
                Endpoint(
                    name="ep1",
                    description="An endpoint",
                    parameters=[Parameter(name="x", required=True)],
                )
            ],
        )
        registry.add_tool(tool)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        registry.save(path)

        loaded = ToolRegistry.load(path)
        assert loaded.num_tools == 1
        result = loaded.get_endpoint("Test/ep1")
        assert result is not None

    def test_search_tools(self):
        registry = ToolRegistry()
        registry.add_tool(Tool(tool_name="Hotels_API", description="Book hotels", endpoints=[Endpoint(name="e")]))
        registry.add_tool(Tool(tool_name="Flights_API", description="Book flights", endpoints=[Endpoint(name="e")]))

        results = registry.search_tools("hotel")
        assert len(results) == 1
        assert results[0].tool_name == "Hotels_API"

    def test_summary(self):
        tmpdir = create_sample_toolbench_dir()
        loader = ToolBenchLoader(tmpdir)
        tools = loader.load_all()
        registry = ToolRegistry()
        registry.add_tools(tools)

        summary = registry.summary()
        assert summary["num_tools"] == 6
        assert summary["num_endpoints"] > 0
        assert summary["num_categories"] > 0


class TestRealToolBenchFormat:
    """Tests that exercise the actual ToolBench JSON format from OpenBMB/ToolBench."""

    def _make_toolbench_tool(self):
        """A realistic ToolBench tool definition matching the real repo format."""
        return {
            "tool_description": "Provides hotel search, details, and booking functionality.",
            "tool_name": "booking_com",
            "title": "Booking.com",
            "standardized_name": "booking_com",
            "api_list": [
                {
                    "name": "search_hotels",
                    "url": "https://api.example.com/search",
                    "description": "Search for available hotels in a city",
                    "method": "GET",
                    "required_parameters": [
                        {"name": "city", "type": "STRING", "description": "The city to search in", "default": ""},
                        {"name": "checkin_date", "type": "STRING", "description": "Check-in date", "default": ""},
                    ],
                    "optional_parameters": [
                        {"name": "max_price", "type": "NUMBER", "description": "Maximum price", "default": "500"},
                        {"name": "num_adults", "type": "NUMBER", "description": "Number of adults", "default": "2"},
                        {"name": "sort_by", "type": "STRING", "description": "Sort order", "default": "popularity"},
                    ],
                    "template_response": {
                        "results": [{"hotel_id": "abc", "name": "Test Hotel", "price": 100}],
                        "total": 42,
                        "status": "ok",
                    },
                },
                {
                    "name": "get_hotel_details",
                    "url": "https://api.example.com/details",
                    "description": "Get detailed information about a hotel",
                    "method": "GET",
                    "required_parameters": [
                        {"name": "hotel_id", "type": "STRING", "description": "The hotel ID", "default": ""},
                    ],
                    "optional_parameters": [],
                },
                {
                    "name": "book_hotel",
                    "url": "https://api.example.com/book",
                    "description": "Book a hotel room",
                    "method": "POST",
                    "required_parameters": [
                        {"name": "hotel_id", "type": "STRING", "description": "Hotel to book", "default": ""},
                        {"name": "checkin_date", "type": "STRING", "description": "Check-in date", "default": ""},
                        {"name": "checkout_date", "type": "STRING", "description": "Check-out date", "default": ""},
                    ],
                    "optional_parameters": [
                        {"name": "num_rooms", "type": "NUMBER", "description": "Number of rooms", "default": "1"},
                        {"name": "special_requests", "type": "STRING", "description": "Special requests", "default": ""},
                    ],
                },
            ],
        }

    def test_load_real_format(self):
        """Loader correctly parses real ToolBench format with uppercase types and template_response."""
        tmpdir = tempfile.mkdtemp()
        tool_data = self._make_toolbench_tool()
        with open(Path(tmpdir) / "booking_com.json", "w") as f:
            json.dump(tool_data, f)

        loader = ToolBenchLoader(tmpdir)
        tools = loader.load_all()
        assert len(tools) == 1
        tool = tools[0]
        assert tool.tool_name == "booking_com"
        assert len(tool.endpoints) == 3

    def test_uppercase_types_normalized(self):
        """ToolBench uses STRING/NUMBER/BOOLEAN — should normalize to lowercase enums."""
        tmpdir = tempfile.mkdtemp()
        with open(Path(tmpdir) / "tool.json", "w") as f:
            json.dump(self._make_toolbench_tool(), f)

        loader = ToolBenchLoader(tmpdir)
        tools = loader.load_all()
        ep = tools[0].endpoints[0]  # search_hotels
        city_param = next(p for p in ep.parameters if p.name == "city")
        max_price = next(p for p in ep.parameters if p.name == "max_price")
        assert city_param.param_type == ParamType.STRING
        assert max_price.param_type == ParamType.NUMBER
        assert city_param.required is True
        assert max_price.required is False

    def test_empty_string_defaults_normalized(self):
        """ToolBench uses empty string defaults — should be treated as None."""
        tmpdir = tempfile.mkdtemp()
        with open(Path(tmpdir) / "tool.json", "w") as f:
            json.dump(self._make_toolbench_tool(), f)

        loader = ToolBenchLoader(tmpdir)
        tools = loader.load_all()
        ep = tools[0].endpoints[0]
        city_param = next(p for p in ep.parameters if p.name == "city")
        assert city_param.default is None  # empty string default → None

    def test_template_response_parsed(self):
        """template_response from ToolBench is used as response schema."""
        tmpdir = tempfile.mkdtemp()
        with open(Path(tmpdir) / "tool.json", "w") as f:
            json.dump(self._make_toolbench_tool(), f)

        loader = ToolBenchLoader(tmpdir)
        tools = loader.load_all()
        ep = tools[0].endpoints[0]  # search_hotels — has template_response
        field_names = {f.name for f in ep.response_schema}
        assert "results" in field_names
        assert "total" in field_names
        assert "status" in field_names

    def test_standardized_name_used(self):
        """If only standardized_name is present, it should be used as tool_name."""
        tool_data = {
            "standardized_name": "my_custom_tool",
            "tool_description": "A test tool",
            "api_list": [
                {
                    "name": "do_thing",
                    "description": "Does a thing",
                    "method": "GET",
                    "required_parameters": [],
                    "optional_parameters": [],
                }
            ],
        }
        tmpdir = tempfile.mkdtemp()
        with open(Path(tmpdir) / "tool.json", "w") as f:
            json.dump(tool_data, f)

        loader = ToolBenchLoader(tmpdir)
        tools = loader.load_all()
        assert len(tools) == 1
        assert tools[0].tool_name == "my_custom_tool"

    def test_category_directory_layout(self):
        """Real ToolBench uses Category/tool_name.json layout."""
        tmpdir = tempfile.mkdtemp()
        cat_dir = Path(tmpdir) / "Trip Advisors"
        cat_dir.mkdir()
        with open(cat_dir / "booking_com.json", "w") as f:
            json.dump(self._make_toolbench_tool(), f)

        loader = ToolBenchLoader(tmpdir)
        tools = loader.load_all()
        assert len(tools) == 1
        assert tools[0].category.name == "Trip Advisors"

    def test_malformed_json_skipped(self):
        """Malformed JSON files should be skipped, not crash the loader."""
        tmpdir = tempfile.mkdtemp()
        with open(Path(tmpdir) / "bad.json", "w") as f:
            f.write("{broken json!!")
        with open(Path(tmpdir) / "good.json", "w") as f:
            json.dump(self._make_toolbench_tool(), f)

        loader = ToolBenchLoader(tmpdir)
        tools = loader.load_all()
        assert len(tools) == 1  # bad.json skipped, good.json loaded

    def test_response_schema_crash_resistance(self):
        """response_schema with non-dict properties should not crash."""
        tool_data = self._make_toolbench_tool()
        tool_data["api_list"][0]["response_schema"] = {
            "type": "array",
            "items": "string",  # Not a dict — should not crash
        }
        tmpdir = tempfile.mkdtemp()
        with open(Path(tmpdir) / "tool.json", "w") as f:
            json.dump(tool_data, f)

        loader = ToolBenchLoader(tmpdir)
        tools = loader.load_all()
        assert len(tools) == 1  # did not crash
