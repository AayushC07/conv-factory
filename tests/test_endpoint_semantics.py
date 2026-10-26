"""Generic endpoint FQN semantics (listing vs commit, registry resolve)."""

from conversation_factory.registry.endpoint_semantics import (
    endpoint_operation_name,
    is_likely_commit_operation,
    is_likely_listing_search_operation,
    resolve_endpoint_fqn,
)
from conversation_factory.registry.loader import ToolBenchLoader
from conversation_factory.registry.registry import ToolRegistry
from tests.fixtures import create_sample_toolbench_dir


def _reg() -> ToolRegistry:
    loader = ToolBenchLoader(create_sample_toolbench_dir())
    r = ToolRegistry()
    r.add_tools(loader.load_all(max_tools=6))
    return r


def test_endpoint_operation_name():
    assert endpoint_operation_name("Foo_API/bar_baz") == "bar_baz"


def test_listing_search_heuristic():
    assert is_likely_listing_search_operation("AnyTool/search_items") is True
    assert is_likely_listing_search_operation("AnyTool/query_catalog") is True
    assert is_likely_listing_search_operation("AnyTool/get_item_details") is False


def test_commit_heuristic():
    assert is_likely_commit_operation("AnyTool/book_foo") is True
    assert is_likely_commit_operation("AnyTool/post_order") is True
    assert is_likely_commit_operation("AnyTool/get_foo") is False
    assert is_likely_commit_operation("AnyTool/search_foo") is False


def test_resolve_only_when_registered():
    reg = _reg()
    assert resolve_endpoint_fqn("Hotels_API/search_hotels", reg) == "Hotels_API/search_hotels"
    typo = "Hotels/search_hotels"
    if reg.get_endpoint("Hotels_API/search_hotels"):
        assert resolve_endpoint_fqn(typo, reg) == "Hotels_API/search_hotels"
    else:
        assert resolve_endpoint_fqn(typo, reg) == typo
