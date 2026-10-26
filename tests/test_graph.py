"""Unit tests for Tool Graph and Sampler."""

import tempfile
from collections import defaultdict
from pathlib import Path

import pytest

from conversation_factory.graph.chain_export import (
    chains_document,
    save_chains_json,
    write_chains_html,
)
from conversation_factory.registry.models import ParamType

from conversation_factory.graph.tool_graph import ToolGraph
from conversation_factory.graph.sampler import (
    SamplingConstraints,
    ToolChain,
    ToolChainSampler,
)
from tests.fixtures import build_test_registry, build_test_graph


class TestToolGraph:
    def test_build(self):
        graph, registry = build_test_graph()
        assert graph.num_nodes > 0
        assert graph.num_edges > 0

    def test_nodes_match_endpoints(self):
        registry = build_test_registry()
        graph = ToolGraph()
        graph.build(registry)

        expected_count = sum(len(t.endpoints) for t in registry.tools)
        assert graph.num_nodes == expected_count

    def test_same_tool_edges(self):
        graph, registry = build_test_graph()
        hotel_eps = graph.get_endpoints_by_tool("Hotels_API")
        assert len(hotel_eps) == 3

        # Same-tool successors (may be relation same_tool or upgraded data_flow)
        hotel_next = [
            dst
            for _, dst, d in graph.graph.out_edges("Hotels_API/search_hotels", data=True)
            if dst.startswith("Hotels_API/")
        ]
        assert len(hotel_next) > 0

    def test_same_tool_edges_prefer_forward_operation_order(self):
        graph, _ = build_test_graph()
        # Keyword ranks: search < get < book — forward edges stronger than backward
        w_fwd = graph.graph["Hotels_API/search_hotels"]["Hotels_API/book_hotel"]["weight"]
        w_back = graph.graph["Hotels_API/book_hotel"]["Hotels_API/search_hotels"]["weight"]
        assert w_fwd > w_back
        assert graph.graph.has_edge("Hotels_API/search_hotels", "Hotels_API/get_hotel_details")

    def test_category_edges(self):
        graph, registry = build_test_graph()
        neighbors = graph.neighbors("Hotels_API/search_hotels", relation="same_category")
        # Hotels and Flights are both "Travel" category
        flight_neighbors = [n for n in neighbors if n.startswith("Flights_API")]
        assert len(flight_neighbors) > 0

    def test_same_category_edges_not_pure_data_flow(self):
        graph, _ = build_test_graph()
        for _, _, data in graph.graph.edges(data=True):
            if data.get("relation") == "same_category":
                assert data.get("data_flow") is False

    def test_node_data(self):
        graph, registry = build_test_graph()
        data = graph.get_node_data("Hotels_API/search_hotels")
        assert data["tool_name"] == "Hotels_API"
        assert data["category"] == "Travel"

    def test_get_endpoints_by_category(self):
        graph, registry = build_test_graph()
        travel_eps = graph.get_endpoints_by_category("Travel")
        assert len(travel_eps) >= 5  # 3 hotel + 2 flights

    def test_save_and_load(self):
        import tempfile
        graph, registry = build_test_graph()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        graph.save(path)

        loaded = ToolGraph.load(path)
        assert loaded.num_nodes == graph.num_nodes
        assert loaded.num_edges == graph.num_edges

    def test_summary(self):
        graph, _ = build_test_graph()
        summary = graph.summary()
        assert summary["num_nodes"] > 0
        assert summary["num_edges"] > 0
        assert "same_tool" in summary["edge_types"]
        assert "data_flow_edges" in summary
        assert summary["data_flow_edges"] >= 0


class TestToolChainSampler:
    def test_sample_default(self):
        graph, _ = build_test_graph()
        sampler = ToolChainSampler(graph, seed=42)
        chain = sampler.sample()
        assert chain.num_steps >= 1
        assert len(chain.endpoints) > 0

    def test_sample_min_steps(self):
        graph, _ = build_test_graph()
        sampler = ToolChainSampler(graph, seed=42)
        constraints = SamplingConstraints(min_steps=3, max_steps=5)
        chain = sampler.sample(constraints)
        assert chain.num_steps >= 3

    def test_sample_multi_tool(self):
        graph, _ = build_test_graph()
        sampler = ToolChainSampler(graph, seed=42)
        constraints = SamplingConstraints(
            min_steps=3, max_steps=5, require_multi_tool=True, min_distinct_tools=2
        )
        chain = sampler.sample(constraints)
        assert chain.is_multi_tool

    def test_sample_required_category(self):
        graph, _ = build_test_graph()
        sampler = ToolChainSampler(graph, seed=42)
        constraints = SamplingConstraints(
            min_steps=2, max_steps=4, required_categories=["Travel"]
        )
        chain = sampler.sample(constraints)
        travel_eps = set(graph.get_endpoints_by_category("Travel"))
        assert any(ep in travel_eps for ep in chain.endpoints)

    def test_sample_excluded_endpoints(self):
        graph, _ = build_test_graph()
        sampler = ToolChainSampler(graph, seed=42)
        excluded = {"Hotels_API/search_hotels", "Hotels_API/book_hotel"}
        constraints = SamplingConstraints(excluded_endpoints=excluded)
        chain = sampler.sample(constraints)
        assert not any(ep in excluded for ep in chain.endpoints)

    def test_sample_batch(self):
        graph, _ = build_test_graph()
        sampler = ToolChainSampler(graph, seed=42)
        chains = sampler.sample_batch(5)
        assert len(chains) == 5

    def test_sample_batch_diversity(self):
        graph, _ = build_test_graph()
        sampler = ToolChainSampler(graph, seed=42)
        chains = sampler.sample_batch(
            10, diversity_penalty={"max_reuse": 2}
        )
        assert len(chains) == 10
        # Check that diversity penalty works — endpoints shouldn't all be the same
        all_endpoints = set()
        for chain in chains:
            all_endpoints.update(chain.endpoints)
        assert len(all_endpoints) > 3

    def test_deterministic_with_seed(self):
        graph, _ = build_test_graph()
        sampler1 = ToolChainSampler(graph, seed=42)
        sampler2 = ToolChainSampler(graph, seed=42)
        chain1 = sampler1.sample()
        chain2 = sampler2.sample()
        assert chain1.endpoints == chain2.endpoints

    def test_strict_walk_preserves_monotone_tool_ranks(self):
        graph, _ = build_test_graph()
        constraints = SamplingConstraints(
            min_steps=3,
            max_steps=6,
            require_multi_tool=True,
            min_distinct_tools=2,
            strict_graph_walk=True,
        )

        def workflow_ok(endpoints: list[str]) -> bool:
            by_tool: dict[str, list[int]] = defaultdict(list)
            for ep in endpoints:
                t = ep.split("/")[0]
                r = graph._get_operation_rank(ep.split("/")[-1])
                by_tool[t].append(r)
            for ranks in by_tool.values():
                for a, b in zip(ranks, ranks[1:]):
                    if b < a:
                        return False
            return True

        for seed in range(40):
            sampler = ToolChainSampler(graph, seed=seed)
            chain = sampler.sample(constraints)
            assert workflow_ok(chain.endpoints), chain.endpoints

class TestSamplingConstraints:
    def test_defaults(self):
        c = SamplingConstraints()
        assert c.min_steps == 1
        assert c.max_steps == 5

    def test_min_max_validation(self):
        c = SamplingConstraints(min_steps=0, max_steps=2)
        assert c.min_steps == 1  # Auto-corrected
        c2 = SamplingConstraints(min_steps=5, max_steps=3)
        assert c2.max_steps == 5  # Auto-corrected

    def test_strict_graph_walk_default(self):
        assert SamplingConstraints().strict_graph_walk is False
        assert SamplingConstraints(strict_graph_walk=True).strict_graph_walk is True


class TestToolChain:
    def test_properties(self):
        chain = ToolChain(
            endpoints=["A/ep1", "B/ep2", "A/ep3"],
            pattern="sequential",
        )
        assert chain.num_steps == 3
        assert chain.is_multi_tool
        assert chain.is_multi_step
        assert chain.distinct_tools == {"A", "B"}


class TestParamsCompatible:
    def test_generic_id_requires_entity_in_source_tool(self):
        assert ToolGraph._params_compatible(
            "id",
            ParamType.STRING,
            "hotel_id",
            ParamType.STRING,
            source_fqn="Hotels_API/search_hotels",
        )
        assert not ToolGraph._params_compatible(
            "id",
            ParamType.STRING,
            "hotel_id",
            ParamType.STRING,
            source_fqn="Weather_API/get_forecast",
        )


class TestChainExport:
    def test_chains_document_roundtrip_json(self):
        chains = [
            ToolChain(endpoints=["a/x", "b/y"], pattern="sequential", metadata={"k": 1}),
        ]
        c = SamplingConstraints(min_steps=2, max_steps=3)
        doc = chains_document(chains, seed=7, constraints=c, graph_path="/tmp/g.json")
        assert doc["seed"] == 7
        assert doc["num_chains"] == 1
        assert doc["chains"][0]["endpoints"] == ["a/x", "b/y"]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "c.json"
            save_chains_json(p, doc)
            loaded = p.read_text()
            assert "a/x" in loaded

    def test_write_chains_html(self):
        doc = {
            "seed": 1,
            "constraints": {"min_steps": 2},
            "chains": [{"index": 0, "endpoints": ["T/e1", "T/e2"], "pattern": "sequential", "metadata": {}}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            hp = Path(tmp) / "x.html"
            write_chains_html(hp, doc)
            text = hp.read_text()
            assert "T/e1" in text and "T/e2" in text
