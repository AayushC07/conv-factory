"""Unit tests for the offline execution model."""

import pytest

from conversation_factory.execution.session import SessionState
from conversation_factory.execution.mock_executor import MockExecutor
from tests.fixtures import build_test_registry


class TestSessionState:
    def test_record_and_retrieve(self):
        session = SessionState()
        session.record_call(
            "Hotels_API/search_hotels",
            {"city": "Paris"},
            {"results": [{"id": "htl_001", "name": "Hotel A", "price": 150}]},
        )
        assert len(session.history) == 1
        assert session.get_value("results") is not None

    def test_id_extraction(self):
        session = SessionState()
        session.record_call(
            "Hotels_API/search_hotels",
            {"city": "Paris"},
            {"results": [
                {"id": "htl_001", "name": "Hotel A"},
                {"id": "htl_002", "name": "Hotel B"},
            ]},
        )
        ids = session.get_available_ids()
        # Should find ID-like values
        assert any("id" in k.lower() for k in ids)

    def test_multi_step_chaining(self):
        session = SessionState()
        session.record_call(
            "Hotels_API/search_hotels",
            {"city": "Paris"},
            {"results": [{"id": "htl_001", "name": "Hotel A"}]},
        )
        session.record_call(
            "Hotels_API/book_hotel",
            {"hotel_id": "htl_001"},
            {"booking_id": "bk_001", "status": "confirmed"},
        )
        assert len(session.history) == 2
        assert session.get_value("booking_id") == "bk_001"

    def test_context_summary(self):
        session = SessionState()
        session.record_call(
            "Hotels_API/search_hotels",
            {"city": "Paris"},
            {"results": [{"id": "htl_001"}]},
        )
        summary = session.get_context_summary()
        assert "Hotels_API/search_hotels" in summary

    def test_generate_id(self):
        session = SessionState()
        id1 = session.generate_id("hotel")
        id2 = session.generate_id("hotel")
        assert id1 != id2
        assert id1.startswith("hotel_")

    def test_reset(self):
        session = SessionState()
        session.record_call("test", {}, {"id": "123"})
        session.reset()
        assert len(session.history) == 0
        assert len(session.values) == 0

    def test_get_response_for_endpoint(self):
        session = SessionState()
        session.record_call("ep1", {}, {"a": 1})
        session.record_call("ep2", {}, {"b": 2})
        assert session.get_response_for_endpoint("ep1") == {"a": 1}
        assert session.get_response_for_endpoint("ep2") == {"b": 2}
        assert session.get_response_for_endpoint("ep3") is None


class TestMockExecutor:
    def test_schema_response(self):
        registry = build_test_registry()
        executor = MockExecutor(registry, llm_client=None, use_llm=False, seed=42)
        session = SessionState()

        import asyncio
        response = asyncio.run(executor.execute(
            "Hotels_API/search_hotels",
            {"city": "Paris"},
            session,
        ))
        assert isinstance(response, dict)
        assert "results" in response

    def test_create_response(self):
        registry = build_test_registry()
        executor = MockExecutor(registry, llm_client=None, use_llm=False, seed=42)
        session = SessionState()

        import asyncio
        response = asyncio.run(executor.execute(
            "Hotels_API/book_hotel",
            {"hotel_id": "htl_001", "check_in": "2026-04-11"},
            session,
        ))
        assert "status" in response
        assert isinstance(response["status"], str)

    def test_unknown_endpoint(self):
        registry = build_test_registry()
        executor = MockExecutor(registry, llm_client=None, use_llm=False, seed=42)
        session = SessionState()

        import asyncio
        response = asyncio.run(executor.execute(
            "Unknown/endpoint",
            {"arg": "val"},
            session,
        ))
        assert response["status"] == "success"

    def test_session_state_updated(self):
        registry = build_test_registry()
        executor = MockExecutor(registry, llm_client=None, use_llm=False, seed=42)
        session = SessionState()

        import asyncio
        asyncio.run(executor.execute(
            "Hotels_API/search_hotels",
            {"city": "Paris"},
            session,
        ))
        assert len(session.history) == 1

    def test_chain_consistency(self):
        """Verify that IDs from step 1 are available for step 2."""
        registry = build_test_registry()
        executor = MockExecutor(registry, llm_client=None, use_llm=False, seed=42)
        session = SessionState()

        import asyncio
        search_response = asyncio.run(executor.execute(
            "Hotels_API/search_hotels",
            {"city": "Paris"},
            session,
        ))
        # Extract an ID from search results
        hotel_id = search_response.get("results", [{}])[0].get("id")
        assert hotel_id is not None

        # Book using that ID
        book_response = asyncio.run(executor.execute(
            "Hotels_API/book_hotel",
            {"hotel_id": hotel_id},
            session,
        ))
        assert book_response is not None
        assert len(session.history) == 2

    def test_three_step_hotel_chain_ids_thread(self):
        """search → get_hotel_details → book_hotel uses one hotel_id throughout."""
        registry = build_test_registry()
        executor = MockExecutor(registry, llm_client=None, use_llm=False, seed=42)
        session = SessionState()

        import asyncio

        async def chain():
            r1 = await executor.execute(
                "Hotels_API/search_hotels", {"city": "Paris"}, session
            )
            hid = r1["results"][0]["id"]
            await executor.execute(
                "Hotels_API/get_hotel_details", {"hotel_id": hid}, session
            )
            r3 = await executor.execute(
                "Hotels_API/book_hotel",
                {
                    "hotel_id": hid,
                    "check_in": "2026-04-11",
                    "check_out": "2026-04-13",
                    "guest_name": "Test User",
                },
                session,
            )
            return hid, r3

        hotel_id, book = asyncio.run(chain())
        assert book.get("hotel_id") == hotel_id
        assert len(session.history) == 3
