#!/usr/bin/env python3
"""Stress test: 50 tools, 200 conversations, full analysis.

Requires OPENAI_API_KEY environment variable.
Produces a full report with diversity metrics, quality scores,
and coverage analysis.
"""

import asyncio
import json
import logging
import random
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conversation_factory.pipeline import ConversationPipeline
from conversation_factory.evaluation.metrics import corpus_quality_summary
from conversation_factory.utils.llm import LLMClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 50 tools across 10 categories — realistic ToolBench-style definitions
# ---------------------------------------------------------------------------

CATEGORIES = [
    "Travel", "Finance", "Weather", "Food", "Entertainment",
    "Shopping", "Health", "Education", "Communication", "Productivity",
]

TOOL_TEMPLATES = {
    "Travel": [
        ("Hotels_API", "Search and book hotels worldwide", [
            ("search_hotels", "GET", [("city", True), ("checkin_date", True), ("max_price", False), ("stars", False)]),
            ("get_hotel_details", "GET", [("hotel_id", True)]),
            ("book_hotel", "POST", [("hotel_id", True), ("checkin_date", True), ("checkout_date", True), ("guests", False)]),
        ]),
        ("Flights_API", "Search and book flights", [
            ("search_flights", "GET", [("origin", True), ("destination", True), ("date", True), ("passengers", False)]),
            ("get_flight_details", "GET", [("flight_id", True)]),
            ("book_flight", "POST", [("flight_id", True), ("passenger_name", True), ("seat_class", False)]),
        ]),
        ("Car_Rental_API", "Rent cars at destinations", [
            ("search_cars", "GET", [("city", True), ("pickup_date", True), ("return_date", True)]),
            ("get_car_details", "GET", [("car_id", True)]),
            ("book_car", "POST", [("car_id", True), ("driver_name", True), ("insurance", False)]),
        ]),
        ("Train_API", "Book train tickets", [
            ("search_trains", "GET", [("from_city", True), ("to_city", True), ("date", True)]),
            ("book_train", "POST", [("train_id", True), ("passenger_name", True), ("class", False)]),
        ]),
        ("Cruise_API", "Search and book cruise trips", [
            ("search_cruises", "GET", [("destination", True), ("departure_date", False)]),
            ("get_cruise_details", "GET", [("cruise_id", True)]),
            ("book_cruise", "POST", [("cruise_id", True), ("cabin_type", True), ("passengers", False)]),
        ]),
    ],
    "Finance": [
        ("Banking_API", "Banking operations and account management", [
            ("get_account_balance", "GET", [("account_id", True)]),
            ("list_transactions", "GET", [("account_id", True), ("start_date", False), ("end_date", False)]),
            ("transfer_money", "POST", [("from_account", True), ("to_account", True), ("amount", True)]),
        ]),
        ("Stock_API", "Stock market data and trading", [
            ("search_stocks", "GET", [("query", True), ("exchange", False)]),
            ("get_stock_price", "GET", [("symbol", True)]),
            ("buy_stock", "POST", [("symbol", True), ("quantity", True), ("order_type", False)]),
        ]),
        ("Crypto_API", "Cryptocurrency data and exchange", [
            ("get_crypto_price", "GET", [("coin", True), ("currency", False)]),
            ("list_exchanges", "GET", [("coin", False)]),
            ("create_order", "POST", [("coin", True), ("amount", True), ("side", True)]),
        ]),
        ("Insurance_API", "Insurance quotes and policies", [
            ("get_quote", "GET", [("policy_type", True), ("coverage_amount", True), ("age", False)]),
            ("list_policies", "GET", [("customer_id", True)]),
            ("purchase_policy", "POST", [("quote_id", True), ("customer_id", True)]),
        ]),
        ("Tax_API", "Tax calculation and filing", [
            ("calculate_tax", "GET", [("income", True), ("filing_status", True), ("state", False)]),
            ("get_deductions", "GET", [("category", False)]),
        ]),
    ],
    "Weather": [
        ("Weather_API", "Current weather and forecasts", [
            ("get_current_weather", "GET", [("city", True)]),
            ("get_forecast", "GET", [("city", True), ("days", False)]),
            ("get_historical", "GET", [("city", True), ("date", True)]),
        ]),
        ("Air_Quality_API", "Air quality index and pollution data", [
            ("get_aqi", "GET", [("city", True)]),
            ("get_pollen_count", "GET", [("city", True), ("date", False)]),
        ]),
        ("Marine_Weather_API", "Ocean and marine weather conditions", [
            ("get_tide_times", "GET", [("location", True), ("date", False)]),
            ("get_wave_forecast", "GET", [("latitude", True), ("longitude", True)]),
        ]),
    ],
    "Food": [
        ("Restaurant_API", "Find and book restaurants", [
            ("search_restaurants", "GET", [("city", True), ("cuisine", False), ("price_range", False)]),
            ("get_restaurant_details", "GET", [("restaurant_id", True)]),
            ("make_reservation", "POST", [("restaurant_id", True), ("date", True), ("party_size", True)]),
        ]),
        ("Recipe_API", "Search and get recipes", [
            ("search_recipes", "GET", [("query", True), ("cuisine", False), ("diet", False)]),
            ("get_recipe_details", "GET", [("recipe_id", True)]),
        ]),
        ("Grocery_API", "Order groceries online", [
            ("search_products", "GET", [("query", True), ("store", False)]),
            ("get_product_details", "GET", [("product_id", True)]),
            ("add_to_cart", "POST", [("product_id", True), ("quantity", True)]),
            ("checkout", "POST", [("cart_id", True), ("delivery_address", True)]),
        ]),
        ("Food_Delivery_API", "Order food delivery", [
            ("search_nearby", "GET", [("location", True), ("cuisine", False)]),
            ("get_menu", "GET", [("restaurant_id", True)]),
            ("place_order", "POST", [("restaurant_id", True), ("items", True), ("delivery_address", True)]),
        ]),
        ("Nutrition_API", "Nutritional information lookup", [
            ("get_nutrition_info", "GET", [("food_name", True)]),
            ("calculate_calories", "GET", [("meal_items", True)]),
        ]),
    ],
    "Entertainment": [
        ("Events_API", "Find local events and buy tickets", [
            ("search_events", "GET", [("city", True), ("category", False), ("date", False)]),
            ("get_event_details", "GET", [("event_id", True)]),
            ("buy_tickets", "POST", [("event_id", True), ("quantity", True), ("seat_type", False)]),
        ]),
        ("Movies_API", "Movie showtimes and tickets", [
            ("search_movies", "GET", [("city", True), ("genre", False)]),
            ("get_showtimes", "GET", [("movie_id", True), ("city", True)]),
            ("book_tickets", "POST", [("showtime_id", True), ("seats", True)]),
        ]),
        ("Music_API", "Music streaming and discovery", [
            ("search_songs", "GET", [("query", True), ("genre", False)]),
            ("get_album_details", "GET", [("album_id", True)]),
            ("create_playlist", "POST", [("name", True), ("song_ids", True)]),
        ]),
        ("Gaming_API", "Game info and purchases", [
            ("search_games", "GET", [("query", True), ("platform", False)]),
            ("get_game_details", "GET", [("game_id", True)]),
            ("purchase_game", "POST", [("game_id", True), ("platform", True)]),
        ]),
        ("Sports_API", "Sports scores and ticket booking", [
            ("get_live_scores", "GET", [("sport", True)]),
            ("search_matches", "GET", [("sport", True), ("date", False), ("league", False)]),
            ("book_match_tickets", "POST", [("match_id", True), ("quantity", True)]),
        ]),
    ],
    "Shopping": [
        ("Ecommerce_API", "Online shopping and orders", [
            ("search_products", "GET", [("query", True), ("category", False), ("max_price", False)]),
            ("get_product_details", "GET", [("product_id", True)]),
            ("add_to_cart", "POST", [("product_id", True), ("quantity", True)]),
            ("place_order", "POST", [("cart_id", True), ("shipping_address", True)]),
        ]),
        ("Auction_API", "Online auction platform", [
            ("search_auctions", "GET", [("query", True), ("category", False)]),
            ("get_auction_details", "GET", [("auction_id", True)]),
            ("place_bid", "POST", [("auction_id", True), ("amount", True)]),
        ]),
        ("Coupon_API", "Find deals and coupons", [
            ("search_coupons", "GET", [("store", False), ("category", False)]),
            ("get_coupon_details", "GET", [("coupon_id", True)]),
        ]),
        ("Fashion_API", "Fashion and clothing search", [
            ("search_clothing", "GET", [("query", True), ("size", False), ("color", False)]),
            ("get_item_details", "GET", [("item_id", True)]),
            ("order_item", "POST", [("item_id", True), ("size", True), ("quantity", False)]),
        ]),
        ("Electronics_API", "Electronics and gadgets", [
            ("search_electronics", "GET", [("query", True), ("brand", False)]),
            ("get_product_specs", "GET", [("product_id", True)]),
            ("compare_products", "GET", [("product_ids", True)]),
        ]),
    ],
    "Health": [
        ("Doctor_API", "Find doctors and book appointments", [
            ("search_doctors", "GET", [("specialty", True), ("city", True), ("insurance", False)]),
            ("get_doctor_profile", "GET", [("doctor_id", True)]),
            ("book_appointment", "POST", [("doctor_id", True), ("date", True), ("time_slot", True)]),
        ]),
        ("Pharmacy_API", "Pharmacy and medication services", [
            ("search_medications", "GET", [("name", True)]),
            ("check_availability", "GET", [("medication_id", True), ("pharmacy_id", True)]),
            ("order_prescription", "POST", [("medication_id", True), ("pharmacy_id", True), ("prescription_id", True)]),
        ]),
        ("Fitness_API", "Fitness tracking and workout plans", [
            ("search_workouts", "GET", [("muscle_group", False), ("difficulty", False)]),
            ("get_workout_details", "GET", [("workout_id", True)]),
            ("log_activity", "POST", [("activity_type", True), ("duration", True), ("calories", False)]),
        ]),
        ("Mental_Health_API", "Mental health resources and booking", [
            ("search_therapists", "GET", [("city", True), ("specialty", False)]),
            ("book_session", "POST", [("therapist_id", True), ("date", True), ("session_type", False)]),
        ]),
    ],
    "Education": [
        ("Courses_API", "Online course discovery and enrollment", [
            ("search_courses", "GET", [("query", True), ("subject", False), ("level", False)]),
            ("get_course_details", "GET", [("course_id", True)]),
            ("enroll", "POST", [("course_id", True), ("student_name", True)]),
        ]),
        ("Library_API", "Digital library and book search", [
            ("search_books", "GET", [("query", True), ("author", False), ("genre", False)]),
            ("get_book_details", "GET", [("book_id", True)]),
            ("borrow_book", "POST", [("book_id", True), ("borrower_id", True)]),
        ]),
        ("Tutoring_API", "Find and book tutors", [
            ("search_tutors", "GET", [("subject", True), ("level", False)]),
            ("get_tutor_profile", "GET", [("tutor_id", True)]),
            ("book_session", "POST", [("tutor_id", True), ("date", True), ("duration", False)]),
        ]),
    ],
    "Communication": [
        ("Email_API", "Email services", [
            ("search_emails", "GET", [("query", True), ("folder", False)]),
            ("get_email", "GET", [("email_id", True)]),
            ("send_email", "POST", [("to", True), ("subject", True), ("body", True)]),
        ]),
        ("Messaging_API", "Instant messaging platform", [
            ("list_conversations", "GET", [("limit", False)]),
            ("get_messages", "GET", [("conversation_id", True), ("limit", False)]),
            ("send_message", "POST", [("conversation_id", True), ("content", True)]),
        ]),
        ("SMS_API", "SMS messaging services", [
            ("send_sms", "POST", [("phone_number", True), ("message", True)]),
            ("get_sms_history", "GET", [("phone_number", True), ("limit", False)]),
        ]),
    ],
    "Productivity": [
        ("Calendar_API", "Calendar and scheduling", [
            ("list_events", "GET", [("date", False), ("calendar_id", False)]),
            ("create_event", "POST", [("title", True), ("start_time", True), ("end_time", True), ("attendees", False)]),
            ("update_event", "POST", [("event_id", True), ("title", False), ("start_time", False)]),
        ]),
        ("Tasks_API", "Task management", [
            ("list_tasks", "GET", [("project_id", False), ("status", False)]),
            ("create_task", "POST", [("title", True), ("description", False), ("due_date", False), ("priority", False)]),
            ("update_task", "POST", [("task_id", True), ("status", True)]),
        ]),
        ("Notes_API", "Note-taking and organization", [
            ("search_notes", "GET", [("query", True), ("tag", False)]),
            ("create_note", "POST", [("title", True), ("content", True), ("tags", False)]),
            ("get_note", "GET", [("note_id", True)]),
        ]),
        ("Maps_API", "Maps and navigation", [
            ("search_places", "GET", [("query", True), ("location", False), ("radius", False)]),
            ("get_directions", "GET", [("origin", True), ("destination", True), ("mode", False)]),
            ("get_place_details", "GET", [("place_id", True)]),
        ]),
    ],
}


def build_tool_json(tool_name, description, endpoints, category):
    """Build a ToolBench-format JSON dict."""
    api_list = []
    for ep_name, method, params in endpoints:
        req_params = [{"name": p[0], "type": "STRING", "description": f"The {p[0].replace('_', ' ')}", "default": ""}
                      for p in params if p[1]]
        opt_params = [{"name": p[0], "type": "STRING", "description": f"Optional {p[0].replace('_', ' ')}", "default": ""}
                      for p in params if not p[1]]
        api_list.append({
            "name": ep_name,
            "url": f"https://api.example.com/{ep_name}",
            "description": f"{ep_name.replace('_', ' ').title()} endpoint",
            "method": method,
            "required_parameters": req_params,
            "optional_parameters": opt_params,
        })
    return {
        "tool_name": tool_name,
        "tool_description": description,
        "title": tool_name.replace("_", " "),
        "standardized_name": tool_name.lower(),
        "category_name": category,
        "api_list": api_list,
    }


def create_50_tool_dataset(output_dir: Path):
    """Write 50 tools in ToolBench category-directory layout."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tool_count = 0
    for category, tools in TOOL_TEMPLATES.items():
        cat_dir = output_dir / category
        cat_dir.mkdir(exist_ok=True)
        for tool_name, description, endpoints in tools:
            tool_json = build_tool_json(tool_name, description, endpoints, category)
            with open(cat_dir / f"{tool_name.lower()}.json", "w") as f:
                json.dump(tool_json, f, indent=2)
            tool_count += 1
    return tool_count


async def run_stress_test():
    start = time.time()
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "stress_test_data"
    output_a = base_dir / "stress_output_run_a"
    output_b = base_dir / "stress_output_run_b"

    # ----------------------------------------------------------------
    # 1. Create 50-tool dataset
    # ----------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: Creating 50-tool dataset")
    print("=" * 70)
    n_tools = create_50_tool_dataset(data_dir)
    print(f"  Created {n_tools} tools across {len(CATEGORIES)} categories\n")

    # ----------------------------------------------------------------
    # 2. Run A — steering OFF
    # ----------------------------------------------------------------
    print("=" * 70)
    print("STEP 2: Run A — 200 conversations, steering OFF")
    print("=" * 70)
    pipeline_a = ConversationPipeline(
        data_dir=data_dir, output_dir=output_a,
        seed=42, cross_conversation_steering=False,
    )

    build_a = pipeline_a.build()
    print(f"  Registry: {build_a['registry']['num_tools']} tools, {build_a['registry']['num_endpoints']} endpoints")
    print(f"  Graph: {build_a['graph']['num_edges']} edges ({build_a['graph'].get('data_flow_edges', '?')} data-flow)")

    convs_a = await pipeline_a.generate(
        num_conversations=200, min_steps=1, max_steps=8,
        multi_step_ratio=0.65, short_conversation_ratio=0.15,
        min_multi_step_calls=3, concurrency=5,
    )
    time_a = time.time() - start
    print(f"  Generated {len(convs_a)} conversations in {time_a:.1f}s\n")

    # ----------------------------------------------------------------
    # 3. Run B — steering ON
    # ----------------------------------------------------------------
    print("=" * 70)
    print("STEP 3: Run B — 200 conversations, steering ON")
    print("=" * 70)
    t2 = time.time()
    pipeline_b = ConversationPipeline(
        data_dir=data_dir, output_dir=output_b,
        seed=42, cross_conversation_steering=True,
    )

    pipeline_b.build()
    convs_b = await pipeline_b.generate(
        num_conversations=200, min_steps=1, max_steps=8,
        multi_step_ratio=0.65, short_conversation_ratio=0.15,
        min_multi_step_calls=3, concurrency=5,
    )
    time_b = time.time() - t2
    print(f"  Generated {len(convs_b)} conversations in {time_b:.1f}s\n")

    # ----------------------------------------------------------------
    # 4. Compute metrics
    # ----------------------------------------------------------------
    print("=" * 70)
    print("STEP 4: Analysis")
    print("=" * 70)

    cats = list(CATEGORIES)
    all_eps_a = list(pipeline_a.graph.graph.nodes())
    all_eps_b = list(pipeline_b.graph.graph.nodes())

    metrics_a = corpus_quality_summary(convs_a, cats, all_eps_a)
    metrics_b = corpus_quality_summary(convs_b, cats, all_eps_b)

    # Compute extra stats not in corpus_quality_summary
    def _extra_stats(convs, all_eps):
        used = set()
        cat_counts = {}
        tc_list = []
        for c in convs:
            m = c.get("metadata", {})
            for t in m.get("tools_used", []):
                used.add(t)
            for cat in m.get("categories", []):
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
            tc_list.append(m.get("num_tool_calls", 0))
        return {
            "unique_eps": len(used), "total_eps": len(all_eps),
            "cat_counts": cat_counts, "tc_list": tc_list,
        }

    extra_a = _extra_stats(convs_a, all_eps_a)
    extra_b = _extra_stats(convs_b, all_eps_b)

    def fmt(v):
        if isinstance(v, float):
            return f"{v * 100:.1f}%" if v <= 1.0 else f"{v:.3f}"
        return str(v)

    print(f"\n{'Metric':<40} {'Run A (no steer)':<20} {'Run B (steered)':<20}")
    print("-" * 80)
    rows = [
        ("Conversations generated",         len(convs_a),                                  len(convs_b)),
        ("Multi-step ratio (>=3 calls)",     metrics_a["coverage"]["multi_step_ratio"],     metrics_b["coverage"]["multi_step_ratio"]),
        ("Multi-tool ratio (>=2 tools)",     metrics_a["coverage"]["multi_tool_ratio"],     metrics_b["coverage"]["multi_tool_ratio"]),
        ("Joint (multi-step + multi-tool)",  metrics_a["coverage"]["joint_multi_step_and_multi_tool_ratio"], metrics_b["coverage"]["joint_multi_step_and_multi_tool_ratio"]),
        ("Mean tool calls / conversation",   metrics_a["coverage"]["mean_tool_calls"],      metrics_b["coverage"]["mean_tool_calls"]),
        ("Mean distinct tools / conv",       metrics_a["coverage"]["mean_distinct_tools"],  metrics_b["coverage"]["mean_distinct_tools"]),
        ("Mean turns / conversation",        metrics_a["coverage"]["mean_turns"],           metrics_b["coverage"]["mean_turns"]),
        ("Mean judge score",                 metrics_a["quality"]["mean_judge_score"],      metrics_b["quality"]["mean_judge_score"]),
        ("Endpoint entropy (norm)",          metrics_a["diversity"].get("endpoint_entropy_normalized", "N/A"), metrics_b["diversity"].get("endpoint_entropy_normalized", "N/A")),
        ("Category coverage",               metrics_a["diversity"]["category_coverage"],   metrics_b["diversity"]["category_coverage"]),
        ("Category entropy",                 metrics_a["diversity"].get("category_entropy", "N/A"), metrics_b["diversity"].get("category_entropy", "N/A")),
        ("Mean pairwise Jaccard",            metrics_a["diversity"]["mean_pairwise_jaccard_distance"], metrics_b["diversity"]["mean_pairwise_jaccard_distance"]),
        ("Unique endpoints used",            extra_a["unique_eps"],                         extra_b["unique_eps"]),
        ("Total endpoints available",        extra_a["total_eps"],                          extra_b["total_eps"]),
    ]
    for label, va, vb in rows:
        print(f"  {label:<40} {fmt(va):<20} {fmt(vb):<20}")

    # Tool call distribution
    import statistics
    print(f"\n{'Tool calls per conversation':<40} {'Run A':<20} {'Run B':<20}")
    print("-" * 80)
    for label, fn in [("min", min), ("max", max), ("mean", statistics.mean), ("median", statistics.median)]:
        va = fn(extra_a["tc_list"]) if extra_a["tc_list"] else 0
        vb = fn(extra_b["tc_list"]) if extra_b["tc_list"] else 0
        print(f"  {label:<40} {va:<20.1f} {vb:<20.1f}")

    # Category breakdown
    print(f"\n{'Category usage (conversations)':<40} {'Run A':<20} {'Run B':<20}")
    print("-" * 80)
    all_cat_names = sorted(set(list(extra_a["cat_counts"].keys()) + list(extra_b["cat_counts"].keys())))
    for cat in all_cat_names:
        va = extra_a["cat_counts"].get(cat, 0)
        vb = extra_b["cat_counts"].get(cat, 0)
        print(f"  {cat:<40} {va:<20} {vb:<20}")

    # ----------------------------------------------------------------
    # 5. Pass/Fail checks
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: Pass/Fail Checks")
    print("=" * 70)

    checks = [
        ("Generated >= 200 convs (A)",     len(convs_a) >= 200),
        ("Generated >= 200 convs (B)",     len(convs_b) >= 200),
        ("Multi-step >= 50% (A)",          metrics_a["coverage"]["multi_step_ratio"] >= 0.50),
        ("Multi-step >= 45% (B)",          metrics_b["coverage"]["multi_step_ratio"] >= 0.45),
        ("Multi-tool >= 50% (A)",          metrics_a["coverage"]["multi_tool_ratio"] >= 0.50),
        ("Multi-tool >= 50% (B)",          metrics_b["coverage"]["multi_tool_ratio"] >= 0.50),
        ("Mean judge >= 3.0 (A)",          metrics_a["quality"]["mean_judge_score"] >= 3.0),
        ("Mean judge >= 3.0 (B)",          metrics_b["quality"]["mean_judge_score"] >= 3.0),
        ("Category coverage >= 80% (B)",   metrics_b["diversity"]["category_coverage"] >= 0.80),
        ("Unique endpoints B >= A",        extra_b["unique_eps"] >= extra_a["unique_eps"]),
    ]
    all_pass = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {label}")

    total_time = time.time() - start
    print(f"\nTotal time: {total_time:.1f}s")

    # Save full report
    report = {
        "run_a": {**metrics_a, "num_conversations": len(convs_a), "time_seconds": round(time_a, 1)},
        "run_b": {**metrics_b, "num_conversations": len(convs_b), "time_seconds": round(time_b, 1)},
        "steering_stats": pipeline_b.steering.get_stats() if hasattr(pipeline_b.steering, "get_stats") else {},
        "total_time_seconds": round(total_time, 1),
        "all_checks_passed": all_pass,
    }
    report_path = base_dir / "stress_test_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to: {report_path}")

    if all_pass:
        print("\n ALL CHECKS PASSED")
    else:
        print("\n SOME CHECKS FAILED")
    return all_pass


if __name__ == "__main__":
    result = asyncio.run(run_stress_test())
    sys.exit(0 if result else 1)
