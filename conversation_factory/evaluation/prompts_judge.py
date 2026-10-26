"""Judge and repair prompt templates (separated for clarity)."""

JUDGE_SYSTEM = """You are an expert evaluator of AI tool-use conversations.
Score each conversation on the specified dimensions.

Output JSON with this exact schema:
{
  "scores": {
    "naturalness": <float 1-5>,
    "tool_correctness": <float 1-5>,
    "task_completion": <float 1-5>,
    "coherence": <float 1-5>
  },
  "explanation": "Brief justification for scores",
  "issues": ["list of specific issues found, if any"]
}

Scoring rubric (be strict — 5.0 is rare):
- 5.0 = flawless on that dimension; no issues worth mentioning.
- 4.0 = good with minor issues (small inconsistencies, one redundant tool call, slightly generic wording).
- 3.0 = acceptable but clearly needs improvement (hallucinated details vs tool output, weak chaining, or awkward flow).
- 2.0 = significant problems; 1.0 = broken.

Dimension guides:
- naturalness (1-5): Flow and language. 5 only if dialogue feels human and varied; deduct for repetition, filler loops, or robotic patterns.
- tool_correctness (1-5): Valid calls, correct arguments, and assistant **user-facing** text must **numerically and lexically match** tool outputs: every price, count, date, phone number, venue name, and ticket/total line in assistant prose must appear verbatim (or clearly paraphrased without changing numbers) in the preceding tool JSON. Deduct heavily if any number or name in assistant text cannot be traced to a tool message, or repeats the same call with identical arguments without reason.
- task_completion (1-5): User goal reached or clearly addressed. Deduct if the task is abandoned or key steps are missing.
- coherence (1-5): Logical tool order and referencing prior outputs. Deduct for redundant identical tool calls, contradictions, or ignoring tool data."""


JUDGE_PROMPT = """Evaluate this tool-use conversation:

{conversation}

Score on all four dimensions (1-5 each). Output ONLY valid JSON matching the schema."""


REPAIR_SYSTEM = """You are a conversation repair agent. Given a conversation that scored poorly
on specific dimensions, fix the identified issues while preserving the overall structure and tools used.

Output JSON with this schema:
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "content": {...}},
    ...
  ]
}

Rules:
- Preserve the same tools and endpoints used
- **Schema:** In every assistant message with tools, use ``"tool_calls": [{"endpoint": "ToolName_API/operation", "arguments": {...}}]``. Never use ``tool``/``args`` instead of ``endpoint``/``arguments``. Use full FQNs (e.g. ``Events_API/search_events``, not ``Events/search_events``). Include one ``tool`` role message per tool call, with JSON ``content`` matching the tool output shape.
- Fix naturalness issues by making language more conversational
- Fix tool_correctness issues: assistant **user-facing text** must list only names, prices, addresses, venues, and times that appear in the **tool** messages (same spelling; no different venue names or ticket counts); replace any invented rows with verbatim data from the nearest prior tool result
- Fix incorrect **book_flight** arguments: **flight_id** must come from **search_flights** output, never from event or unrelated listings
- Fix **Hotels_API/book_hotel**: **hotel_id** must come from **Hotels_API/search_hotels** results in the transcript, never an airport/city code (e.g. MCO, MIA) or invented id
- Fix **Restaurant_API/make_reservation**: **restaurant_id** must come from **Restaurant_API/search_restaurants** or **get_restaurant_details** in the transcript — never a restaurant **name** or a place from **Maps_API/search_places** alone
- Fix **Maps_API/get_directions**: the assistant's next user-facing reply must copy **each** step **instruction** string from the tool JSON (``steps`` or ``directions`` array); do not substitute different street names, exits, or highways
- Fix task_completion issues by ensuring the conversation reaches a conclusion
- Fix coherence issues by properly chaining tool outputs to inputs; remove redundant duplicate tool calls that repeat the same fetch without new user intent"""


REPAIR_PROMPT = """This conversation scored poorly and needs repair.

Original conversation:
{conversation}

Issues identified:
{issues}

Low-scoring dimensions:
{low_scores}

Available tools (keep the same endpoints):
{tool_descriptions}

Fix the issues and output the repaired conversation. Output ONLY valid JSON."""
