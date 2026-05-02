"""
Task 03: AI-Powered Portfolio Explainer
========================================
A production-quality script that uses the Gemini API to generate
plain-English explanations of portfolio risk, structured insights,
and actionable suggestions.

Python  : 3.10+
API     : Google Gemini
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ──────────────────────────────────────────────────────────────
#  Third-party imports
# ──────────────────────────────────────────────────────────────

try:
    import yfinance as yf
except ImportError:
    print(
        "❌  Missing dependency: yfinance\n"
        "   Install it with:  pip install yfinance"
    )
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
#  Custom Exceptions
# ──────────────────────────────────────────────────────────────

class PortfolioInputError(Exception):
    """Raised when portfolio input validation fails."""


class APIError(Exception):
    """Raised when the Gemini API call fails."""


class ResponseParsingError(Exception):
    """Raised when the LLM response cannot be parsed."""


# ──────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────

VALID_TONES = ("beginner", "experienced", "expert")

DEFAULT_MODEL = "gemini-1.5-flash"

REQUIRED_FIELDS = {"summary", "good_practice", "improvement_suggestion",
                   "verdict", "asset_analysis"}

ASSET_FIELDS = {"name", "estimated_risk", "reason", "approx_price"}

PRICE_ALIASES = {
    "BTC": "BTC-USD",
    "BITCOIN": "BTC-USD",
    "NIFTY": "^NSEI",
    "NIFTY50": "^NSEI",
    "GOLD": "GC=F",
    "XAU": "GC=F",
    "CASH": None,
}


# ──────────────────────────────────────────────────────────────
#  1. Portfolio Input
# ──────────────────────────────────────────────────────────────

def get_portfolio_input() -> dict[str, Any]:
    """
    Prompt the user for portfolio data interactively.

    Returns:
        A validated portfolio dictionary with an 'assets' key.

    Raises:
        PortfolioInputError: On any validation failure.
    """
    print("\n" + "=" * 62)
    print("        AI-POWERED PORTFOLIO EXPLAINER")
    print("=" * 62)
    print("\n  Enter your portfolio assets one by one.")
    print("  Type 'done' when finished.\n")

    assets: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    while True:
        raw = input("  Asset name (or 'done'): ").strip()
        if raw.lower() == "done":
            break
        if not raw:
            print("  ⚠  Asset name cannot be empty. Try again.")
            continue

        name = raw.upper()

        # Duplicate check
        if name in seen_names:
            print(f"  ⚠  '{name}' already added. Skipping duplicate.")
            continue

        # Allocation input
        try:
            alloc = float(input(f"  Allocation % for {name}: ").strip())
        except ValueError:
            print("  ⚠  Allocation must be a number. Skipping this asset.")
            continue

        if alloc < 0:
            print("  ⚠  Allocation cannot be negative. Skipping this asset.")
            continue
        if alloc > 100:
            print("  ⚠  Allocation cannot exceed 100%. Skipping this asset.")
            continue

        assets.append({"name": name, "allocation_pct": alloc})
        seen_names.add(name)
        print(f"  ✓  Added {name} at {alloc}%\n")

    # ── Post-collection validation ────────────────────────────
    if not assets:
        raise PortfolioInputError("Portfolio must contain at least one asset.")

    total = sum(a["allocation_pct"] for a in assets)
    if abs(total - 100.0) > 1.0:
        raise PortfolioInputError(
            f"Allocations must sum to ~100% (got {total:.1f}%). "
            "Please re-run and adjust."
        )

    return {"assets": assets}


def get_default_portfolio() -> dict[str, Any]:
    """
    Return the default sample portfolio for quick demonstration.

    Returns:
        Portfolio dictionary with four diversified assets.
    """
    return {
        "assets": [
            {"name": "BTC", "allocation_pct": 30},
            {"name": "NIFTY50", "allocation_pct": 40},
            {"name": "GOLD", "allocation_pct": 20},
            {"name": "CASH", "allocation_pct": 10},
        ]
    }


def validate_portfolio(portfolio: dict[str, Any]) -> None:
    """
    Validate portfolio structure and data integrity.

    Args:
        portfolio: Portfolio dictionary to validate.

    Raises:
        PortfolioInputError: If any validation rule is violated.
    """
    if not isinstance(portfolio, dict):
        raise PortfolioInputError("Portfolio must be a dictionary.")

    if "assets" not in portfolio:
        raise PortfolioInputError("Portfolio must have an 'assets' key.")

    assets = portfolio["assets"]
    if not isinstance(assets, list) or len(assets) == 0:
        raise PortfolioInputError("Portfolio must contain at least one asset.")

    seen: set[str] = set()
    for i, asset in enumerate(assets):
        if not isinstance(asset, dict):
            raise PortfolioInputError(f"Asset at index {i} must be a dict.")
        if "name" not in asset or "allocation_pct" not in asset:
            raise PortfolioInputError(
                f"Asset at index {i} missing 'name' or 'allocation_pct'."
            )
        if not isinstance(asset["name"], str) or not asset["name"].strip():
            raise PortfolioInputError(
                f"Asset at index {i}: name must be a non-empty string."
            )
        if not isinstance(asset["allocation_pct"], (int, float)):
            raise PortfolioInputError(
                f"Asset '{asset['name']}': allocation_pct must be numeric."
            )
        if asset["allocation_pct"] < 0:
            raise PortfolioInputError(
                f"Asset '{asset['name']}': allocation cannot be negative."
            )
        name_upper = asset["name"].upper()
        if name_upper in seen:
            raise PortfolioInputError(
                f"Duplicate asset detected: '{asset['name']}'."
            )
        seen.add(name_upper)

    total = sum(a["allocation_pct"] for a in assets)
    if abs(total - 100.0) > 1.0:
        raise PortfolioInputError(
            f"Allocations must sum to ~100% (got {total:.1f}%)."
        )


def resolve_price_symbol(asset_name: str) -> str | None:
    """Map a portfolio asset name to a Yahoo Finance ticker symbol."""
    return PRICE_ALIASES.get(asset_name.strip().upper(), asset_name.strip().upper())


def fetch_current_prices(portfolio: dict[str, Any]) -> dict[str, str]:
    """Fetch current prices for each asset using yfinance."""
    prices: dict[str, str] = {}

    for asset in portfolio["assets"]:
        name = asset["name"].strip().upper()
        symbol = resolve_price_symbol(name)

        if symbol is None:
            prices[name] = "₹0.00 (cash equivalent)"
            continue

        try:
            history = yf.Ticker(symbol).history(period="1d", interval="1d", auto_adjust=True)
            if history.empty or "Close" not in history:
                prices[name] = "N/A"
                continue

            current_price = float(history["Close"].dropna().iloc[-1])
            currency = "USD" if symbol.endswith("-USD") or symbol == "GC=F" else "INR"
            if symbol == "GC=F":
                currency = "USD"
            prices[name] = f"{current_price:,.2f} {currency}"
        except Exception:
            prices[name] = "N/A"

    return prices


# ──────────────────────────────────────────────────────────────
#  2. Prompt Construction
# ──────────────────────────────────────────────────────────────

def build_prompt(
    portfolio: dict[str, Any],
    tone: str = "beginner",
    price_map: dict[str, str] | None = None,
) -> str:
    """
    Construct a carefully engineered prompt for the Gemini LLM.

    Args:
        portfolio: Validated portfolio dictionary.
        tone:      One of 'beginner', 'experienced', or 'expert'.

    Returns:
        Complete prompt string ready for the API call.
    """
    tone = tone.lower()
    if tone not in VALID_TONES:
        tone = "beginner"

    # ── Tone-specific instructions ────────────────────────────
    tone_instructions = {
        "beginner": (
            "Use very simple, everyday language. Avoid all financial jargon. "
            "Explain concepts as if talking to someone who has never invested before. "
            "Use analogies where helpful."
        ),
        "experienced": (
            "Use clear, professional language suitable for someone with "
            "a few years of investing experience. You may use common financial "
            "terms but briefly explain any advanced concepts."
        ),
        "expert": (
            "Use precise financial terminology. Assume the reader understands "
            "concepts like Sharpe ratio, beta, volatility, and correlation. "
            "Be concise and data-driven."
        ),
    }

    # ── Asset list formatting ─────────────────────────────────
    asset_lines = []
    for asset in portfolio["assets"]:
        asset_name = asset["name"]
        asset_price = (price_map or {}).get(asset_name.upper(), "N/A")
        asset_lines.append(
            f"  - {asset_name}: {asset['allocation_pct']}% allocation | current price: {asset_price}"
        )
    asset_lines_text = "\n".join(asset_lines)

    prompt = f"""You are an honest, highly experienced financial advisor with 20+ years of experience managing wealth for high-net-worth individuals.

TONE INSTRUCTIONS:
{tone_instructions[tone]}

TASK:
Analyze the following investment portfolio allocation.
Use the provided current prices as factual data.
Do NOT estimate or invent prices yourself.
Focus only on portfolio risk, diversification, and plain-English explanation.

PORTFOLIO:
{asset_lines_text}

INSTRUCTIONS:
1. Analyze the allocation percentages and what they imply about the investor's strategy.
2. Infer the risk characteristics of EACH asset based on real-world behavior (volatility, historical drawdowns, correlation to markets).
3. Provide:
   a) A 3–4 sentence plain-English summary of the overall portfolio risk.
   b) ONE thing the investor is doing well (be specific).
   c) ONE improvement suggestion with clear reasoning.
   d) A one-line verdict: exactly one of "Aggressive", "Balanced", or "Conservative".
    e) Per-asset analysis with estimated risk level and explanation based on the provided current price.

CRITICAL OUTPUT FORMAT:
You MUST return your response as STRICT, VALID JSON and NOTHING ELSE.
Do NOT include any text, explanation, markdown formatting, or code fences outside the JSON.
Do NOT wrap the JSON in ```json``` or any other markers.

The JSON MUST follow this exact structure:
{{
  "summary": "<3-4 sentence plain-English summary of overall portfolio risk>",
  "good_practice": "<one specific thing the investor is doing well>",
  "improvement_suggestion": "<one improvement suggestion with reasoning>",
  "verdict": "<exactly one of: Aggressive / Balanced / Conservative>",
  "asset_analysis": [
    {{
      "name": "<asset ticker>",
      "estimated_risk": "<Low / Medium / High / Very High>",
      "reason": "<1-2 sentence explanation of risk assessment>",
      "approx_price": "<approximate current price in relevant currency>"
    }}
  ]
}}

Return ONLY the JSON object. No preamble, no postscript, no additional commentary."""

    return prompt


def build_critique_prompt(
    portfolio: dict[str, Any],
    first_response: dict[str, Any],
) -> str:
    """
    Build a second prompt that asks the LLM to critique its own analysis.

    Args:
        portfolio:      The original portfolio dictionary.
        first_response: Parsed JSON from the first LLM call.

    Returns:
        Critique prompt string.
    """
    asset_lines = "\n".join(
        f"  - {a['name']}: {a['allocation_pct']}% allocation"
        for a in portfolio["assets"]
    )

    first_json = json.dumps(first_response, indent=2)

    return f"""You are a senior portfolio risk auditor reviewing a financial advisor's analysis.

PORTFOLIO UNDER REVIEW:
{asset_lines}

ADVISOR'S ANALYSIS:
{first_json}

TASK:
Review the above analysis for accuracy and completeness. Check:
1. Are the risk assessments reasonable given real-world data?
2. Are the approximate prices in the right ballpark?
3. Is the verdict (Aggressive/Balanced/Conservative) justified?
4. Is the improvement suggestion practical and sound?
5. Any factual errors or misleading statements?

CRITICAL OUTPUT FORMAT:
Return ONLY valid JSON with this exact structure:
{{
  "accuracy_score": "<1-10 rating>",
  "is_verdict_correct": true/false,
  "corrections": ["<list of specific corrections, if any>"],
  "additional_insight": "<one additional insight the advisor missed>"
}}

Return ONLY the JSON. No other text."""


# ──────────────────────────────────────────────────────────────
#  3. API Integration
# ──────────────────────────────────────────────────────────────

def _http_post_json(url: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    """Perform an HTTP POST request and return parsed JSON."""
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    if not body.strip():
        raise ValueError("Empty response body")
    return json.loads(body)


def configure_api(api_key: str | None = None) -> str:
    """
    Resolve the Gemini API key from the explicit value, environment, or fallback.

    Args:
        api_key: Optional explicit API key. Falls back to Gemini env vars.

    Returns:
        The API key to use for Gemini.

    Raises:
        APIError: If no API key is available.
    """
    key = (
        api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not key:
        raise APIError(
            "No API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY."
        )
    return key


def call_gemini_api(
    prompt: str,
    model_name: str = DEFAULT_MODEL,
    max_retries: int = 3,
    timeout_seconds: int = 60,
) -> str:
    """
    Send a prompt to the Gemini API and return the raw text response.

    Implements retry logic with exponential backoff for transient
    failures (network errors, rate limits, timeouts).

    Args:
        prompt:          The complete prompt string.
        model_name:      Gemini model identifier.
        max_retries:     Maximum number of retry attempts.
        timeout_seconds: Request timeout in seconds.

    Returns:
        Raw response text from the LLM.

    Raises:
        APIError: After all retries are exhausted or on fatal errors.
    """
    api_key = configure_api()
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
        },
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = _http_post_json(url, payload, timeout=timeout_seconds)
            candidates = response.get("candidates") or []
            if not candidates:
                raise APIError("Gemini returned no candidates.")

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            text = "".join(
                part.get("text", "") for part in parts if isinstance(part, dict)
            )
            if not text or not text.strip():
                raise APIError("Gemini returned an empty response.")

            return text

        except Exception as exc:
            error_msg = str(exc).lower()

            # Identify retryable errors
            is_retryable = any(
                keyword in error_msg
                for keyword in ("rate limit", "429", "timeout", "503",
                                "deadline", "unavailable", "connection")
            )

            if is_retryable and attempt < max_retries:
                wait = 2 ** attempt  # Exponential backoff: 2s, 4s, 8s
                print(f"  ⏳ Retry {attempt}/{max_retries} in {wait}s "
                      f"({type(exc).__name__})...")
                time.sleep(wait)
                continue

            raise APIError(
                f"Gemini API call failed after {attempt} attempt(s): {exc}"
            ) from exc

    # Should never reach here, but just in case
    raise APIError("Gemini API call failed: max retries exhausted.")


# ──────────────────────────────────────────────────────────────
#  4. Response Parsing
# ──────────────────────────────────────────────────────────────

def extract_json_from_text(raw: str) -> str:
    """
    Extract JSON from raw LLM output, handling markdown code fences
    and other extraneous text.

    Args:
        raw: Raw text from the LLM.

    Returns:
        Cleaned JSON string.
    """
    text = raw.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(fence_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find JSON object boundaries
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace:last_brace + 1]

    return text


def parse_response(raw_response: str) -> dict[str, Any]:
    """
    Parse and validate the LLM's raw JSON response.

    Ensures all required fields are present and asset_analysis
    entries have the expected structure.

    Args:
        raw_response: Raw text from the Gemini API.

    Returns:
        Parsed and validated dictionary.

    Raises:
        ResponseParsingError: If JSON is invalid or fields are missing.
    """
    cleaned = extract_json_from_text(raw_response)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ResponseParsingError(
            f"Failed to parse JSON from LLM response: {e}\n"
            f"Cleaned text was:\n{cleaned[:500]}"
        ) from e

    if not isinstance(data, dict):
        raise ResponseParsingError(
            f"Expected a JSON object, got {type(data).__name__}."
        )

    # ── Check required top-level fields ──────────────────────
    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise ResponseParsingError(
            f"Missing required fields in response: {missing}"
        )

    # ── Validate verdict ─────────────────────────────────────
    valid_verdicts = {"aggressive", "balanced", "conservative"}
    if data["verdict"].lower().strip() not in valid_verdicts:
        # Attempt soft correction
        for v in valid_verdicts:
            if v in data["verdict"].lower():
                data["verdict"] = v.capitalize()
                break
        else:
            data["verdict"] = data["verdict"]  # Keep as-is with warning

    # ── Validate asset_analysis ──────────────────────────────
    if not isinstance(data["asset_analysis"], list):
        raise ResponseParsingError("'asset_analysis' must be a list.")

    for i, asset in enumerate(data["asset_analysis"]):
        if not isinstance(asset, dict):
            raise ResponseParsingError(
                f"Asset at index {i} in 'asset_analysis' must be a dict."
            )
        missing_asset_fields = ASSET_FIELDS - set(asset.keys())
        if missing_asset_fields:
            # Provide fallback for missing fields
            for field in missing_asset_fields:
                asset[field] = "N/A"

    return data


def parse_critique_response(raw_response: str) -> dict[str, Any] | None:
    """
    Parse the critique response. Returns None on failure (non-critical).

    Args:
        raw_response: Raw text from the critique API call.

    Returns:
        Parsed critique dictionary, or None if parsing fails.
    """
    try:
        cleaned = extract_json_from_text(raw_response)
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, Exception):
        return None


# ──────────────────────────────────────────────────────────────
#  5. Display Output
# ──────────────────────────────────────────────────────────────

def display_raw_response(raw: str) -> None:
    """
    Print the raw API response for transparency and debugging.

    Args:
        raw: Raw text from the Gemini API.
    """
    print("\n" + "─" * 62)
    print("  📡 RAW API RESPONSE")
    print("─" * 62)
    print(raw)
    print("─" * 62)


def display_output(parsed: dict[str, Any]) -> None:
    """
    Display the parsed LLM response in a clean, professional format.

    Args:
        parsed: Validated dictionary from parse_response().
    """
    print("\n" + "=" * 62)
    print("        📊 PORTFOLIO ANALYSIS — STRUCTURED OUTPUT")
    print("=" * 62)

    # ── Asset-wise analysis ──────────────────────────────────
    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │              ASSET-WISE ANALYSIS                    │")
    print("  └─────────────────────────────────────────────────────┘\n")

    for asset in parsed["asset_analysis"]:
        name = asset.get("name", "N/A")
        price = asset.get("approx_price", "N/A")
        risk = asset.get("estimated_risk", "N/A")
        reason = asset.get("reason", "N/A")

        print(f"  {'─' * 54}")
        print(f"  │ Asset          : {name}")
        print(f"  │ Approx Price   : {price}")
        print(f"  │ Estimated Risk : {risk}")
        print(f"  │ Reason         : {reason}")

    print(f"  {'─' * 54}")

    # ── Portfolio-level insights ─────────────────────────────
    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │              PORTFOLIO-LEVEL INSIGHTS               │")
    print("  └─────────────────────────────────────────────────────┘\n")

    # Summary — word-wrap for readability
    summary = parsed.get("summary", "N/A")
    print(f"  📝 Summary:")
    _print_wrapped(summary, indent=5, width=55)

    # Good practice
    good = parsed.get("good_practice", "N/A")
    print(f"\n  ✅ Good Practice:")
    _print_wrapped(good, indent=5, width=55)

    # Improvement suggestion
    improve = parsed.get("improvement_suggestion", "N/A")
    print(f"\n  💡 Improvement Suggestion:")
    _print_wrapped(improve, indent=5, width=55)

    # Verdict
    verdict = parsed.get("verdict", "N/A")
    verdict_emoji = {
        "aggressive": "🔴",
        "balanced": "🟡",
        "conservative": "🟢",
    }.get(verdict.lower().strip(), "⚪")

    print(f"\n  {verdict_emoji} Verdict: {verdict}")
    print("\n" + "=" * 62)


def display_critique(critique: dict[str, Any]) -> None:
    """
    Display the self-critique analysis results.

    Args:
        critique: Parsed critique dictionary.
    """
    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │              🔍 SELF-CRITIQUE ANALYSIS              │")
    print("  └─────────────────────────────────────────────────────┘\n")

    score = critique.get("accuracy_score", "N/A")
    print(f"  Accuracy Score     : {score}/10")

    verdict_ok = critique.get("is_verdict_correct", "N/A")
    print(f"  Verdict Correct    : {'✅ Yes' if verdict_ok else '❌ No'}")

    corrections = critique.get("corrections", [])
    if corrections:
        print(f"\n  ⚠  Corrections:")
        for c in corrections:
            _print_wrapped(f"• {c}", indent=5, width=55)
    else:
        print(f"\n  ✅ No corrections needed.")

    insight = critique.get("additional_insight", "N/A")
    print(f"\n  💎 Additional Insight:")
    _print_wrapped(insight, indent=5, width=55)

    print("\n" + "=" * 62)


def _print_wrapped(text: str, indent: int = 5, width: int = 55) -> None:
    """
    Print text with word wrapping at the specified width.

    Args:
        text:   Text to wrap.
        indent: Number of leading spaces per line.
        width:  Maximum character width per line.
    """
    prefix = " " * indent
    words = text.split()
    line = prefix

    for word in words:
        if len(line) + len(word) + 1 > width + indent:
            print(line)
            line = prefix + word
        else:
            line = line + " " + word if line.strip() else prefix + word

    if line.strip():
        print(line)


# ──────────────────────────────────────────────────────────────
#  Main Orchestration
# ──────────────────────────────────────────────────────────────

def main() -> None:
    """
    Entry point: orchestrates input → prompt → API call → parse → display.
    """
    # Ensure Unicode output on Windows terminals
    sys.stdout.reconfigure(encoding="utf-8")

    try:
        configure_api()
    except APIError as e:
        print(f"\n  ❌ {e}")
        sys.exit(1)

    # ── Portfolio Input ──────────────────────────────────────
    print("\n" + "=" * 62)
    print("        AI-POWERED PORTFOLIO EXPLAINER")
    print("=" * 62)
    print("\n  Choose input mode:")
    print("    [1] Use default sample portfolio")
    print("    [2] Enter portfolio manually")

    choice = input("\n  Your choice (1/2): ").strip()

    if choice == "2":
        try:
            portfolio = get_portfolio_input()
        except PortfolioInputError as e:
            print(f"\n  ❌ Input Error: {e}")
            sys.exit(1)
    else:
        portfolio = get_default_portfolio()
        print("\n  Using default portfolio:")
        for a in portfolio["assets"]:
            print(f"    • {a['name']:8s} → {a['allocation_pct']}%")

    # Validate
    try:
        validate_portfolio(portfolio)
    except PortfolioInputError as e:
        print(f"\n  ❌ Validation Error: {e}")
        sys.exit(1)

    # ── Fetch current prices ────────────────────────────────
    print("\n  ⏳ Fetching current prices from yfinance...")
    price_map = fetch_current_prices(portfolio)

    # ── Tone Selection ───────────────────────────────────────
    print(f"\n  Select explanation tone:")
    print(f"    [1] Beginner   — simple, jargon-free")
    print(f"    [2] Experienced — professional, moderate detail")
    print(f"    [3] Expert     — technical, data-driven")

    tone_choice = input("\n  Your choice (1/2/3): ").strip()
    tone_map = {"1": "beginner", "2": "experienced", "3": "expert"}
    tone = tone_map.get(tone_choice, "beginner")
    print(f"  → Tone: {tone.capitalize()}")

    # ── Critique option ──────────────────────────────────────
    enable_critique = (
        input("\n  Enable self-critique? (y/n): ").strip().lower() == "y"
    )

    # ── Build Prompt ─────────────────────────────────────────
    print("\n  ⏳ Building prompt...")
    prompt = build_prompt(portfolio, tone=tone, price_map=price_map)

    # ── Call Gemini API ──────────────────────────────────────
    print("  ⏳ Calling Gemini API...")
    try:
        raw_response = call_gemini_api(prompt)
    except APIError as e:
        print(f"\n  ❌ API Error: {e}")
        sys.exit(1)

    # ── Display raw response ─────────────────────────────────
    display_raw_response(raw_response)

    # ── Parse response ───────────────────────────────────────
    print("\n  ⏳ Parsing structured output...")
    try:
        parsed = parse_response(raw_response)
    except ResponseParsingError as e:
        print(f"\n  ❌ Parsing Error: {e}")
        sys.exit(1)

    # Replace model-estimated prices with fetched prices so display stays factual.
    for asset in parsed.get("asset_analysis", []):
        asset_name = str(asset.get("name", "")).strip().upper()
        if asset_name in price_map:
            asset["approx_price"] = price_map[asset_name]

    # ── Display structured output ────────────────────────────
    display_output(parsed)

    # ── Optional: Self-Critique (Bonus) ──────────────────────
    if enable_critique:
        print("\n  ⏳ Running self-critique analysis...")
        critique_prompt = build_critique_prompt(portfolio, parsed)

        try:
            critique_raw = call_gemini_api(critique_prompt)
            display_raw_response(critique_raw)
            critique = parse_critique_response(critique_raw)
            if critique:
                display_critique(critique)
            else:
                print("  ⚠  Could not parse critique response. Skipping.")
        except APIError as e:
            print(f"  ⚠  Critique call failed: {e}. Skipping.")

    print("\n  ✅ Analysis complete.\n")


if __name__ == "__main__":
    main()
