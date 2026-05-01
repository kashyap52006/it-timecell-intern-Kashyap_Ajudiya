"""
Task 02: Live Market Data Fetch
================================
A robust data-fetching pipeline that retrieves live prices for
stocks, indices, and cryptocurrencies from free public APIs
and displays them in a clean, formatted terminal table.

APIs Used:
  - CoinGecko (crypto)  : https://api.coingecko.com  — free, no key
  - Yahoo Finance (stocks): query2.finance.yahoo.com   — free, no key

Python : 3.10+
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ──────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────

IST = timezone(timedelta(hours=5, minutes=30))

REQUEST_TIMEOUT_SECS = 10

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)


# ──────────────────────────────────────────────────────────────
#  Logging Setup
# ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("market_fetch")


# ──────────────────────────────────────────────────────────────
#  Data Model
# ──────────────────────────────────────────────────────────────

@dataclass
class AssetPrice:
    """
    Validated result of a single asset price fetch.

    Attributes:
        name       : Human-readable asset name (e.g. "BTC", "NIFTY 50").
        price      : Current market price, or None on failure.
        currency   : Currency code the price is quoted in (e.g. "USD", "INR").
        fetched_at : Timestamp string of when the fetch occurred.
        error      : Error message if the fetch failed, None otherwise.
    """

    name: str
    price: Optional[float]
    currency: str
    fetched_at: str
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """True when a numeric price was successfully retrieved."""
        return self.price is not None and self.error is None


# ──────────────────────────────────────────────────────────────
#  HTTP Helper
# ──────────────────────────────────────────────────────────────

def _http_get_json(url: str) -> Dict[str, Any]:
    """
    Perform an HTTP GET request and return parsed JSON.

    Raises:
        HTTPError : On non-2xx status codes (includes 429 rate-limit).
        URLError  : On network / DNS failures.
        ValueError: On non-JSON or empty response bodies.
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    req = Request(url, headers=headers)

    with urlopen(req, timeout=REQUEST_TIMEOUT_SECS) as resp:
        body = resp.read().decode("utf-8")

    if not body.strip():
        raise ValueError("Empty response body")

    return json.loads(body)


# ──────────────────────────────────────────────────────────────
#  Fetchers  (Data Fetching + Parsing/Validation)
# ──────────────────────────────────────────────────────────────

def _now_ist() -> str:
    """Current time formatted as IST string."""
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def fetch_coingecko(asset_id: str, display_name: str,
                    vs_currency: str = "usd") -> AssetPrice:
    """
    Fetch a cryptocurrency price from the CoinGecko public API.

    Args:
        asset_id     : CoinGecko identifier (e.g. "bitcoin", "ethereum").
        display_name : Name shown in the output table (e.g. "BTC").
        vs_currency  : Quote currency (default "usd").

    Returns:
        AssetPrice with price populated on success, or error on failure.
    """
    timestamp = _now_ist()
    currency_upper = vs_currency.upper()

    try:
        url = (
            f"https://api.coingecko.com/api/v3/simple/price"
            f"?ids={asset_id}&vs_currencies={vs_currency}"
        )
        logger.info("Fetching %s from CoinGecko ...", display_name)
        data = _http_get_json(url)

        # ── Validate response structure ──────────────────────
        if asset_id not in data:
            raise ValueError(
                f"Asset '{asset_id}' not found in CoinGecko response"
            )
        asset_data = data[asset_id]

        if vs_currency not in asset_data:
            raise ValueError(
                f"Currency '{vs_currency}' not available for '{asset_id}'"
            )

        raw_price = asset_data[vs_currency]

        # ── Type validation ──────────────────────────────────
        if not isinstance(raw_price, (int, float)):
            raise TypeError(
                f"Expected numeric price, got {type(raw_price).__name__}: {raw_price}"
            )

        logger.info("  ✓ %s = %s %s", display_name, f"{raw_price:,.2f}", currency_upper)
        return AssetPrice(
            name=display_name,
            price=float(raw_price),
            currency=currency_upper,
            fetched_at=timestamp,
        )

    except HTTPError as exc:
        msg = f"HTTP {exc.code}"
        if exc.code == 429:
            msg = "Rate limit exceeded (HTTP 429) — try again later"
        logger.error("  ✗ %s failed: %s", display_name, msg)
        return AssetPrice(display_name, None, currency_upper, timestamp, msg)

    except URLError as exc:
        msg = f"Network error: {exc.reason}"
        logger.error("  ✗ %s failed: %s", display_name, msg)
        return AssetPrice(display_name, None, currency_upper, timestamp, msg)

    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        msg = f"Data error: {exc}"
        logger.error("  ✗ %s failed: %s", display_name, msg)
        return AssetPrice(display_name, None, currency_upper, timestamp, msg)


def fetch_yahoo(symbol: str, display_name: str) -> AssetPrice:
    """
    Fetch a stock, index, or commodity price from Yahoo Finance's
    public chart API (v8).

    Args:
        symbol       : Yahoo Finance ticker (e.g. "^NSEI", "GC=F").
        display_name : Name shown in the output table (e.g. "NIFTY 50").

    Returns:
        AssetPrice with price populated on success, or error on failure.
    """
    timestamp = _now_ist()

    try:
        url = (
            f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
            f"?interval=1d&range=1d"
        )
        logger.info("Fetching %s from Yahoo Finance ...", display_name)
        data = _http_get_json(url)

        # ── Navigate response structure ──────────────────────
        chart = data.get("chart")
        if not chart:
            raise ValueError("Missing 'chart' key in Yahoo response")

        error_block = chart.get("error")
        if error_block:
            raise ValueError(
                f"Yahoo API error: {error_block.get('description', 'unknown')}"
            )

        results = chart.get("result")
        if not results or not isinstance(results, list):
            raise ValueError("Empty or invalid 'result' array in Yahoo response")

        meta = results[0].get("meta", {})

        # ── Extract and validate price ───────────────────────
        raw_price = meta.get("regularMarketPrice")
        if raw_price is None:
            raise ValueError("'regularMarketPrice' missing from response meta")

        if not isinstance(raw_price, (int, float)):
            raise TypeError(
                f"Expected numeric price, got {type(raw_price).__name__}: {raw_price}"
            )

        currency = meta.get("currency", "USD")

        logger.info("  ✓ %s = %s %s", display_name, f"{raw_price:,.2f}", currency)
        return AssetPrice(
            name=display_name,
            price=float(raw_price),
            currency=currency,
            fetched_at=timestamp,
        )

    except HTTPError as exc:
        msg = f"HTTP {exc.code}"
        if exc.code == 403:
            msg = "Access denied (HTTP 403) — Yahoo may be blocking requests"
        elif exc.code == 429:
            msg = "Rate limit exceeded (HTTP 429) — try again later"
        logger.error("  ✗ %s failed: %s", display_name, msg)
        return AssetPrice(display_name, None, "—", timestamp, msg)

    except URLError as exc:
        msg = f"Network error: {exc.reason}"
        logger.error("  ✗ %s failed: %s", display_name, msg)
        return AssetPrice(display_name, None, "—", timestamp, msg)

    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        msg = f"Data error: {exc}"
        logger.error("  ✗ %s failed: %s", display_name, msg)
        return AssetPrice(display_name, None, "—", timestamp, msg)


# ──────────────────────────────────────────────────────────────
#  Asset Registry
# ──────────────────────────────────────────────────────────────
#  Each entry defines: (display_name, source, api_id, extra_args)
#  This makes it trivial to add or remove assets.

ASSET_REGISTRY: List[Dict[str, str]] = [
    {
        "name": "BTC",
        "source": "coingecko",
        "api_id": "bitcoin",
        "vs_currency": "usd",
    },
    {
        "name": "ETH",
        "source": "coingecko",
        "api_id": "ethereum",
        "vs_currency": "usd",
    },
    {
        "name": "NIFTY 50",
        "source": "yahoo",
        "api_id": "^NSEI",
    },
    {
        "name": "GOLD",
        "source": "yahoo",
        "api_id": "GC=F",
    },
]


# ──────────────────────────────────────────────────────────────
#  Orchestration
# ──────────────────────────────────────────────────────────────

def fetch_all_assets(registry: List[Dict[str, str]]) -> List[AssetPrice]:
    """
    Iterate through the asset registry and fetch each price.

    If a single asset fails, its error is captured and the loop
    continues — the script never crashes due to one bad fetch.
    """
    results: List[AssetPrice] = []

    for entry in registry:
        name = entry["name"]
        source = entry["source"]
        api_id = entry["api_id"]

        try:
            if source == "coingecko":
                result = fetch_coingecko(
                    asset_id=api_id,
                    display_name=name,
                    vs_currency=entry.get("vs_currency", "usd"),
                )
            elif source == "yahoo":
                result = fetch_yahoo(symbol=api_id, display_name=name)
            else:
                result = AssetPrice(
                    name=name,
                    price=None,
                    currency="—",
                    fetched_at=_now_ist(),
                    error=f"Unknown source '{source}'",
                )
        except Exception as exc:
            # Absolute safety net — nothing can crash the pipeline
            logger.exception("Unexpected error fetching %s", name)
            result = AssetPrice(
                name=name,
                price=None,
                currency="—",
                fetched_at=_now_ist(),
                error=f"Unexpected: {exc}",
            )

        results.append(result)

    return results


# ──────────────────────────────────────────────────────────────
#  Display (Presentation Layer)
# ──────────────────────────────────────────────────────────────

def display_results(results: List[AssetPrice]) -> None:
    """
    Print a well-aligned terminal table of fetched asset prices.

    Failed assets are shown with a descriptive error instead of a price,
    ensuring the table remains consistent regardless of partial failures.
    """
    if not results:
        print("\n  No assets were fetched.\n")
        return

    # ── Column widths (adaptive to data) ─────────────────────
    col_asset = max(len(r.name) for r in results)
    col_asset = max(col_asset, 7)  # minimum header width

    col_price = 14
    col_curr = max((len(r.currency) for r in results), default=8)
    col_curr = max(col_curr, 8)

    col_time = 25

    # ── Build the table ──────────────────────────────────────
    sep = (
        f"  +{'-' * (col_asset + 2)}"
        f"+{'-' * (col_price + 2)}"
        f"+{'-' * (col_curr + 2)}"
        f"+{'-' * (col_time + 2)}+"
    )

    header = (
        f"  | {'Asset':<{col_asset}} "
        f"| {'Price':>{col_price}} "
        f"| {'Currency':<{col_curr}} "
        f"| {'Fetched At':<{col_time}} |"
    )

    # Title
    fetch_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    print(f"\n  Asset Prices — fetched at {fetch_time}")
    print(sep)
    print(header)
    print(sep)

    # Data rows
    for r in results:
        if r.is_valid:
            price_str = f"{r.price:>14,.2f}"
            row = (
                f"  | {r.name:<{col_asset}} "
                f"| {price_str} "
                f"| {r.currency:<{col_curr}} "
                f"| {r.fetched_at:<{col_time}} |"
            )
        else:
            error_display = f"ERROR: {r.error}"
            row = (
                f"  | {r.name:<{col_asset}} "
                f"| {'—':>{col_price}} "
                f"| {'—':<{col_curr}} "
                f"| {r.fetched_at:<{col_time}} |"
            )
            # Print the row, then the error reason on the next line
            print(row)
            print(f"  |   └─ {error_display}")
            continue

        print(row)

    print(sep)

    # Summary
    ok = sum(1 for r in results if r.is_valid)
    fail = len(results) - ok
    print(f"\n  Fetched: {ok}/{len(results)} assets successfully", end="")
    if fail:
        print(f"  ({fail} failed)")
    else:
        print()
    print()


# ──────────────────────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────────────────────

def main() -> None:
    """Fetch live prices for all registered assets and display results."""

    # Ensure Unicode output works on Windows terminals
    sys.stdout.reconfigure(encoding="utf-8")

    print()
    print("=" * 60)
    print("       LIVE MARKET DATA FETCH — Task 02")
    print("=" * 60)

    results = fetch_all_assets(ASSET_REGISTRY)
    display_results(results)


if __name__ == "__main__":
    main()
