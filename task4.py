"""
Task 04: Portfolio Correlation Risk Analyzer
=============================================
A production-style portfolio risk tool that focuses on how assets move together
over time, detects weakening diversification, forecasts correlation persistence,
and uses Gemini to produce simple human-readable risk insights.

Python : 3.10+
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from statsmodels.tsa.ar_model import AutoReg
except ImportError as exc:
    print(
        "❌  Missing dependency: "
        f"{exc.name or 'required package'}\n"
        "   Install the project requirements before running task4.py."
    )
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────

DEFAULT_TICKERS = ["NIFTY", "BTC", "GOLD"]
DEFAULT_PERIOD = "2y"
ROLLING_WINDOW = 30
SHORT_WINDOW = 30
LONG_WINDOW = 60
AR_LAGS = 1
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
VIX_TICKER = "^VIX"

TICKER_ALIASES = {
    "NIFTY": "^NSEI",
    "NIFTY50": "^NSEI",
    "BTC": "BTC-USD",
    "BITCOIN": "BTC-USD",
    "GOLD": "GLD",
    "XAU": "GC=F",
    "SILVER": "SLV",
    "SPY": "SPY",
    "VIX": VIX_TICKER,
}


# ──────────────────────────────────────────────────────────────
#  Exceptions
# ──────────────────────────────────────────────────────────────


class DataFetchError(Exception):
    """Raised when price data or market stress data cannot be fetched."""


class AnalysisError(Exception):
    """Raised when correlation analysis cannot be completed."""


class LLMError(Exception):
    """Raised when the Gemini API call or parsing fails."""


class APIError(Exception):
    """Raised when the Gemini API call fails."""


class ResponseParsingError(Exception):
    """Raised when the Gemini response cannot be parsed."""


# ──────────────────────────────────────────────────────────────
#  Data Model
# ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class LLMInsight:
    """Structured response from Gemini."""

    summary: str
    risk_level: str
    key_insight: str
    suggestions: list[str]


@dataclass(slots=True)
class AnalysisBundle:
    """Container for the computed portfolio risk metrics."""

    tickers: list[str]
    prices: pd.DataFrame
    returns: pd.DataFrame
    current_correlation: float
    drift_value: float
    short_term_average: float
    long_term_average: float
    drift_slope: float
    forecast_value: float
    diversification_score_value: float
    vix_value: float | None
    vix_level: str
    vix_status: str
    rolling_series: pd.Series


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────


def resolve_ticker(symbol: str) -> str:
    """Map common human-friendly aliases to Yahoo Finance symbols."""
    normalized = symbol.strip().upper()
    return TICKER_ALIASES.get(normalized, symbol.strip())


def _normalize_price_index(series: pd.Series) -> pd.Series:
    """Return a timezone-naive DatetimeIndex for price series alignment."""
    index = pd.to_datetime(series.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert(None)
    series.index = index
    return series


def _http_post_json(url: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    """Perform a JSON POST request and return parsed JSON."""
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
    """Resolve the Gemini API key from arguments or the environment."""
    key = (
        api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not key:
        raise APIError(
            "No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY."
        )
    return key


def call_gemini_api(
    prompt: str,
    api_key: str | None = None,
    model_name: str = DEFAULT_GEMINI_MODEL,
    max_retries: int = 3,
) -> str:
    """Send a prompt to Gemini and return the raw text response."""
    key = configure_api(api_key)
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={key}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 512,
            "responseMimeType": "application/json",
        },
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = _http_post_json(url, payload, timeout=60)
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
            is_retryable = any(
                keyword in error_msg
                for keyword in (
                    "rate limit",
                    "429",
                    "timeout",
                    "503",
                    "deadline",
                    "unavailable",
                    "connection",
                )
            )

            if is_retryable and attempt < max_retries:
                wait = 2 ** attempt
                print(
                    f"  ⏳ Retry {attempt}/{max_retries} in {wait}s ({type(exc).__name__})..."
                )
                time.sleep(wait)
                continue

            raise APIError(
                f"Gemini API call failed after {attempt} attempt(s): {exc}"
            ) from exc

    raise APIError("Gemini API call failed: max retries exhausted.")


def parse_response(raw_response: str) -> LLMInsight:
    """Parse and validate the Gemini JSON response."""
    cleaned = extract_json_from_text(raw_response)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ResponseParsingError(
            f"Failed to parse JSON from Gemini response: {exc}\n"
            f"Cleaned text was:\n{cleaned[:500]}"
        ) from exc

    if not isinstance(data, dict):
        raise ResponseParsingError(
            f"Expected a JSON object, got {type(data).__name__}."
        )

    required_fields = {"summary", "risk_level", "key_insight", "suggestions"}
    missing = required_fields - set(data.keys())
    if missing:
        raise ResponseParsingError(
            f"Missing required fields in Gemini response: {missing}"
        )

    suggestions = data["suggestions"]
    if not isinstance(suggestions, list):
        raise ResponseParsingError("Field 'suggestions' must be a list.")

    risk_level = str(data["risk_level"]).upper().strip()
    if risk_level not in {"LOW", "MODERATE", "HIGH"}:
        raise ResponseParsingError(
            f"Invalid risk level '{data['risk_level']}'. Expected LOW, MODERATE, or HIGH."
        )

    return LLMInsight(
        summary=str(data["summary"]),
        risk_level=risk_level,
        key_insight=str(data["key_insight"]),
        suggestions=[str(item) for item in suggestions],
    )


def _mean_abs_offdiag_correlation(frame: pd.DataFrame) -> float:
    """Return the mean absolute off-diagonal correlation for a DataFrame."""
    if frame.shape[1] < 2:
        return 0.0
    corr = frame.corr().abs().to_numpy()
    upper = corr[np.triu_indices_from(corr, k=1)]
    if upper.size == 0:
        return 0.0
    return float(np.nanmean(upper))


def _series_trend_slope(series: pd.Series) -> float:
    """Fit a simple linear slope over the supplied series values."""
    clean = series.dropna().astype(float)
    if len(clean) < 2:
        return 0.0
    x = np.arange(len(clean), dtype=float)
    slope, _intercept = np.polyfit(x, clean.to_numpy(), 1)
    return float(slope)


def extract_json_from_text(raw: str) -> str:
    """Extract JSON from raw LLM output, handling markdown code fences."""
    text = raw.strip()

    fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(fence_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace:last_brace + 1]

    return text


# ──────────────────────────────────────────────────────────────
#  1. Data Handling
# ──────────────────────────────────────────────────────────────


def fetch_data(tickers: list[str], period: str = DEFAULT_PERIOD) -> pd.DataFrame:
    """Fetch historical close prices for each ticker and align them on dates."""
    if not tickers:
        raise DataFetchError("At least one ticker is required.")

    price_frames: list[pd.Series] = []
    failed: list[str] = []

    for ticker in tickers:
        symbol = resolve_ticker(ticker)
        try:
            history = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
            if history.empty or "Close" not in history:
                failed.append(f"{ticker} ({symbol})")
                continue

            close = history["Close"].copy()
            close.name = ticker.strip().upper()
            price_frames.append(_normalize_price_index(close))
        except Exception:
            failed.append(f"{ticker} ({symbol})")

    if not price_frames:
        raise DataFetchError(
            "No valid price series could be fetched for the requested tickers."
        )

    prices = pd.concat(price_frames, axis=1, sort=False).sort_index()
    prices = prices.ffill().dropna(how="all")
    prices = prices.dropna(axis=1, how="all")

    if prices.empty or prices.shape[1] < 2:
        raise DataFetchError(
            "Need at least two valid tickers with overlapping price history."
        )

    if failed:
        print("⚠  Skipped invalid or empty tickers: " + ", ".join(failed))

    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert aligned price data into daily returns."""
    if prices.empty:
        raise AnalysisError("Cannot compute returns from an empty price table.")

    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="all")
    if returns.empty:
        raise AnalysisError("Return series is empty after transformation.")
    return returns


# ──────────────────────────────────────────────────────────────
#  2. Correlation Analysis
# ──────────────────────────────────────────────────────────────


def rolling_correlation(returns: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.Series:
    """Compute the rolling average absolute pairwise correlation."""
    if returns.empty:
        raise AnalysisError("Returns are empty; cannot compute rolling correlation.")

    values: list[float] = []
    indices: list[pd.Timestamp] = []

    for end_idx in range(window - 1, len(returns)):
        window_frame = returns.iloc[end_idx - window + 1 : end_idx + 1].dropna(axis=0, how="any")
        if len(window_frame) < max(10, window // 2):
            continue
        correlation_value = _mean_abs_offdiag_correlation(window_frame)
        values.append(correlation_value)
        indices.append(returns.index[end_idx])

    if not values:
        raise AnalysisError(
            "Not enough overlapping return observations to compute rolling correlation."
        )

    return pd.Series(values, index=indices, name="rolling_mean_abs_correlation")


def compute_drift(rolling_series: pd.Series) -> tuple[float, float, float, float]:
    """Compare short- and long-term correlation to measure drift and trend."""
    clean = rolling_series.dropna().astype(float)
    if len(clean) < 5:
        raise AnalysisError("Need more rolling correlation values to compute drift.")

    short_term = float(clean.tail(min(SHORT_WINDOW, len(clean))).mean())
    long_term = float(clean.tail(min(LONG_WINDOW, len(clean))).mean())
    drift_value = short_term - long_term
    slope = _series_trend_slope(clean.tail(min(LONG_WINDOW, len(clean))))
    return short_term, long_term, drift_value, slope


def diversification_score(returns: pd.DataFrame) -> float:
    """Convert the average absolute off-diagonal correlation into a 0-100 score."""
    clean = returns.dropna(axis=0, how="any")
    if clean.shape[1] < 2:
        return 100.0

    mean_abs_corr = _mean_abs_offdiag_correlation(clean)
    score = (1.0 - mean_abs_corr) * 100.0
    return float(max(0.0, min(100.0, score)))


def forecast_ar1(rolling_series: pd.Series) -> float:
    """Forecast the next correlation value using a simple AR(1) model."""
    clean = rolling_series.dropna().astype(float)
    if len(clean) < 10:
        return float(clean.iloc[-1])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = AutoReg(clean, lags=AR_LAGS, old_names=False).fit()
            prediction = model.predict(start=len(clean), end=len(clean))
            forecast = float(prediction.iloc[0])
    except Exception:
        forecast = float(clean.iloc[-1])

    return float(max(0.0, min(1.0, forecast)))


def vix_signal(period: str = "1mo") -> tuple[float | None, str, str]:
    """Fetch VIX and classify it into low, moderate, or high risk."""
    try:
        history = yf.Ticker(VIX_TICKER).history(period=period, interval="1d", auto_adjust=True)
        if history.empty or "Close" not in history:
            raise DataFetchError("VIX history is empty.")

        close = history["Close"].dropna()
        if close.empty:
            raise DataFetchError("VIX close series is empty.")

        value = float(close.iloc[-1])
        if value < 15:
            level = "LOW"
            status = "Low market stress"
        elif value < 25:
            level = "MODERATE"
            status = "Moderate market stress"
        else:
            level = "HIGH"
            status = "High market stress"

        return value, level, status
    except Exception:
        return None, "UNKNOWN", "VIX data unavailable"


# ──────────────────────────────────────────────────────────────
#  3. LLM Integration
# ──────────────────────────────────────────────────────────────


def build_llm_prompt(
    tickers: list[str],
    current_correlation: float,
    drift_value: float,
    short_term_average: float,
    long_term_average: float,
    forecast_value: float,
    diversification_score_value: float,
    vix_value: float | None,
    vix_level: str,
) -> str:
    """Build the prompt for Gemini with the exact JSON output contract."""
    ticker_list = ", ".join(tickers)
    vix_text = "unavailable" if vix_value is None else f"{vix_value:.2f}"

    return f"""You are an expert financial advisor with 20+ years of experience.

You are analyzing a portfolio correlation risk report for these assets: {ticker_list}.

Use the following computed metrics:
- Current rolling correlation: {current_correlation:.4f}
- Short-term correlation average (30d): {short_term_average:.4f}
- Long-term correlation average (60d): {long_term_average:.4f}
- Correlation drift (short - long): {drift_value:.4f}
- AR(1) next-step forecast: {forecast_value:.4f}
- Diversification score (0-100): {diversification_score_value:.2f}
- VIX level: {vix_text} ({vix_level})

Task:
Explain the portfolio risk in simple, non-technical language.
State whether diversification is weakening.
Give a concise key insight.
Suggest practical actions such as rebalancing, hedging, or increasing cash during high-risk periods.

Return ONLY valid JSON with this exact structure:
{{
  "summary": "...",
  "risk_level": "LOW | MODERATE | HIGH",
  "key_insight": "...",
  "suggestions": ["...", "..."]
}}

Do not include markdown, code fences, or any extra text."""


def call_llm(prompt: str, api_key: str | None = None, model: str = DEFAULT_GEMINI_MODEL) -> LLMInsight:
    """Call Gemini and parse the structured JSON response."""
    try:
        raw_response = call_gemini_api(prompt=prompt, api_key=api_key, model_name=model)
        return parse_response(raw_response)
    except APIError as exc:
        raise LLMError("Gemini request failed. Try again or check the model/key.") from exc
    except ResponseParsingError as exc:
        raise LLMError("Gemini returned unreadable output. Please retry.") from exc


# ──────────────────────────────────────────────────────────────
#  4. Display
# ──────────────────────────────────────────────────────────────


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def display_results(bundle: AnalysisBundle, insight: LLMInsight | None) -> None:
    """Print the computed metrics and LLM summary in a clean terminal layout."""
    print("\n" + "=" * 72)
    print("PORTFOLIO CORRELATION RISK ANALYZER")
    print("=" * 72)

    print("\nRaw Data")
    print("-" * 72)
    print(f"Tickers: {', '.join(bundle.tickers)}")
    print(f"Current correlation: {_format_percent(bundle.current_correlation)}")
    print(f"Drift: {_format_percent(bundle.drift_value)}")
    print(f"Short-term average: {_format_percent(bundle.short_term_average)}")
    print(f"Long-term average: {_format_percent(bundle.long_term_average)}")
    print(f"AR(1) forecast: {_format_percent(bundle.forecast_value)}")
    print(f"Diversification score: {bundle.diversification_score_value:.2f} / 100")
    if bundle.vix_value is None:
        print(f"VIX: {bundle.vix_status}")
    else:
        print(f"VIX: {bundle.vix_value:.2f} ({bundle.vix_level} - {bundle.vix_status})")

    print("\nLLM Insights")
    print("-" * 72)
    if insight is None:
        print("Gemini insights unavailable.")
    else:
        print(f"Summary: {insight.summary}")
        print(f"Risk level: {insight.risk_level}")
        print(f"Key insight: {insight.key_insight}")
        if insight.suggestions:
            print("Suggestions:")
            for suggestion in insight.suggestions:
                print(f"- {suggestion}")
        else:
            print("Suggestions: None returned.")


# ──────────────────────────────────────────────────────────────
#  5. Orchestration
# ──────────────────────────────────────────────────────────────


def run_analysis(tickers: list[str], api_key: str | None = None, model: str = DEFAULT_GEMINI_MODEL) -> tuple[AnalysisBundle, LLMInsight | None]:
    """Run the full correlation-risk pipeline and optionally call Gemini."""
    resolved = [ticker.strip() for ticker in tickers if ticker.strip()]
    if not resolved:
        resolved = DEFAULT_TICKERS.copy()

    prices = fetch_data(resolved, period=DEFAULT_PERIOD)
    returns = compute_returns(prices)
    rolling_series = rolling_correlation(returns, window=ROLLING_WINDOW)
    current_correlation = float(rolling_series.dropna().iloc[-1])
    short_term_average, long_term_average, drift_value, drift_slope = compute_drift(rolling_series)
    forecast_value = forecast_ar1(rolling_series)
    diversification_score_value = diversification_score(returns)
    vix_value, vix_level, vix_status = vix_signal()

    bundle = AnalysisBundle(
        tickers=[ticker.strip().upper() for ticker in resolved],
        prices=prices,
        returns=returns,
        current_correlation=current_correlation,
        drift_value=drift_value,
        short_term_average=short_term_average,
        long_term_average=long_term_average,
        drift_slope=drift_slope,
        forecast_value=forecast_value,
        diversification_score_value=diversification_score_value,
        vix_value=vix_value,
        vix_level=vix_level,
        vix_status=vix_status,
        rolling_series=rolling_series,
    )

    prompt = build_llm_prompt(
        tickers=bundle.tickers,
        current_correlation=bundle.current_correlation,
        drift_value=bundle.drift_value,
        short_term_average=bundle.short_term_average,
        long_term_average=bundle.long_term_average,
        forecast_value=bundle.forecast_value,
        diversification_score_value=bundle.diversification_score_value,
        vix_value=bundle.vix_value,
        vix_level=bundle.vix_level,
    )

    insight: LLMInsight | None = None
    try:
        insight = call_llm(prompt=prompt, api_key=api_key, model=model)
    except LLMError as exc:
        print(f"⚠  Gemini insight skipped: {exc}")

    return bundle, insight


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Portfolio Correlation Risk Analyzer",
    )
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Asset tickers or aliases (e.g. NIFTY BTC GOLD).",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="Gemini API key (otherwise reads GEMINI_API_KEY or GOOGLE_API_KEY).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_GEMINI_MODEL,
        help=f"Gemini model name (default: {DEFAULT_GEMINI_MODEL}).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    try:
        bundle, insight = run_analysis(args.tickers, api_key=args.api_key, model=args.model)
        display_results(bundle, insight)
    except (DataFetchError, AnalysisError, LLMError) as exc:
        print(f"❌  {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()