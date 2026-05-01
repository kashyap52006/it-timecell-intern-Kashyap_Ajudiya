"""
Task 01: Portfolio Risk Calculator
===================================
A production-quality portfolio risk assessment system that computes
post-crash values, financial runway, ruin probability, and concentration
risk using object-oriented design.
Python : 3.10+
"""

from __future__ import annotations
import sys


# ──────────────────────────────────────────────────────────────
#  Custom Exceptions
# ──────────────────────────────────────────────────────────────

class PortfolioValidationError(Exception):
    """Raised when portfolio-level validation fails."""


class AssetValidationError(Exception):
    """Raised when asset-level validation fails."""


# ──────────────────────────────────────────────────────────────
#  Asset Class
# ──────────────────────────────────────────────────────────────

class Asset:
    """
    Represents a single financial asset in a portfolio.

    Attributes:
        name              : Human-readable identifier (e.g. "BTC", "GOLD").
        allocation_pct    : Percentage of total portfolio allocated (0–100).
        expected_crash_pct: Expected decline in a worst-case crash scenario,
                           expressed as a negative percentage (-100 to 0).
    """

    def __init__(self, name: str, allocation_pct: float, expected_crash_pct: float) -> None:
        # ── Type validation ──────────────────────────────────
        if not isinstance(name, str) or not name.strip():
            raise AssetValidationError("Asset name must be a non-empty string.")

        if not isinstance(allocation_pct, (int, float)):
            raise AssetValidationError(
                f"allocation_pct must be numeric, got {type(allocation_pct).__name__}."
            )

        if not isinstance(expected_crash_pct, (int, float)):
            raise AssetValidationError(
                f"expected_crash_pct must be numeric, got {type(expected_crash_pct).__name__}."
            )

        # ── Value validation ─────────────────────────────────
        if allocation_pct < 0:
            raise AssetValidationError(
                f"Allocation for '{name}' cannot be negative ({allocation_pct}%)."
            )

        if not (-100 <= expected_crash_pct <= 0):
            raise AssetValidationError(
                f"Crash % for '{name}' must be between -100 and 0, got {expected_crash_pct}%."
            )

        self.name: str = name.strip()
        self.allocation_pct: float = float(allocation_pct)
        self.expected_crash_pct: float = float(expected_crash_pct)

    # ── Readable representation ──────────────────────────────
    def __repr__(self) -> str:
        return (
            f"Asset(name='{self.name}', allocation={self.allocation_pct}%, "
            f"crash={self.expected_crash_pct}%)"
        )


# ──────────────────────────────────────────────────────────────
#  Portfolio Class
# ──────────────────────────────────────────────────────────────

class Portfolio:
    """
    Models a multi-asset investment portfolio and exposes five core
    risk-analysis methods plus dynamic management operations.

    Attributes:
        total_value_inr     : Current market value of the portfolio (INR).
        monthly_expenses_inr: Holder's fixed monthly living expenses (INR).
        assets              : List of Asset objects that compose the portfolio.
    """

    # ── Construction & validation ────────────────────────────

    def __init__(
        self,
        total_value_inr: float,
        monthly_expenses_inr: float,
        assets: list[Asset],
    ) -> None:
        # Type checks
        if not isinstance(total_value_inr, (int, float)):
            raise PortfolioValidationError(
                f"total_value_inr must be numeric, got {type(total_value_inr).__name__}."
            )
        if not isinstance(monthly_expenses_inr, (int, float)):
            raise PortfolioValidationError(
                f"monthly_expenses_inr must be numeric, got {type(monthly_expenses_inr).__name__}."
            )
        if not isinstance(assets, list):
            raise PortfolioValidationError("assets must be a list of Asset objects.")

        # Value checks
        if total_value_inr <= 0:
            raise PortfolioValidationError(
                f"Total portfolio value must be positive, got ₹{total_value_inr:,.2f}."
            )
        if monthly_expenses_inr <= 0:
            raise PortfolioValidationError(
                f"Monthly expenses must be positive, got ₹{monthly_expenses_inr:,.2f}."
            )
        if not assets:
            raise PortfolioValidationError("Portfolio must contain at least one asset.")

        # Verify every element is an Asset
        for i, a in enumerate(assets):
            if not isinstance(a, Asset):
                raise PortfolioValidationError(
                    f"Item at index {i} is not an Asset (got {type(a).__name__})."
                )

        self.total_value_inr: float = float(total_value_inr)
        self.monthly_expenses_inr: float = float(monthly_expenses_inr)
        self.assets: list[Asset] = list(assets)  # shallow copy to own the list

        # Run aggregate validations (allocation sum, duplicate names)
        self._validate_portfolio()

    def _validate_portfolio(self) -> None:
        """Check constraints that span the entire asset list."""
        # Allocation must sum to 100%
        total_alloc = sum(a.allocation_pct for a in self.assets)
        if abs(total_alloc - 100.0) > 1e-6:
            raise PortfolioValidationError(
                f"Asset allocations must total 100%, currently {total_alloc:.2f}%."
            )

        # Warn (but do not crash) on duplicate asset names
        names = [a.name for a in self.assets]
        seen: set[str] = set()
        for n in names:
            if n in seen:
                print(f"⚠  Warning: duplicate asset name '{n}' detected.")
            seen.add(n)

    # ──────────────────────────────────────────────────────────
    #  Core Risk Methods (exactly 5)
    # ──────────────────────────────────────────────────────────

    def compute_post_crash_value(self, severity: float = 1.0) -> float:
        """
        Portfolio value after applying each asset's expected crash.

        Args:
            severity: Multiplier for crash magnitude (1.0 = full crash,
                      0.5 = moderate / 50% of worst-case).

        Returns:
            Post-crash portfolio value in INR.
        """
        if not (0.0 <= severity <= 1.0):
            raise ValueError("Severity must be between 0.0 and 1.0.")

        post_crash = 0.0
        for asset in self.assets:
            asset_value = self.total_value_inr * (asset.allocation_pct / 100)
            crash_impact = asset.expected_crash_pct / 100 * severity
            post_crash += asset_value * (1 + crash_impact)
        return post_crash

    def compute_runway_months(self, severity: float = 1.0) -> float:
        """
        Number of months the post-crash portfolio can cover living expenses.

        Args:
            severity: Crash severity multiplier (see compute_post_crash_value).

        Returns:
            Runway in months (float).
        """
        post_crash = self.compute_post_crash_value(severity)
        # monthly_expenses_inr is guaranteed > 0 by the constructor
        return post_crash / self.monthly_expenses_inr

    def ruin_test(self, severity: float = 1.0) -> str:
        """
        Determines if the portfolio survives at least 6 months of expenses
        after the crash scenario.

        Returns:
            'PASS' if runway ≥ 6 months, 'FAIL' otherwise.
        """
        return "PASS" if self.compute_runway_months(severity) >= 6 else "FAIL"

    def largest_risk_asset(self) -> Asset:
        """
        Identifies the asset contributing the most absolute risk to the
        portfolio, measured as allocation_pct × |expected_crash_pct|.

        Returns:
            The Asset with the highest risk contribution.
        """
        return max(
            self.assets,
            key=lambda a: a.allocation_pct * abs(a.expected_crash_pct),
        )

    def concentration_warning(self) -> bool:
        """
        Checks if any single asset exceeds 40% of the portfolio allocation.

        Returns:
            True if concentration risk exists, False otherwise.
        """
        return any(a.allocation_pct > 40 for a in self.assets)

    # ──────────────────────────────────────────────────────────
    #  Dynamic Management
    # ──────────────────────────────────────────────────────────

    def add_asset(self, asset: Asset) -> None:
        """Add an asset and re-validate portfolio constraints."""
        if not isinstance(asset, Asset):
            raise PortfolioValidationError("Can only add Asset instances.")
        self.assets.append(asset)
        self._validate_portfolio()

    def remove_asset(self, name: str) -> Asset:
        """
        Remove an asset by name and re-validate.

        Raises PortfolioValidationError if the asset is not found or
        removing it would leave the portfolio empty.
        """
        for i, asset in enumerate(self.assets):
            if asset.name == name:
                removed = self.assets.pop(i)
                if not self.assets:
                    # Restore and raise — portfolio cannot be empty
                    self.assets.append(removed)
                    raise PortfolioValidationError(
                        "Cannot remove the last asset from a portfolio."
                    )
                self._validate_portfolio()
                return removed
        raise PortfolioValidationError(f"Asset '{name}' not found in portfolio.")

    def update_asset(self, name: str, **kwargs) -> None:
        """
        Update allocation_pct and/or expected_crash_pct for a named asset.

        Accepted keyword arguments:
            allocation_pct    (float)
            expected_crash_pct (float)

        Raises AssetValidationError on invalid values and re-validates
        portfolio-level constraints after the update.
        """
        target: Asset | None = None
        for asset in self.assets:
            if asset.name == name:
                target = asset
                break

        if target is None:
            raise PortfolioValidationError(f"Asset '{name}' not found in portfolio.")

        # Stash originals for rollback on failure
        orig_alloc = target.allocation_pct
        orig_crash = target.expected_crash_pct

        try:
            if "allocation_pct" in kwargs:
                val = kwargs["allocation_pct"]
                if not isinstance(val, (int, float)):
                    raise AssetValidationError(
                        f"allocation_pct must be numeric, got {type(val).__name__}."
                    )
                if val < 0:
                    raise AssetValidationError(
                        f"Allocation cannot be negative ({val}%)."
                    )
                target.allocation_pct = float(val)

            if "expected_crash_pct" in kwargs:
                val = kwargs["expected_crash_pct"]
                if not isinstance(val, (int, float)):
                    raise AssetValidationError(
                        f"expected_crash_pct must be numeric, got {type(val).__name__}."
                    )
                if not (-100 <= val <= 0):
                    raise AssetValidationError(
                        f"Crash % must be between -100 and 0, got {val}%."
                    )
                target.expected_crash_pct = float(val)

            self._validate_portfolio()

        except (AssetValidationError, PortfolioValidationError):
            # Rollback to keep portfolio in a consistent state
            target.allocation_pct = orig_alloc
            target.expected_crash_pct = orig_crash
            raise

    # ──────────────────────────────────────────────────────────
    #  Bonus: CLI Allocation Visualisation
    # ──────────────────────────────────────────────────────────

    def display_allocation_bar(self, bar_width: int = 40) -> str:
        """
        Render a simple text bar chart of asset allocations.

        Example output:
            BTC     | ████████████           | 30.0%
            NIFTY50 | ████████████████       | 40.0%
        """
        lines: list[str] = []
        max_name_len = max(len(a.name) for a in self.assets)

        for asset in self.assets:
            filled = round(bar_width * asset.allocation_pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            lines.append(
                f"  {asset.name:<{max_name_len}}  │ {bar} │ {asset.allocation_pct:5.1f}%"
            )
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
#  Demonstration
# ──────────────────────────────────────────────────────────────

def main() -> None:
    """Build the portfolio from the specification and print all results."""

    # Ensure Unicode output works on Windows terminals (cp1252 → utf-8)
    sys.stdout.reconfigure(encoding="utf-8")

    portfolio_data = {
        "total_value_inr": 10_000_000,
        "monthly_expenses_inr": 80_000,
        "assets": [
            {"name": "BTC", "allocation_pct": 30, "expected_crash_pct": -80},
            {"name": "NIFTY50", "allocation_pct": 40, "expected_crash_pct": -40},
            {"name": "GOLD", "allocation_pct": 20, "expected_crash_pct": -15},
            {"name": "CASH", "allocation_pct": 10, "expected_crash_pct": 0},
        ],
    }

    # ── Build objects ────────────────────────────────────────
    assets = [Asset(**a) for a in portfolio_data["assets"]]
    portfolio = Portfolio(
        total_value_inr=portfolio_data["total_value_inr"],
        monthly_expenses_inr=portfolio_data["monthly_expenses_inr"],
        assets=assets,
    )

    # ── Scenario definitions ─────────────────────────────────
    scenarios = {
        "Full Crash (100%)": 1.0,
        "Moderate Crash (50%)": 0.5,
    }

    # ── Header ───────────────────────────────────────────────
    print("=" * 62)
    print("        PORTFOLIO RISK CALCULATOR — CRASH ANALYSIS")
    print("=" * 62)
    print(f"  Portfolio Value : ₹{portfolio.total_value_inr:>14,.2f}")
    print(f"  Monthly Expenses: ₹{portfolio.monthly_expenses_inr:>14,.2f}")
    print("-" * 62)

    # ── Allocation chart ─────────────────────────────────────
    print("\n  📊 Asset Allocation\n")
    print(portfolio.display_allocation_bar())

    # ── Side-by-side scenario comparison ─────────────────────
    print("\n" + "=" * 62)
    print("  CRASH SCENARIO COMPARISON")
    print("=" * 62)

    header = f"  {'Metric':<28}"
    for label in scenarios:
        header += f"{'│ ' + label:>18}"
    print(header)
    print("  " + "─" * 58)

    # Post-crash values
    row_val = f"  {'Post-Crash Value (₹)':<28}"
    for severity in scenarios.values():
        val = portfolio.compute_post_crash_value(severity)
        row_val += f"│ ₹{val:>13,.0f} "
    print(row_val)

    # Runway months
    row_run = f"  {'Runway (months)':<28}"
    for severity in scenarios.values():
        months = portfolio.compute_runway_months(severity)
        row_run += f"│ {months:>14.1f} "
    print(row_run)

    # Ruin test
    row_ruin = f"  {'Ruin Test':<28}"
    for severity in scenarios.values():
        result = portfolio.ruin_test(severity)
        row_ruin += f"│ {result:>14} "
    print(row_ruin)

    print("  " + "─" * 58)

    # ── Single-value outputs ─────────────────────────────────
    risk_asset = portfolio.largest_risk_asset()
    risk_score = risk_asset.allocation_pct * abs(risk_asset.expected_crash_pct)

    print(f"\n  🔴 Largest Risk Asset : {risk_asset.name} "
          f"(score = {risk_asset.allocation_pct}% × "
          f"|{risk_asset.expected_crash_pct}%| = {risk_score:.0f})")

    conc = portfolio.concentration_warning()
    print(f"  ⚠️  Concentration Risk : {'YES — rebalance recommended' if conc else 'No'}")

    print("\n" + "=" * 62)


if __name__ == "__main__":
    main()
