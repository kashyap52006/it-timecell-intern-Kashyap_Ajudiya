# Timecell Internship Assignment

## Introduction

This project is part of the Timecell internship assessment.

---

## Task 1: Portfolio Risk Analysis

### Objective

To calculate how a portfolio performs under a crash scenario and determine whether the user can financially survive.

---

## Approach

The solution follows these steps:

1. **Calculate Post-Crash Value**

   * Each asset is reduced based on its crash percentage
   * Formula used:

     ```
     asset_value = total_value × allocation × (1 + crash%)
     ```
   * All asset values are summed to get final portfolio value

2. **Calculate Runway**

   * Runway shows how many months the user can survive
   * Formula:

     ```
     runway = post_crash_value / monthly_expenses
     ```

3. **Ruin Test**

   * If runway > 12 → PASS (safe)
   * Else → FAIL (risky)

4. **Identify Highest Risk Asset**

   * Based on:

     ```
     risk = allocation × crash%
     ```
   * Asset with highest value is considered most risky

5. **Check Concentration Risk**

   * If any asset > 40% → flagged as risky

---

## Thought Process

* Focused on breaking the problem into small steps
* Used simple mathematical formulas for clarity
* Ensured calculations are easy to understand and debug
* Designed logic to be clean and modular

---

## Edge Cases Handled

* Zero monthly expenses (infinite runway)
* Missing or incorrect allocation values
* Extremely high crash percentages
* Empty asset list

---

## Example Output

```
Post Crash Value: 5,700,000 INR
Runway: 57 months
Ruin Test: PASS
Highest Risk Asset: BTC
```

---

## Key Learnings

* Importance of diversification
* Understanding portfolio risk
* Translating real-world finance into code
* Handling edge cases properly

---
