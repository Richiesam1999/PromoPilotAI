"""Central configuration: paths, dates, seeds, global constants."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INPUTS = DATA / "inputs"   # messy text inputs (what the Input Agent will read)
GOLD = DATA / "gold"       # ground truth: true demand params + gold extractions (NEVER shown to agents)

SEED = 42
AS_OF = pd.Timestamp("2026-10-11")             # "today" in the scenario
HISTORY_WEEKS = 104
HISTORY_LAST_WEEK = pd.Timestamp("2026-10-05")  # Monday of last full history week
PLAN_START = pd.Timestamp("2026-10-26")         # Monday
PLAN_WEEKS = 5
PLAN_END = pd.Timestamp("2026-11-29")           # Sunday
COMPETITOR_SNAPSHOT = pd.Timestamp("2026-10-02")  # deliberately stale vs. field notes of Oct 9

EXEC_COST = 15.0          # $ per store-SKU-week promoted (shelf tags + signage)
BASELINE_DISCOUNT = 0.20  # the flat-20% baseline


def plan_weeks():
    return list(pd.date_range(PLAN_START, periods=PLAN_WEEKS, freq="7D"))
