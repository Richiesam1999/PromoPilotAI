"""Ground-truth demand model (log-log own & cross elasticity + seasonality + holidays + region).

Used (a) to generate synthetic history, (b) as the 'real world' in the plan evaluator.
The optimizer/agents never import this: they estimate parameters from sales_history.csv.
"""
import numpy as np
import pandas as pd

from .config import DATA, GOLD


class DemandModel:
    def __init__(self, data_dir=DATA, gold_dir=GOLD):
        self.products = pd.read_csv(data_dir / "products.csv")
        self.stores = pd.read_csv(data_dir / "stores.csv")
        tp = pd.read_csv(gold_dir / "true_demand_params.csv").set_index("sku").loc[self.products.sku]
        self.skus = list(self.products.sku)
        self.store_ids = list(self.stores.store_id)
        self.sku_idx = {s: i for i, s in enumerate(self.skus)}
        self.store_idx = {s: i for i, s in enumerate(self.store_ids)}
        self.base = tp.base_weekly_units.to_numpy(float)
        self.E = tp.own_elasticity.to_numpy(float)
        self.peak = tp.season_peak_month.to_numpy(float)
        self.amp = tp.season_amp.to_numpy(float)
        self.C = pd.read_csv(gold_dir / "true_cross_elasticity.csv", index_col=0).loc[self.skus, self.skus].to_numpy(float)
        self.size = (self.stores.sq_ft / 40000.0).to_numpy(float)
        ri = pd.read_csv(gold_dir / "region_category_index.csv")
        lut = {(r.region, r.category): r.index for r in ri.itertuples()}
        self.region_idx = np.array([[lut.get((reg, cat), 1.0) for cat in self.products.category]
                                    for reg in self.stores.region])
        self.holidays = pd.read_csv(data_dir / "holidays.csv", parse_dates=["holiday_date"])
        lifts = pd.read_csv(data_dir / "holiday_lifts.csv")
        self._lift = {(r.name, r.category): r.lift for r in lifts.itertuples()}
        self.cats = list(self.products.category)

    def seasonal(self, ws):
        frac = (ws.dayofyear - 1) / 365.25 * 12 + 0.5 + 0.25  # mid-week
        return np.maximum(0.2, 1 + self.amp * np.cos(2 * np.pi * (frac - self.peak) / 12))

    def holiday_factor(self, ws):
        if not hasattr(self, "_hf_cache"):
            self._hf_cache = {}
        if ws not in self._hf_cache:
            f = np.ones(len(self.skus))
            for h in self.holidays.itertuples():
                d = (h.holiday_date - ws).days
                w = 1.0 if 0 <= d <= 6 else 0.4 if 7 <= d <= 13 else 0.0
                if w:
                    f += w * np.array([self._lift.get((h.name, c), 0.0) for c in self.cats])
            self._hf_cache[ws] = f
        return self._hf_cache[ws]

    def baseline_units(self, ws):
        """Expected units at regular price per (store, sku) for the week starting ws."""
        return self.base[None, :] * self.size[:, None] * self.region_idx * (self.seasonal(ws) * self.holiday_factor(ws))[None, :]

    def expected_units(self, ws, price_ratio):
        """price_ratio: (n_store, n_sku) sell_price / list_price."""
        lr = np.log(np.clip(price_ratio, 0.2, 1.5))
        log_mult = np.clip(lr * self.E[None, :] + lr @ self.C.T, -1.5, 1.5)
        return self.baseline_units(ws) * np.exp(log_mult)
