"""Phase 2a: estimate own- and cross-price elasticities and a baseline-demand forecaster from sales history only.

Reads: sales_history, products, stores, holidays, holiday_lifts.  Nothing from the ground-truth folder.
Run:   python -m src.elasticity     (writes data/estimates/own_elasticity.csv, cross_elasticity.csv, relationships.csv)

Model: one equation per SKU k, one row per (week, store):

    ln(units + 0.5) - ln(holiday_factor_k(week))
        = store effect + annual seasonality (Fourier terms)
          + E_k * ln(r_k) + sum_{j != k} C_kj * ln(r_j) + noise,         r = sell_price / list_price

* Every equation has the same regressors, so all SKUs are solved with one least-squares call.
* ln(units + 0.5) is nearly unbiased for Poisson counts (the +0.5 cancels the Jensen term), so low-volume SKUs are not biased.
* Holiday lifts are taken as given from the holiday_lifts input table (used as an offset), not estimated.
* Seasonality is estimated per SKU with N_HARMONICS sine/cosine pairs; the store effect absorbs store size and region.
* Orientation of C: row = SKU whose demand responds, column = SKU whose price moves (substitute > 0, complement < 0).
* Raw OLS gives 48 x 47 cross coefficients; most are noise, so a significance screen keeps only credible links.
  Standard errors are kept so Phase 4 can sample elasticities for Monte Carlo (see DemandEstimates.draw).
"""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .config import DATA

N_HARMONICS = 4            # 4 pairs are enough even for the sharply peaked sunscreen season
T_SAME_SUBCATEGORY = 2.0   # |t| needed to keep a positive (substitute) link inside the same subcategory
T_OTHER = 3.5              # |t| needed to keep any other cross link (about 2,200 candidate pairs, so be strict)


def seasonal_basis(ws, n_harm=N_HARMONICS):
    ang = 2 * np.pi * ws.dayofyear / 365.25
    return np.array([f(h * ang) for h in range(1, n_harm + 1) for f in (np.sin, np.cos)])


def common_design(weeks, n_stores, n_harm=N_HARMONICS):
    """Regressors shared by every SKU equation. Rows: week-major, then store.
    Columns: intercept, store dummies (first store dropped), seasonality."""
    dummies = np.eye(n_stores)[:, 1:]
    blocks = [np.hstack([np.ones((n_stores, 1)), dummies, np.tile(seasonal_basis(ws, n_harm), (n_stores, 1))])
              for ws in weeks]
    return np.vstack(blocks)


def holiday_factor(ws, cats, hol_dates, hol_names, lift):
    """Multiplicative holiday lift per SKU for the week starting ws: full lift in the holiday week,
    40% of it in the week before. Built only from the holidays and holiday_lifts input tables."""
    f = np.ones(len(cats))
    for d0, nm in zip(hol_dates, hol_names):
        d = (d0 - ws).days
        w = 1.0 if 0 <= d <= 6 else 0.4 if 7 <= d <= 13 else 0.0
        if w:
            f = f + w * np.array([lift.get((nm, c), 0.0) for c in cats])
    return f


def screen_cross(C_raw, C_se, subcat, t_same=T_SAME_SUBCATEGORY, t_other=T_OTHER):
    """Keep a cross link only if it is statistically credible. Positive links inside one subcategory
    (plausible substitutes) need a lower bar than links between unrelated SKUs."""
    t = np.abs(C_raw) / np.maximum(C_se, 1e-9)
    substitute_like = (subcat[:, None] == subcat[None, :]) & (C_raw > 0)
    keep = t >= np.where(substitute_like, t_same, t_other)
    np.fill_diagonal(keep, False)
    return np.where(keep, C_raw, 0.0)


@dataclass
class DemandEstimates:
    skus: list
    store_ids: list
    cats: list
    E: np.ndarray            # (K,) own-price elasticity
    E_se: np.ndarray         # (K,) standard error of E
    C_raw: np.ndarray        # (K, K) unscreened cross elasticities, diagonal 0
    C_se: np.ndarray         # (K, K) standard errors of C_raw
    C: np.ndarray            # (K, K) screened (sparse) cross elasticities used for planning
    coef_base: np.ndarray    # (p0, K) store effects + seasonality on the log scale
    n_harm: int
    hol_dates: list
    hol_names: list
    lift: dict
    resid_sd: np.ndarray     # (K,) residual standard deviation of each equation (log scale)

    def baseline_units(self, ws):
        """Expected units per (store, sku) at regular price for the week starting ws."""
        Z = common_design([ws], len(self.store_ids), self.n_harm)
        off = np.log(holiday_factor(ws, self.cats, self.hol_dates, self.hol_names, self.lift))
        return np.exp(Z @ self.coef_base + off[None, :])

    def expected_units(self, ws, price_ratio):
        """price_ratio: (n_store, n_sku) sell_price / list_price. Same functional form as the planner will assume."""
        lr = np.log(np.clip(price_ratio, 0.2, 1.5))
        log_mult = np.clip(lr * self.E[None, :] + lr @ self.C.T, -1.5, 1.5)
        return self.baseline_units(ws) * np.exp(log_mult)

    def draw(self, rng):
        """One plausible parameter set (for Monte Carlo): E and the kept cross links perturbed by their standard errors."""
        E = self.E + rng.normal(size=self.E.shape) * self.E_se
        C = np.where(self.C != 0, self.C + rng.normal(size=self.C.shape) * self.C_se, 0.0)
        return replace(self, E=E, C=C)

    def own_frame(self):
        return pd.DataFrame({"sku": self.skus, "elasticity": self.E, "se": self.E_se})

    def cross_frame(self, raw=False):
        return pd.DataFrame(self.C_raw if raw else self.C, index=self.skus, columns=self.skus)

    def pairs(self):
        """Credible cross links as a tidy table (also the input the Phase 3 Relationship agent can reason over)."""
        rows = []
        for a, b in zip(*np.nonzero(self.C)):
            rows.append(dict(affected_sku=self.skus[a], price_sku=self.skus[b], elasticity=self.C[a, b],
                             t_stat=abs(self.C_raw[a, b]) / max(self.C_se[a, b], 1e-9),
                             relation="substitute" if self.C[a, b] > 0 else "complement"))
        cols = ["affected_sku", "price_sku", "elasticity", "t_stat", "relation"]
        return pd.DataFrame(rows, columns=cols).sort_values("elasticity", key=lambda s: s.abs(), ascending=False)


def fit(data_dir=DATA, n_harm=N_HARMONICS, t_same=T_SAME_SUBCATEGORY, t_other=T_OTHER):
    products = pd.read_csv(data_dir / "products.csv")
    stores = pd.read_csv(data_dir / "stores.csv")
    sales = pd.read_csv(data_dir / "sales_history.csv", parse_dates=["week_start"])
    hol = pd.read_csv(data_dir / "holidays.csv", parse_dates=["holiday_date"])
    lifts = pd.read_csv(data_dir / "holiday_lifts.csv")

    skus, store_ids, cats = list(products.sku), list(stores.store_id), list(products.category)
    lift = {(n, c): v for n, c, v in zip(lifts["name"], lifts["category"], lifts["lift"])}
    hol_dates, hol_names = list(hol["holiday_date"]), list(hol["name"])
    weeks = [pd.Timestamp(w) for w in sorted(sales.week_start.unique())]
    W, S, K = len(weeks), len(store_ids), len(skus)
    assert len(sales) == W * S * K, "sales history must be a complete week x store x SKU panel"

    wi = sales.week_start.map({w: i for i, w in enumerate(weeks)}).to_numpy().astype(int)
    si = sales.store_id.map({s: i for i, s in enumerate(store_ids)}).to_numpy().astype(int)
    ki = sales.sku.map({k: i for i, k in enumerate(skus)}).to_numpy().astype(int)
    lp = products.list_price.to_numpy(float)
    U = np.zeros((W, S, K))
    LR = np.zeros((W, S, K))
    U[wi, si, ki] = sales.units.to_numpy(float)
    LR[wi, si, ki] = np.log(sales.sell_price.to_numpy(float) / lp[ki])

    off = np.log(np.vstack([holiday_factor(ws, cats, hol_dates, hol_names, lift) for ws in weeks]))   # (W, K)
    Y = (np.log(U + 0.5) - off[:, None, :]).reshape(W * S, K)
    P = LR.reshape(W * S, K)
    Z = common_design(weeks, S, n_harm)
    X = np.hstack([Z, P])
    p0 = Z.shape[1]

    B, *_ = np.linalg.lstsq(X, Y, rcond=None)                        # (p0 + K, K): one column per SKU equation
    resid = Y - X @ B
    s2 = (resid ** 2).sum(axis=0) / (X.shape[0] - X.shape[1])         # residual variance per equation
    inv_diag = np.diag(np.linalg.inv(X.T @ X))[p0:]                   # variance factor of each price column
    se = np.sqrt(np.outer(inv_diag, s2))                              # (price j, equation k)
    coef = B[p0:, :]                                                  # (price j, equation k)

    E = np.diag(coef).copy()
    E_se = np.diag(se).copy()
    C_raw = coef.T.copy()
    np.fill_diagonal(C_raw, 0.0)
    C_se = se.T.copy()
    np.fill_diagonal(C_se, 0.0)
    C = screen_cross(C_raw, C_se, products.subcategory.to_numpy(), t_same, t_other)

    return DemandEstimates(skus=skus, store_ids=store_ids, cats=cats, E=E, E_se=E_se, C_raw=C_raw, C_se=C_se, C=C,
                           coef_base=B[:p0, :], n_harm=n_harm, hol_dates=hol_dates, hol_names=hol_names, lift=lift,
                           resid_sd=np.sqrt(s2))


def main():
    est = fit()
    out = DATA / "estimates"
    out.mkdir(exist_ok=True)
    est.own_frame().to_csv(out / "own_elasticity.csv", index=False)
    est.cross_frame().to_csv(out / "cross_elasticity.csv")
    pairs = est.pairs()
    pairs.to_csv(out / "relationships.csv", index=False)
    print(f"own elasticity: mean {est.E.mean():.2f}, range [{est.E.min():.2f}, {est.E.max():.2f}], "
          f"median s.e. {np.median(est.E_se):.2f}")
    print(f"cross links kept: {len(pairs)} ({(pairs.relation == 'substitute').sum()} substitutes, "
          f"{(pairs.relation == 'complement').sum()} complements)")
    print(pairs.head(12).to_string(index=False))
    print("written to", out)


if __name__ == "__main__":
    main()
