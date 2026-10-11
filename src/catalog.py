"""Static definition of the synthetic retail 'world': products, stores, segments, holidays.

Design note: the sales data is synthetic but M5/Walmart-shaped (weekly store x SKU x sell_price x units).
Ground-truth demand parameters live in data/gold and are only used by the generator, the
evaluator (as the 'real world') and tests. Agents must *estimate* elasticities from sales history.
"""
import numpy as np
import pandas as pd

# sku, name, category, brand, private_label, list_price, margin, base_weekly_units (store size 1.0),
# own_elasticity, subcategory, season_peak_month, season_amp
_P = [
    # Beverages
    ("B001", "FizzUp Cola 12pk", "Beverages", "FizzUp", 0, 6.99, 0.24, 120, -2.3, "Soda", 7, 0.15),
    ("B002", "FizzUp Diet Cola 12pk", "Beverages", "FizzUp", 0, 6.99, 0.24, 70, -2.1, "Soda", 7, 0.15),
    ("B003", "Citrus Pop Lemon-Lime 12pk", "Beverages", "Citrus Pop", 0, 6.49, 0.23, 60, -2.2, "Soda", 7, 0.15),
    ("B004", "Store Brand Cola 12pk", "Beverages", "Store Brand", 1, 4.99, 0.38, 80, -1.7, "Soda", 7, 0.15),
    ("B005", "Sunrise Orange Juice 52oz", "Beverages", "Sunrise", 0, 4.49, 0.27, 55, -1.5, "Juice", 1, 0.08),
    ("B006", "Pure Spring Water 24pk", "Beverages", "Pure Spring", 0, 4.99, 0.30, 90, -1.2, "Water", 7, 0.20),
    ("B007", "Volt Energy Drink 4pk", "Beverages", "Volt", 0, 7.99, 0.33, 40, -1.8, "Energy Drinks", 6, 0.05),
    ("B008", "Brewed Iced Tea 6pk", "Beverages", "Brewed", 0, 5.49, 0.31, 35, -1.9, "Tea", 7, 0.25),
    # Snacks
    ("S001", "Crunchy Chips Original 10oz", "Snacks", "Crunchy", 0, 3.99, 0.30, 100, -2.2, "Chips", 1, 0.0),
    ("S002", "Crunchy Chips Sour Cream 10oz", "Snacks", "Crunchy", 0, 3.99, 0.30, 70, -2.1, "Chips", 1, 0.0),
    ("S003", "Casa Tortilla Chips 12oz", "Snacks", "Casa", 0, 3.79, 0.32, 75, -2.0, "Chips", 1, 0.0),
    ("S004", "Store Brand Potato Chips 10oz", "Snacks", "Store Brand", 1, 2.49, 0.42, 60, -1.6, "Chips", 1, 0.0),
    ("S005", "Casa Salsa Verde 16oz", "Snacks", "Casa", 0, 3.29, 0.36, 45, -1.4, "Salsa", 1, 0.0),
    ("S006", "Choc Chunk Cookies 13oz", "Snacks", "Choc Chunk", 0, 3.99, 0.33, 55, -2.0, "Cookies", 1, 0.0),
    ("S007", "Oat Crunch Granola Bars 12ct", "Snacks", "Oat Crunch", 0, 4.49, 0.34, 50, -1.7, "Bars", 1, 0.0),
    ("S008", "Trailblazer Trail Mix 16oz", "Snacks", "Trailblazer", 0, 6.99, 0.36, 25, -1.6, "Trail Mix", 7, 0.30),
    # Dairy & Breakfast
    ("D001", "Meadow Whole Milk 1gal", "Dairy_Breakfast", "Meadow", 0, 3.79, 0.20, 150, -0.7, "Milk", 1, 0.0),
    ("D002", "Meadow Greek Yogurt 32oz", "Dairy_Breakfast", "Meadow", 0, 4.99, 0.28, 60, -1.5, "Yogurt", 1, 0.0),
    ("D003", "Valley Cheddar Block 16oz", "Dairy_Breakfast", "Valley", 0, 5.49, 0.30, 50, -1.4, "Cheese", 1, 0.0),
    ("D004", "Store Brand Butter 1lb", "Dairy_Breakfast", "Store Brand", 1, 4.29, 0.32, 45, -1.2, "Butter", 1, 0.05),
    ("D005", "Morning Crunch Cereal 18oz", "Dairy_Breakfast", "Morning Crunch", 0, 4.79, 0.31, 80, -1.9, "Cereal", 1, 0.0),
    ("D006", "Store Brand Cereal 18oz", "Dairy_Breakfast", "Store Brand", 1, 2.99, 0.40, 55, -1.5, "Cereal", 1, 0.0),
    ("D007", "Hearty Rolled Oats 42oz", "Dairy_Breakfast", "Hearty", 0, 4.99, 0.29, 35, -1.3, "Oats", 1, 0.15),
    ("D008", "Roast Masters Ground Coffee 12oz", "Dairy_Breakfast", "Roast Masters", 0, 8.99, 0.34, 45, -1.1, "Coffee", 1, 0.05),
    # Pantry & Frozen
    ("P001", "Bella Spaghetti 16oz", "Pantry_Frozen", "Bella", 0, 1.79, 0.28, 120, -1.6, "Pasta", 1, 0.0),
    ("P002", "Bella Marinara Sauce 24oz", "Pantry_Frozen", "Bella", 0, 3.49, 0.33, 70, -1.5, "Pasta Sauce", 1, 0.0),
    ("P003", "Store Brand Spaghetti 16oz", "Pantry_Frozen", "Store Brand", 1, 1.19, 0.38, 80, -1.3, "Pasta", 1, 0.0),
    ("P004", "Nonna's Frozen Pizza Pepperoni", "Pantry_Frozen", "Nonna's", 0, 5.99, 0.30, 65, -2.2, "Frozen Pizza", 1, 0.05),
    ("P005", "Nonna's Frozen Pizza Four Cheese", "Pantry_Frozen", "Nonna's", 0, 5.99, 0.30, 50, -2.1, "Frozen Pizza", 1, 0.05),
    ("P006", "Creamery Vanilla Ice Cream 48oz", "Pantry_Frozen", "Creamery", 0, 5.99, 0.32, 40, -1.9, "Ice Cream", 7, 0.45),
    ("P007", "Golden Long Grain Rice 5lb", "Pantry_Frozen", "Golden", 0, 4.99, 0.26, 45, -1.0, "Rice", 1, 0.0),
    ("P008", "Hearthside Chicken Noodle Soup 18oz", "Pantry_Frozen", "Hearthside", 0, 2.49, 0.34, 70, -1.4, "Soup", 1, 0.35),
    # Household
    ("H001", "SparkleClean Dish Soap 24oz", "Household", "SparkleClean", 0, 3.99, 0.35, 60, -1.5, "Dish Soap", 1, 0.0),
    ("H002", "Store Brand Dish Soap 24oz", "Household", "Store Brand", 1, 2.49, 0.45, 45, -1.3, "Dish Soap", 1, 0.0),
    ("H003", "FreshWave Laundry Detergent 100oz", "Household", "FreshWave", 0, 13.99, 0.30, 40, -1.6, "Laundry", 1, 0.0),
    ("H004", "Store Brand Laundry Detergent 100oz", "Household", "Store Brand", 1, 8.99, 0.40, 30, -1.4, "Laundry", 1, 0.0),
    ("H005", "SoftRoll Paper Towels 6ct", "Household", "SoftRoll", 0, 8.99, 0.27, 55, -1.7, "Paper Towels", 1, 0.0),
    ("H006", "SoftRoll Toilet Paper 12ct", "Household", "SoftRoll", 0, 11.99, 0.26, 65, -1.1, "Toilet Paper", 1, 0.0),
    ("H007", "StrongHold Trash Bags 40ct", "Household", "StrongHold", 0, 7.49, 0.33, 35, -1.4, "Trash Bags", 1, 0.0),
    ("H008", "Glow Glass Cleaner 23oz", "Household", "Glow", 0, 3.49, 0.38, 25, -1.5, "Cleaners", 1, 0.0),
    # Personal care
    ("C001", "Silk&Shine Shampoo 12oz", "Personal_Care", "Silk&Shine", 0, 6.49, 0.38, 45, -1.6, "Shampoo", 1, 0.0),
    ("C002", "Silk&Shine Conditioner 12oz", "Personal_Care", "Silk&Shine", 0, 6.49, 0.38, 35, -1.6, "Conditioner", 1, 0.0),
    ("C003", "BrightSmile Toothpaste 4oz", "Personal_Care", "BrightSmile", 0, 3.99, 0.40, 55, -1.5, "Toothpaste", 1, 0.0),
    ("C004", "Store Brand Toothpaste 4oz", "Personal_Care", "Store Brand", 1, 1.99, 0.50, 40, -1.3, "Toothpaste", 1, 0.0),
    ("C005", "AquaFresh Body Wash 18oz", "Personal_Care", "AquaFresh", 0, 5.99, 0.38, 40, -1.7, "Body Wash", 1, 0.0),
    ("C006", "DryGuard Deodorant 2.6oz", "Personal_Care", "DryGuard", 0, 4.99, 0.40, 45, -1.6, "Deodorant", 1, 0.0),
    ("C007", "PureTouch Hand Soap 11oz", "Personal_Care", "PureTouch", 0, 2.99, 0.42, 50, -1.4, "Hand Soap", 1, 0.0),
    ("C008", "SunShield SPF50 Sunscreen 8oz", "Personal_Care", "SunShield", 0, 9.99, 0.45, 30, -1.8, "Sun Care", 7, 0.80),
]

PERISHABLE = {"D001": 10, "D002": 28, "D003": 60, "D004": 90, "B005": 14, "P006": 180}  # shelf life, days

# Directed complement pairs: (j, k, value) = % change in demand of j for +1% price of k (negative = complement)
COMPLEMENTS = [
    ("S001", "S005", -0.35), ("S005", "S001", -0.30), ("S002", "S005", -0.35), ("S005", "S002", -0.30),
    ("S003", "S005", -0.45), ("S005", "S003", -0.40),
    ("P001", "P002", -0.50), ("P002", "P001", -0.45), ("P003", "P002", -0.40), ("P002", "P003", -0.30),
    ("D005", "D001", -0.25), ("D006", "D001", -0.25), ("D001", "D005", -0.20),
    ("C001", "C002", -0.60), ("C002", "C001", -0.55),
]

# store_id, city, region, sq_ft
STORES = [
    ("S01", "Boston", "Northeast", 40000), ("S02", "Philadelphia", "Northeast", 32000),
    ("S03", "Atlanta", "South", 44000), ("S04", "Dallas", "South", 52000),
    ("S05", "Chicago", "Midwest", 48000), ("S06", "Columbus", "Midwest", 28000),
    ("S07", "Seattle", "West", 36000), ("S08", "Los Angeles", "West", 56000),
]
SQFT_REF = 40000.0

REGION_CATEGORY_INDEX = {
    "Northeast": {"Pantry_Frozen": 1.10}, "South": {"Beverages": 1.20, "Pantry_Frozen": 0.90},
    "Midwest": {"Snacks": 1.15, "Dairy_Breakfast": 1.10}, "West": {"Personal_Care": 1.15, "Beverages": 0.95},
}

# segment_id, name, customer_share, raw price-sensitivity multiplier, category affinities
SEGMENTS = [
    ("SEG_BARGAIN", "Bargain Hunters", 0.35, 1.35, {"Household": 1.2, "Personal_Care": 1.1, "Pantry_Frozen": 1.1}),
    ("SEG_FAMILY", "Loyal Families", 0.30, 1.00, {"Dairy_Breakfast": 1.3, "Pantry_Frozen": 1.2, "Snacks": 1.1}),
    ("SEG_PREMIUM", "Premium Seekers", 0.15, 0.65, {"Personal_Care": 1.2, "Beverages": 1.1, "Dairy_Breakfast": 1.1}),
    ("SEG_OCCASIONAL", "Occasional Shoppers", 0.20, 1.15, {"Snacks": 1.2, "Beverages": 1.2}),
]

HOLIDAY_LIFT = {
    "Halloween": {"Snacks": 0.30, "Beverages": 0.10},
    "Thanksgiving": {"Dairy_Breakfast": 0.20, "Pantry_Frozen": 0.30, "Beverages": 0.10},
    "Black Friday": {"Household": 0.25, "Personal_Care": 0.20},
    "Christmas": {"Snacks": 0.20, "Beverages": 0.15, "Dairy_Breakfast": 0.10, "Pantry_Frozen": 0.15},
    "New Year's Day": {"Beverages": 0.10},
    "Super Bowl": {"Snacks": 0.40, "Beverages": 0.25, "Pantry_Frozen": 0.15},
    "Memorial Day": {"Beverages": 0.20, "Snacks": 0.15, "Personal_Care": 0.15},
    "Independence Day": {"Beverages": 0.25, "Snacks": 0.20, "Personal_Care": 0.15},
    "Labor Day": {"Beverages": 0.15, "Snacks": 0.10},
}


def products_df():
    rows = []
    for p in _P:
        sku, name, cat, brand, pl, price, margin = p[:7]
        rows.append(dict(sku=sku, name=name, category=cat, subcategory=p[9], brand=brand, is_private_label=pl,
                         list_price=price, unit_cost=round(price * (1 - margin), 2),
                         shelf_life_days=PERISHABLE.get(sku, np.nan)))
    return pd.DataFrame(rows)


def true_demand_params_df():
    return pd.DataFrame([dict(sku=p[0], base_weekly_units=p[7], own_elasticity=p[8],
                              season_peak_month=p[10], season_amp=p[11]) for p in _P])


def true_cross_matrix(prod: pd.DataFrame) -> pd.DataFrame:
    skus = list(prod.sku)
    idx = {s: i for i, s in enumerate(skus)}
    C = np.zeros((len(skus), len(skus)))
    sub = prod.subcategory.to_numpy()
    pl = prod.is_private_label.to_numpy()
    for j in range(len(skus)):
        for k in range(len(skus)):
            if j != k and sub[j] == sub[k]:
                C[j, k] = 0.55 if (pl[j] and not pl[k]) else 0.30 if (pl[k] and not pl[j]) else 0.45
    for j, k, v in COMPLEMENTS:
        C[idx[j], idx[k]] = v
    return pd.DataFrame(C, index=skus, columns=skus)


def stores_df():
    return pd.DataFrame(STORES, columns=["store_id", "city", "region", "sq_ft"])


def _nth_weekday(year, month, weekday, n):
    d = pd.Timestamp(year, month, 1)
    return d + pd.Timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year, month, weekday):
    d = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    return d - pd.Timedelta(days=(d.weekday() - weekday) % 7)


def holidays_df(years=(2024, 2025, 2026)):
    rows = []
    for y in years:
        tg = _nth_weekday(y, 11, 3, 4)
        rows += [(pd.Timestamp(y, 1, 1), "New Year's Day"), (_nth_weekday(y, 2, 6, 2), "Super Bowl"),
                 (_last_weekday(y, 5, 0), "Memorial Day"), (pd.Timestamp(y, 7, 4), "Independence Day"),
                 (_nth_weekday(y, 9, 0, 1), "Labor Day"), (pd.Timestamp(y, 10, 31), "Halloween"),
                 (tg, "Thanksgiving"), (tg + pd.Timedelta(days=1), "Black Friday"),
                 (pd.Timestamp(y, 12, 25), "Christmas")]
    df = pd.DataFrame(rows, columns=["holiday_date", "name"])
    df["region"] = "ALL"
    return df.sort_values("holiday_date").reset_index(drop=True)


def holiday_lifts_df():
    return pd.DataFrame([(h, c, v) for h, d in HOLIDAY_LIFT.items() for c, v in d.items()],
                        columns=["name", "category", "lift"])


def segment_frames(rng):
    seg = pd.DataFrame([(s[0], s[1], s[2]) for s in SEGMENTS], columns=["segment_id", "name", "customer_share"])
    cats = sorted({p[2] for p in _P})
    rows = []
    for c in cats:
        raw = np.array([s[2] * s[4].get(c, 1.0) for s in SEGMENTS])
        share = raw / raw.sum()
        mult = np.array([s[3] for s in SEGMENTS])
        mult = mult / (share * mult).sum()          # unit-weighted mean elasticity multiplier == 1
        rows += [(s[0], c, round(share[i], 4), round(mult[i], 4)) for i, s in enumerate(SEGMENTS)]
    seg_cat = pd.DataFrame(rows, columns=["segment_id", "category", "unit_share", "elasticity_multiplier"])
    mix = []
    base = np.array([s[2] for s in SEGMENTS])
    for sid, *_ in STORES:
        share = rng.dirichlet(base * 40)
        mix += [(sid, SEGMENTS[i][0], round(share[i], 4)) for i in range(len(SEGMENTS))]
    return seg, seg_cat, pd.DataFrame(mix, columns=["store_id", "segment_id", "share"])
