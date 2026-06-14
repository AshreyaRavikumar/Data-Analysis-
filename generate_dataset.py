"""
Generate a realistic synthetic sales transactions dataset.
Includes intentional data quality issues for the cleaning phase.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
N = 2000

products = {
    "Laptop": {"category": "Electronics", "price_range": (800, 2000)},
    "Smartphone": {"category": "Electronics", "price_range": (300, 1200)},
    "Headphones": {"category": "Electronics", "price_range": (50, 350)},
    "Office Chair": {"category": "Furniture", "price_range": (150, 600)},
    "Desk": {"category": "Furniture", "price_range": (200, 800)},
    "Monitor": {"category": "Electronics", "price_range": (250, 900)},
    "Keyboard": {"category": "Accessories", "price_range": (30, 200)},
    "Mouse": {"category": "Accessories", "price_range": (20, 150)},
    "Backpack": {"category": "Accessories", "price_range": (40, 200)},
    "Notebook": {"category": "Stationery", "price_range": (5, 30)},
}

regions = ["North", "South", "East", "West", "Central"]
payment_methods = ["Credit Card", "Debit Card", "PayPal", "Cash", "Bank Transfer"]
customer_segments = ["Individual", "Corporate", "Startup", "Enterprise"]

start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)
date_range_days = (end_date - start_date).days

records = []
for i in range(1, N + 1):
    product = np.random.choice(list(products.keys()))
    info = products[product]
    price = round(np.random.uniform(*info["price_range"]), 2)
    quantity = np.random.randint(1, 6)
    discount_pct = np.random.choice(
        [0, 0, 0, 5, 10, 15, 20], p=[0.35, 0.1, 0.1, 0.15, 0.15, 0.1, 0.05]
    )
    revenue = round(price * quantity * (1 - discount_pct / 100), 2)

    transaction_date = start_date + timedelta(
        days=np.random.randint(0, date_range_days)
    )
    region = np.random.choice(regions, p=[0.25, 0.20, 0.20, 0.20, 0.15])
    payment = np.random.choice(payment_methods)
    segment = np.random.choice(customer_segments)
    rating = np.random.choice(
        [1, 2, 3, 4, 5, np.nan], p=[0.03, 0.07, 0.20, 0.40, 0.25, 0.05]
    )

    records.append(
        {
            "TransactionID": f"TXN{i:05d}",
            "Date": transaction_date.strftime("%Y-%m-%d"),
            "Product": product,
            "Category": info["category"],
            "UnitPrice": price,
            "Quantity": quantity,
            "DiscountPercent": discount_pct,
            "TotalRevenue": revenue,
            "Region": region,
            "PaymentMethod": payment,
            "CustomerSegment": segment,
            "CustomerRating": rating,
        }
    )

df = pd.DataFrame(records)

# Inject data quality issues
dup_indices = np.random.choice(N, size=40, replace=False)
dups = df.iloc[dup_indices].copy()
df = pd.concat([df, dups], ignore_index=True)

missing_price_idx = np.random.choice(len(df), size=30, replace=False)
df.loc[missing_price_idx, "UnitPrice"] = np.nan

missing_region_idx = np.random.choice(len(df), size=25, replace=False)
df.loc[missing_region_idx, "Region"] = ""

wrong_date_idx = np.random.choice(len(df), size=15, replace=False)
df.loc[wrong_date_idx, "Date"] = "2024/13/45"

neg_rev_idx = np.random.choice(len(df), size=10, replace=False)
df.loc[neg_rev_idx, "TotalRevenue"] = -999

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

output_path = "data/sales_transactions_raw.csv"
df.to_csv(output_path, index=False)
print(f"Raw dataset saved to {output_path}")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
