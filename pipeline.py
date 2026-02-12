import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime, timedelta

print(">>> STARTING AUTOMATED PIPELINE (PRECISION + AGGRESSIVE MODES)")

# ==========================================
# 1. SETUP & ETL
# ==========================================
# 1.1 Load Raw Data
if os.path.exists('raw.csv'):
    shutil.copy('raw.csv', 'Input.csv')

if not os.path.exists('Input.csv'):
    # Create dummy if missing (for testing) or raise error
    raise FileNotFoundError("CRITICAL: Input.csv or raw.csv not found.")

df = pd.read_csv("Input.csv")
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

# 1.2 Map Columns
# Adjust these keys if your raw CSV headers change!
df = df[["tracking_number", "channel_created_date", "products", "quantity", "shipping_address_pincode"]]
df.columns = ["awb", "date", "skucode", "quantity", "pincode"]

# 1.3 Filter for Hub (MRTH)
allowed_pincodes = [560037, 560048, 560103, 560066, 560017, 560093, 560038, 560075]
df = df[df["pincode"].isin(allowed_pincodes)]
df["pincode"] = "MRTH"

# 1.4 Date & Hour
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["hour"] = df["date"].dt.hour
df["date_str"] = df["date"].dt.strftime("%b %d, %Y")

# Export Master Data
df.to_csv("Transformed_Output.csv", index=False)

# ==========================================
# 2. SUB-DATASET CREATION
# ==========================================
# We define "Yesterday" dynamically based on the latest date in your data
# This prevents timezone errors on GitHub servers
latest_date_dt = df["date"].max()
# If data is live, latest date is Today. So Yesterday is Today - 1.
yesterday_dt = latest_date_dt - timedelta(days=1)
yesterday_str = yesterday_dt.strftime("%b %d, %Y")

print(f"   Targeting 'Yesterday' as: {yesterday_str}")

# History (Exclude Yesterday)
mask_history = (df['date_str'] != yesterday_str)
df[mask_history].to_csv("ActualOrders_30Days.csv", index=False)

# Night Orders (Yesterday 14:00 - 23:00)
mask_night = (df['date_str'] == yesterday_str) & (df['hour'].between(14, 23))
df[mask_night].to_csv("Night.csv", index=False)

# ==========================================
# 3. FORECASTING ENGINE
# ==========================================
orders = pd.read_csv('ActualOrders_30Days.csv')
night_orders = pd.read_csv('Night.csv') if os.path.exists('Night.csv') else pd.DataFrame()
inventory = pd.read_csv('Live_Inventory.csv')

# Standardize Columns
for d in [orders, night_orders, inventory]:
    if not d.empty:
        d.columns = d.columns.str.strip().str.replace(' ', '_').str.lower()
        if 'count' in d.columns: d.rename(columns={'count': 'current_inventory'}, inplace=True)
        if 'qty' in d.columns: d.rename(columns={'qty': 'quantity'}, inplace=True)

daily_pivot = orders.groupby(['skucode', 'date_str'])['quantity'].sum().unstack().fillna(0)

# 3.1 Metrics Calculation
weights = np.array([0.05, 0.05, 0.10, 0.10, 0.20, 0.20, 0.30])

def calculate_wma(row):
    data = row.tail(7).values
    return np.dot(data, weights) if len(data) == 7 else row.tail(7).mean()

forecast_df = pd.DataFrame(index=daily_pivot.index)
forecast_df['wma_forecast'] = daily_pivot.apply(calculate_wma, axis=1)
# Median Mix: 80% Median, 20% Mean (Spike Resistant)
forecast_df['median_mix'] = (0.8 * daily_pivot.tail(7).median(axis=1)) + (0.2 * daily_pivot.tail(7).mean(axis=1))
forecast_df = forecast_df.reset_index()

# 3.2 Pattern Assignment
def assign_pattern(val):
    if val >= 50: return 'HIGH_VELOCITY'
    elif val >= 10: return 'MEDIUM_VELOCITY'
    elif val >= 2: return 'LOW_VELOCITY'
    return 'VERY_LOW_VELOCITY'

forecast_df['pattern'] = forecast_df['wma_forecast'].apply(assign_pattern)

# 3.3 TARGETED SAFETY LOGIC (The Fix)
def apply_safety(row):
    p = row['pattern']
    
    # CASE A: PRECISION (Medium/High)
    # Use strict Median Mix. NO Multipliers. NO Buffers.
    if p in ['HIGH_VELOCITY', 'MEDIUM_VELOCITY']:
        return row['median_mix']
    
    # CASE B: AGGRESSIVE (Low/Very Low)
    # Use WMA. 3.0x Multiplier. +3 Unit Flat Floor.
    else:
        return (row['wma_forecast'] * 3.0) + 3

forecast_df['demand_forecast'] = forecast_df.apply(apply_safety, axis=1).apply(np.ceil)

# 3.4 Night Orders Integration
if not night_orders.empty:
    night_agg = night_orders.groupby('skucode')['quantity'].sum().reset_index().rename(columns={'quantity': 'night_qty'})
    forecast_df = forecast_df.merge(night_agg, on='skucode', how='left').fillna(0)
    # Add 30% of night orders to the forecast
    forecast_df['demand_forecast'] = (forecast_df['demand_forecast'] + (forecast_df['night_qty'] * 0.3)).apply(np.ceil)

# 3.5 Restock Calculation
if not inventory.empty:
    inv_clean = inventory.groupby('skucode')['current_inventory'].sum().reset_index()
    forecast_df = forecast_df.merge(inv_clean, on='skucode', how='left').fillna(0)
else:
    forecast_df['current_inventory'] = 0

forecast_df['forecast_orders_final'] = (forecast_df['demand_forecast'] - forecast_df['current_inventory']).clip(lower=0)
forecast_df.to_csv('festival_forecast_inventory_optimized.csv', index=False)

# ==========================================
# 4. DISPATCH LIST & 0-19 ACTUALS COMPARISON
# ==========================================
# Prepare Base Data
df_final = forecast_df[['skucode', 'pattern', 'current_inventory', 'forecast_orders_final']].copy()
df_final.rename(columns={'current_inventory': 'opening_inventory', 'forecast_orders_final': 'morning_sent'}, inplace=True)
df_final['afternoon_sent'] = 0 # Default for morning run

# Total Supply = Opening + Sent
df_final['total_forecast'] = df_final['opening_inventory'] + df_final['morning_sent']

# Load Actuals (Strict 0-19 Window from Yesterday)
mask_actuals = (df['date_str'] == yesterday_str) & (df['hour'].between(0, 19))
actuals = df[mask_actuals].groupby('skucode')['quantity'].sum().reset_index().rename(columns={'quantity': 'actual_orders'})

# Merge & Calculate Difference
df_final = df_final.merge(actuals, on='skucode', how='left').fillna(0)
df_final['difference'] = df_final['total_forecast'] - df_final['actual_orders']

df_final.to_csv("Final_Dispatch_List.csv", index=False)

# ==========================================
# 5. INVENTORY UPDATE (For Tomorrow)
# ==========================================
# VLOOKUP Logic: Update 'count' with 'difference'
inventory_update = inventory[['skucode', 'current_inventory']].copy()
inventory_update = inventory_update.merge(df_final[['skucode', 'difference']], on='skucode', how='left')

# Use 'difference' if available, else keep old 'current_inventory'
inventory_update['new_count'] = inventory_update['difference'].fillna(inventory_update['current_inventory']).astype(int)

# Save for next run
inventory_update[['skucode', 'new_count']].rename(columns={'new_count': 'count'}).to_csv('Live_Inventory.csv', index=False)

print("✅ PIPELINE COMPLETE: Forecast generated, List exported, Inventory updated.")