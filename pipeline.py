# pipeline.py

# ==========================================
# CELL 0: SETUP & STANDARDIZATION
# ==========================================
import pandas as pd
import os
import shutil

print(">>> CELL 0 STARTING: SETUP")

# 1. Handle Input File Conversion
if os.path.exists('raw.csv'):
    print("   Found raw.csv, converting to Input.csv...")
    shutil.copy('raw.csv', 'Input.csv')
elif os.path.exists('Input.xlsx'):
    print("   Found Input.xlsx, converting to Input.csv...")
    pd.read_excel('Input.xlsx').to_csv('Input.csv', index=False)
    
if not os.path.exists('Input.csv'):
    raise FileNotFoundError("CRITICAL: No Input file found (checked raw.csv, Input.xlsx, Input.csv)")

# 2. Check Live Inventory (Convert if necessary for Cell 2)
if os.path.exists('Live Inventory.xlsx') and not os.path.exists('Live_Inventory.csv'):
    print("   Converting Live Inventory.xlsx to CSV...")
    inv = pd.read_excel('Live Inventory.xlsx')
    inv.columns = inv.columns.str.strip().str.replace(' ', '_').str.lower()
    inv.to_csv('Live_Inventory.csv', index=False)

print("✅ CELL 0 COMPLETE")


# ==========================================
# CELL 1: ETL PROCESS
# ==========================================
import pandas as pd

print("\n>>> CELL 1 STARTING: ETL")

# 1. Load Input
df = pd.read_csv("Input.csv")

# 2. Select & Rename Columns (Enforcing snake_case manually here for core mapping)
# Note: Adjust the keys on the left if your raw CSV headers differ
df = df[[
    "Tracking Number", 
    "Channel Created Date", 
    "Products", 
    "Quantity", 
    "Shipping Address Pincode"
]]

df.columns = ["awb", "date", "skucode", "quantity", "pincode"]

# 3. Standardize Column Names (Global Rule)
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

# 4. Filter Pincodes
allowed_pincodes = [560037, 560048, 560103, 560066, 560017, 560093, 560038, 560075]
df = df[df["pincode"].isin(allowed_pincodes)]

# 5. Tag Pincode
df["pincode"] = "MRTH"

# 6. Date Formatting
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["hour"] = df["date"].dt.hour
df["date"] = df["date"].dt.strftime("%b %d, %Y")

# 7. Final Reorder & Export
df = df[["awb", "date", "skucode", "quantity", "hour", "pincode"]]
df.to_csv("Transformed_Output.csv", index=False)
print(f"✅ CELL 1 COMPLETE: Transformed_Output.csv created ({len(df)} rows)")


# ==========================================
# CELL 1.5: INTERMEDIATE SPLITTING
# ==========================================
import pandas as pd
from datetime import datetime, timedelta

print("\n>>> CELL 1.5 STARTING: SUB-DATASETS")

df = pd.read_csv("Transformed_Output.csv")
# Enforce naming convention
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

# Dynamic Date
yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%b %d, %Y")
print(f"   Target Yesterday: {yesterday_str}")

# A. Create ActualOrders_30Days.csv (History)
# Remove yesterday, Remove 'other' pincodes
mask_history = (df['date'] != yesterday_str) & (df['pincode'] != 'other')
df_history = df[mask_history].copy()
df_history.to_csv("ActualOrders_30Days.csv", index=False)

# B. Create Night.csv (Recent)
# Keep yesterday, Keep 14:00-23:00
df['hour'] = pd.to_numeric(df['hour'], errors='coerce')
mask_night = (df['date'] == yesterday_str) & (df['hour'].between(14, 23))
df_night = df[mask_night].copy()
df_night.to_csv("Night.csv", index=False)

print("✅ CELL 1.5 COMPLETE")


# ==========================================
# CELL 2: BALANCED TARGETED FORECASTING
# ==========================================
import pandas as pd
import numpy as np
import os

print("\n>>> CELL 2 STARTING: BALANCED FORECAST GENERATION")

# Files
ORDER_FILE = 'ActualOrders_30Days.csv'
NIGHT_FILE = 'Night.csv'
INVENTORY_FILE = 'Live_Inventory.csv'
OUT_MAIN = 'festival_forecast_inventory_optimized.csv'

# CONFIGURATION
OVER_FORECAST_BUFFER = 1.05 # 5% global safety drift

# Load Data
orders = pd.read_csv(ORDER_FILE)
night_orders = pd.read_csv(NIGHT_FILE) if os.path.exists(NIGHT_FILE) else pd.DataFrame()
inventory = pd.read_csv(INVENTORY_FILE) if os.path.exists(INVENTORY_FILE) else pd.DataFrame(columns=['skucode', 'count'])

# Standardize Columns
for d in [orders, night_orders, inventory]:
    if not d.empty:
        d.columns = d.columns.str.strip().str.replace(' ', '_').str.lower()
        if 'count' in d.columns: d.rename(columns={'count': 'current_inventory'}, inplace=True)
        if 'qty' in d.columns: d.rename(columns={'qty': 'quantity'}, inplace=True)

# 1. PREPARE DAILY DATA
orders['date'] = pd.to_datetime(orders['date'])
daily_pivot = orders.groupby(['skucode', 'date'])['quantity'].sum().unstack().fillna(0)

# 2. CALCULATE METRICS
weights = np.array([0.05, 0.05, 0.10, 0.10, 0.20, 0.20, 0.30])
def calculate_wma(row):
    last_7 = row.tail(7).values
    return np.dot(last_7, weights) if len(last_7) == 7 else row.tail(7).mean()

def calculate_median_mix(row):
    return (0.8 * row.tail(7).median()) + (0.2 * row.tail(7).mean())

forecast_df = pd.DataFrame(index=daily_pivot.index)
forecast_df['wma_forecast'] = daily_pivot.apply(calculate_wma, axis=1)
forecast_df['median_mix_forecast'] = daily_pivot.apply(calculate_median_mix, axis=1)
forecast_df = forecast_df.reset_index()

# 3. ASSIGN PATTERNS
def assign_pattern(val):
    if val >= 50: return 'HIGH_VELOCITY'
    elif val >= 10: return 'MEDIUM_VELOCITY'
    elif val >= 2: return 'LOW_VELOCITY'
    elif val > 0: return 'VERY_LOW_VELOCITY'
    else: return 'ZERO_DEMAND'

forecast_df['pattern'] = forecast_df['wma_forecast'].apply(assign_pattern)

# 4. SELECT BASE FORECAST
forecast_df['base_forecast'] = np.where(
    forecast_df['pattern'].isin(['HIGH_VELOCITY', 'MEDIUM_VELOCITY']),
    forecast_df['median_mix_forecast'], 
    forecast_df['wma_forecast']
)

# 5. BALANCED SAFETY LOGIC
def apply_safety(row):
    p = row['pattern']
    f = row['base_forecast'] * OVER_FORECAST_BUFFER
    
    # High/Med: Stable buffers
    if p == 'HIGH_VELOCITY': return f * 1
    elif p == 'MEDIUM_VELOCITY': return f * 0.9
    
    # Low/Very Low: Moderate over-forecasting (30% buffer + 1 unit floor)
    # This prevents negative values without creating 15-unit differences.
    elif p == 'LOW_VELOCITY': return (f * 1.15) + 1 
    elif p == 'VERY_LOW_VELOCITY': return (f * 1.20) + 1
    return f

forecast_df['demand_forecast'] = forecast_df.apply(apply_safety, axis=1).apply(np.ceil)

# 6. RESTOCK CALCULATION
if not inventory.empty:
    inv_clean = inventory.groupby('skucode')['current_inventory'].sum().reset_index()
    forecast_df = forecast_df.merge(inv_clean, on='skucode', how='left').fillna(0)
else: forecast_df['current_inventory'] = 0

forecast_df['forecast_orders_final'] = (forecast_df['demand_forecast'] - forecast_df['current_inventory']).clip(lower=0)

# EXPORT
forecast_df['hub_final'] = 'MRTH'
cols = ['hub_final', 'skucode', 'pattern', 'current_inventory', 'demand_forecast', 'forecast_orders_final']
forecast_df[cols].to_csv(OUT_MAIN, index=False)
print(f"✅ CELL 2 COMPLETE: Balanced safety logic applied.")


# ==========================================
# CELL 2.5: MORNING PERFORMANCE
# ==========================================
import pandas as pd
from datetime import datetime, timedelta

print("\n>>> CELL 2.5 STARTING: ACTUALS CALCULATION")

yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%b %d, %Y")

# 1. Load Transformed Output for Actuals
df_trans = pd.read_csv("Transformed_Output.csv")
df_trans.columns = df_trans.columns.str.strip().str.replace(' ', '_').str.lower()

# 2. Filter: MRTH + Yesterday + Hours 0-13
mask_actuals = (
    (df_trans['pincode'] == 'MRTH') &
    (df_trans['date'] == yesterday_str) &
    (df_trans['hour'].between(0, 13))
)
actuals_df = df_trans[mask_actuals].groupby('skucode')['quantity'].sum().reset_index()
actuals_df.rename(columns={'quantity': 'actual_orders'}, inplace=True)
actuals_df.to_csv("actual_orders.csv", index=False)

# 3. Process Morning Forecast
forecast_file = "festival_forecast_inventory_optimized.csv"
df_forecast = pd.read_csv(forecast_file) # CSV now
df_forecast.columns = df_forecast.columns.str.strip().str.replace(' ', '_').str.lower()

# Merge
df_morning = df_forecast.merge(actuals_df, on='skucode', how='left')
df_morning['actual_orders'] = df_morning['actual_orders'].fillna(0)

# Save Morning_Forecast.csv
cols = ['hub_final', 'skucode', 'forecast_orders_final', 'actual_orders']
# Ensure columns exist
for c in cols:
    if c not in df_morning.columns: df_morning[c] = 0
df_morning[cols].to_csv("Morning_Forecast.csv", index=False)

# 4. Generate ForecastVsActual_Afternoon.csv
df_fva = df_forecast[['skucode', 'demand_forecast']].copy()
df_fva.rename(columns={'demand_forecast': 'forecast'}, inplace=True)

# Merge Actuals (Orders till 2pm)
df_fva = df_fva.merge(actuals_df, on='skucode', how='left')
df_fva.rename(columns={'actual_orders': 'orders_till_2pm'}, inplace=True)
df_fva['orders_till_2pm'] = df_fva['orders_till_2pm'].fillna(0)

# Calculations
df_fva['opening_inventory'] = 0
df_fva['morning_inbound'] = (df_fva['forecast'] - df_fva['opening_inventory']).clip(lower=0)
df_fva['live_inventory'] = (df_fva['opening_inventory'] + df_fva['morning_inbound'] - df_fva['orders_till_2pm']).clip(lower=0)

df_fva.to_csv("ForecastVsActual_Afternoon.csv", index=False)
print("✅ CELL 2.5 COMPLETE")


# ==========================================
# CELL 3: AFTERNOON ANALYSIS (WINDOWED 0-19)
# ==========================================
import pandas as pd
import numpy as np

print("\n>>> CELL 3 STARTING: SHORTAGE ANALYSIS (0-19 WINDOW)")

INPUT_FILE = 'ForecastVsActual_Afternoon.csv'
OUTPUT_FILE = 'Afternoon_Forecast.csv'

df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

# Config - Target window ends at hour 20 (covering up to 19:59)
CURRENT_HOUR = 14
BUSINESS_CLOSE = 20 
hours_remaining = BUSINESS_CLOSE - CURRENT_HOUR
hours_elapsed = CURRENT_HOUR - 0

# 1. Velocity-based projection
df['velocity'] = df['orders_till_2pm'] / hours_elapsed
df['afternoon_demand_0_19'] = (df['velocity'] * hours_remaining).apply(np.ceil)

# 2. INVENTORY-AWARE TOP-UP
# Only send if projected demand for the window exceeds current shelf stock
df['net_quantity_to_send'] = (df['afternoon_demand_0_19'] - df['live_inventory']).clip(lower=0)

# 3. Simple Backlog Fix
df['unfilled_backlog'] = (df['orders_till_2pm'] - (df['forecast'] + df['opening_inventory'])).clip(lower=0)
df['net_quantity_to_send'] = (df['net_quantity_to_send'] + df['unfilled_backlog']).apply(np.ceil)

df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ CELL 3 COMPLETE: Afternoon dispatch limited to 0-19 window.")


# ==========================================
# CELL 3.5: FINAL CLEANUP & SYNCHRONIZED COMPARISON (0-19 WINDOW)
# ==========================================
import pandas as pd
import numpy as np

print("\n>>> CELL 3.5 STARTING: FINAL DISPATCH LIST PREPARATION")

# 1. Load the Afternoon Forecast
df = pd.read_csv("Afternoon_Forecast.csv")
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

# 2. TOTAL SUPPLY CALCULATION (Supply available for the window)
df['opening_inventory'] = pd.to_numeric(df['opening_inventory'], errors='coerce').fillna(0).astype(int)
df['morning_sent'] = pd.to_numeric(df['forecast'], errors='coerce').fillna(0).astype(int)
df['afternoon_sent'] = pd.to_numeric(df['net_quantity_to_send'], errors='coerce').fillna(0).astype(int)

# Total Forecast (Supply) = What was there + What we added
df['total_forecast'] = df['opening_inventory'] + df['morning_sent'] + df['afternoon_sent']

# 3. MERGE SKU TYPE (PATTERN)
try:
    df_p = pd.read_csv("festival_forecast_inventory_optimized.csv")
    df_p.columns = df_p.columns.str.strip().str.replace(' ', '_').str.lower()
    df = df.merge(df_p[['skucode', 'pattern']], on='skucode', how='left').rename(columns={'pattern': 'sku_type'})
except:
    df['sku_type'] = 'UNKNOWN'

# 4. MERGE ACTUAL ORDERS (STRICT 0-19 WINDOW)
try:
    df_t = pd.read_csv("Transformed_Output.csv")
    df_t.columns = df_t.columns.str.strip().str.replace(' ', '_').str.lower()
    df_t['date_dt'] = pd.to_datetime(df_t['date'], errors='coerce')
    latest = df_t['date_dt'].max()
    
    # Filter for Latest Date, MRTH, and strictly Hour 0-19
    mask = (df_t['date_dt'] == latest) & (df_t['pincode'].str.upper() == 'MRTH') & (df_t['hour'].between(0, 19))
    actuals = df_t[mask].groupby('skucode')['quantity'].sum().reset_index().rename(columns={'quantity': 'actual_orders'})
    
    df = df.merge(actuals, on='skucode', how='left').fillna(0)
    df['actual_orders'] = df['actual_orders'].astype(int)
except:
    df['actual_orders'] = 0

# 5. CALCULATE DIFFERENCE
# Difference = Available Supply - Sold in 0-19 Window
df['difference'] = df['total_forecast'] - df['actual_orders']
df['Closing Inventory'] = df['morning_sent'] + df['afternoon_sent'] - df['actual_orders']

# 6. REORDER & FINAL EXPORT
# Put sku_type right after skucode
cols = df.columns.tolist()
if 'sku_type' in cols:
    cols.remove('sku_type')
    skucode_idx = cols.index('skucode')
    cols.insert(skucode_idx + 1, 'sku_type')

# Clean technical columns for final report
report_cols = ['skucode', 'sku_type', 'opening_inventory', 'morning_sent', 'afternoon_sent', 'total_forecast', 'actual_orders', 'difference','Closing Inventory']
df[report_cols].to_csv("Final_Dispatch_List.csv", index=False)

print(f"✅ PIPELINE FINISHED: Final_Dispatch_List.csv generated with balanced comparison.")


# ==========================================
# CELL 4: PREPARE LIVE INVENTORY FOR TOMORROW
# ==========================================
import pandas as pd
import os

print("\n>>> CELL 4 STARTING: UPDATING LIVE INVENTORY")

# 1. Define file paths
# We update the CSV used as the starting point for tomorrow's run
INV_FILE = 'Live_Inventory.csv'
DISPATCH_FILE = 'Final_Dispatch_List.csv'

if not os.path.exists(INV_FILE) or not os.path.exists(DISPATCH_FILE):
    print("⚠️ Error: Required files (Live_Inventory.csv or Final_Dispatch_List.csv) not found.")
else:
    # 2. Load the data
    df_inv = pd.read_csv(INV_FILE)
    df_dispatch = pd.read_csv(DISPATCH_FILE)

    # Standardize headers to ensure matching works
    df_inv.columns = df_inv.columns.str.strip().str.replace(' ', '_').str.lower()
    df_dispatch.columns = df_dispatch.columns.str.strip().str.replace(' ', '_').str.lower()

    # Identify the 'Count' column name (Python likely normalized it to 'count')
    # If the column name in your original file was "Count", it is now "count"
    count_col = 'count' if 'count' in df_inv.columns else 'current_inventory'

    # 3. Perform the "VLOOKUP" Logic
    # We merge the Dispatch results into the Inventory list based on 'skucode'.
    # We take the 'difference' column (Closing Stock = Opening + Sent - Sold).
    df_updated = df_inv.merge(df_dispatch[['skucode', 'difference']], on='skucode', how='left')

    # Update the count: 
    # We use the 'difference' value if found. If a SKU wasn't in the list, 
    # we keep its original value (fillna).
    df_updated[count_col] = df_updated['difference'].fillna(df_updated[count_col]).astype(int)

    # Remove the temporary 'difference' column used for the merge
    df_updated.drop(columns=['difference'], inplace=True)

    # 4. Save the updated file back to Live_Inventory.csv
    # This file will be the "starting inventory" when the script runs tomorrow morning.
    df_updated.to_csv(INV_FILE, index=False)
    
    print(f"✅ CELL 4 COMPLETE: {INV_FILE} updated successfully with tomorrow's starting stock.")
    print(f"📈 Pipeline finished. The system is ready for the next day's run.")