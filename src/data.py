import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window as W

# Seeds
seeds = [_ for _ in range(10)]
seed = seeds[1]

# Create Spark session
spark = SparkSession.builder \
    .appName("M5") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("Loading raw data...")
# Read all data files
calendar = spark.read.option('header', True).csv('../data/raw/m5-forecasting-accuracy/calendar.csv')
sell_prices = spark.read.option('header', True).csv('../data/raw/m5-forecasting-accuracy/sell_prices.csv')
sales_train = spark.read.option('header', True).csv('../data/raw/m5-forecasting-accuracy/sales_train_evaluation.csv')

# Extract metadata
states = sales_train.select('state_id').distinct().toPandas()
stores = sales_train.select('store_id').distinct().toPandas()
categories = sales_train.select('cat_id').distinct().toPandas()
departments = sales_train.select('dept_id').distinct().toPandas()
items = sales_train.select('item_id').distinct().toPandas()

summary = pd.DataFrame({
    'states': len(states),
    'stores': len(stores),
    'categories': len(categories),
    'departments': len(departments),
    'items': len(items)
}, index=[0])

print(f"Dataset size: {summary.to_dict('records')[0]}")

# Sample items and stores for manageable dataset
print(f"Sampling with seed {seed}...")
item_ids_sample = list(items.sample(50, random_state=seed)['item_id'])  # 50 items
store_ids_sample = list(stores.sample(5, random_state=seed)['store_id'])  # 5 stores

# Filter sales_train to sampled items/stores
sales_train_sample = sales_train.filter(
    F.col('item_id').isin(item_ids_sample) & 
    F.col('store_id').isin(store_ids_sample)
).cache()

print(f"Sampled {len(item_ids_sample)} items and {len(store_ids_sample)} stores")

# Fix calendar d column
calendar = calendar.withColumn('d', F.concat(F.lit('d_'), 
    F.row_number().over(W.orderBy('date'))))

# Unpivot to long format
ids = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

# Full dataset - unpivot (kept for reference, not used)
print("Converting full dataset to long format...")
long_sales_train = sales_train.unpivot(
    ids=ids, 
    values=None, 
    variableColumnName='d', 
    valueColumnName='units'
)

# Small sample for visualization - unpivot
print("Converting small sample to long format...")
long_sales_train_sample = sales_train_sample.unpivot(
    ids=ids, 
    values=None, 
    variableColumnName='d', 
    valueColumnName='units'
)

# Join with calendar and prices (full dataset)
print("Joining full dataset with calendar and prices...")
revenue = long_sales_train \
    .join(calendar, on='d') \
    .join(sell_prices, on=['item_id', 'store_id', 'wm_yr_wk']) \
    .withColumn('units', F.col('units').cast('int')) \
    .withColumn('sell_price', F.col('sell_price').cast('double'))

# Join with calendar and prices (small sample for visualization)
print("Joining small sample with calendar and prices...")
revenue_sample = long_sales_train_sample \
    .join(calendar, on='d') \
    .join(sell_prices, on=['item_id', 'store_id', 'wm_yr_wk']) \
    .withColumn('units', F.col('units').cast('int')) \
    .withColumn('sell_price', F.col('sell_price').cast('double')) \
    .cache()

print(f"Revenue sample created with {revenue_sample.count()} rows")

# Convert small sample to pandas for visualization
print("Creating pandas dataframes for visualization...")
items_stores_sample = revenue_sample.toPandas()

items_states_sample = revenue_sample \
    .groupBy('state_id', 'item_id', 'date') \
    .agg(F.sum('units').alias('sum(units)')) \
    .toPandas()

items_total = revenue_sample \
    .groupBy('item_id', 'date') \
    .agg(F.sum('units').alias('sum(units)')) \
    .toPandas()

dept_stores_sample = revenue_sample \
    .groupBy('dept_id', 'store_id', 'date') \
    .agg(F.sum('units').alias('sum(units)')) \
    .toPandas()

cat_stores_sample = revenue_sample \
    .groupBy('cat_id', 'store_id', 'date') \
    .agg(F.sum('units').alias('sum(units)')) \
    .toPandas()

# ============================================================================
# LARGER SAMPLE FOR ANALYSIS (10% of full dataset)
# ============================================================================
print("\n" + "="*60)
print("Creating 10% sample for demand analysis...")
print("="*60)

# Sample 10% of all items (keep all stores for each sampled item)
all_items = sales_train.select('item_id').distinct()
items_10pct = all_items.sample(fraction=0.1, seed=seed)
item_ids_10pct = [row.item_id for row in items_10pct.collect()]

print(f"Sampled {len(item_ids_10pct)} items (10% of {len(items)})")

# Filter to 10% sample
sales_train_10pct = sales_train.filter(F.col('item_id').isin(item_ids_10pct)).cache()

# Unpivot
long_sales_10pct = sales_train_10pct.unpivot(
    ids=ids,
    values=None,
    variableColumnName='d',
    valueColumnName='units'
)

# Join with calendar and prices
revenue_10pct = long_sales_10pct \
    .join(calendar, on='d') \
    .join(sell_prices, on=['item_id', 'store_id', 'wm_yr_wk']) \
    .withColumn('units', F.col('units').cast('int')) \
    .withColumn('sell_price', F.col('sell_price').cast('double')) \
    .cache()

print(f"10% sample has {revenue_10pct.count()} rows")

# Compute date ranges (first/last non-zero dates) on 10% sample
print("Computing date ranges on 10% sample...")
date_ranges = revenue_10pct \
    .filter(F.col('units') > 0) \
    .groupBy('item_id', 'store_id') \
    .agg(
        F.min('date').alias('first_sale_date'),
        F.max('date').alias('last_sale_date')
    )

first_dates = date_ranges.groupBy('first_sale_date').count().orderBy('first_sale_date').toPandas()
last_dates = date_ranges.groupBy('last_sale_date').count().orderBy('last_sale_date').toPandas()

# Convert 10% sample to pandas for analysis
print("Converting 10% sample to pandas...")
revenue_10pct_pd = revenue_10pct.toPandas()

# Create clean revenue (remove leading/trailing zeros)
print("Filtering leading/trailing zeros on 10% sample...")
date_ranges_pd = date_ranges.toPandas()

revenue_clean_pd = revenue_10pct_pd.merge(date_ranges_pd, on=['item_id', 'store_id'])
revenue_clean_pd = revenue_clean_pd[
    (revenue_clean_pd['date'] >= revenue_clean_pd['first_sale_date']) & 
    (revenue_clean_pd['date'] <= revenue_clean_pd['last_sale_date'])
].drop(['first_sale_date', 'last_sale_date'], axis=1)

print(f"Original rows (10% sample): {len(revenue_10pct_pd)}")
print(f"Clean rows: {len(revenue_clean_pd)}")

# Compute demand statistics (ADI and CV²)
print("Computing demand statistics on 10% sample...")
demand_stats = revenue_clean_pd.groupby(['item_id', 'store_id']).agg(
    total_days=('date', 'count'),
    demand_days=('units', lambda x: (x > 0).sum()),
    mean_demand=('units', 'mean'),
    std_demand=('units', 'std')
).reset_index()

demand_stats['adi'] = demand_stats['total_days'] / demand_stats['demand_days']
demand_stats['cv2'] = (demand_stats['std_demand'] / demand_stats['mean_demand']) ** 2

# Classify demand patterns
demand_stats['category'] = 'Smooth'
demand_stats.loc[(demand_stats['adi'] > 1.32) & (demand_stats['cv2'] <= 0.49), 'category'] = 'Intermittent'
demand_stats.loc[(demand_stats['adi'] <= 1.32) & (demand_stats['cv2'] > 0.49), 'category'] = 'Erratic'
demand_stats.loc[(demand_stats['adi'] > 1.32) & (demand_stats['cv2'] > 0.49), 'category'] = 'Lumpy'

print("\nDemand classification on 10% sample:")
print(demand_stats['category'].value_counts())

# Compute zero sequence statistics
print("\nComputing zero sequence statistics on 10% sample...")
zero_stats_list = []

for (item, store), group in revenue_clean_pd.groupby(['item_id', 'store_id']):
    group = group.sort_values('date').reset_index(drop=True)
    zeros = (group['units'] == 0).astype(int)
    
    # Find zero sequences
    zero_stats_list = []

for (item, store), group in revenue_clean_pd.groupby(['item_id', 'store_id']):
    group = group.sort_values('date').reset_index(drop=True)
    zeros = (group['units'] == 0).astype(int)
    
    # Find zero sequences
    zero_starts = (zeros.diff() == 1).fillna(False)
    zero_group = zero_starts.cumsum()
    
    sequences = group[zeros == 1].groupby(zero_group).size()
    
    if len(sequences) > 0:
        zero_stats_list.append({
            'item_id': item,
            'store_id': store,
            'avg_zero_seq': sequences.mean(),
            'max_zero_seq': sequences.max(),
            'num_zero_seqs': len(sequences),
            'zero_ratio': zeros.sum() / len(zeros),
            'zero_seq_dist': sequences.tolist()  # Keep full distribution
        })

zero_stats = pd.DataFrame(zero_stats_list)

print("Data loading complete!")
print(f"\nVisualization dataframes (small sample): items_stores_sample, items_states_sample, items_total, dept_stores_sample, cat_stores_sample")
print(f"Analysis dataframes (10% sample): revenue_10pct_pd, revenue_clean_pd")
print(f"Statistics (10% sample): demand_stats, zero_stats, first_dates, last_dates")