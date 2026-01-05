import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window as W
from pyspark.sql import types as T

# create Spark session
spark = SparkSession.builder \
    .appName("M5") \
    .getOrCreate()

# read all data files
calendar = spark.read.option('header', True) \
            .csv('./data/m5-forecasting-accuracy/calendar.csv', )
sell_prices = spark.read.option('header', True) \
                .csv('./data/m5-forecasting-accuracy/sell_prices.csv')
sales_train = spark.read.option('header', True) \
                    .csv('./data/m5-forecasting-accuracy/sales_train_evaluation.csv')
sales_test = pd.read_csv('./data/m5-forecasting-accuracy/sales_test_evaluation.csv')
weights = pd.read_csv('./data/m5-forecasting-accuracy/weights_evaluation.csv')

# extract all columns
states = sales_train.select('state_id').distinct().toPandas()
stores = sales_train.select('store_id').distinct().toPandas()
categories = sales_train.select('cat_id').distinct().toPandas()
departments = sales_train.select('dept_id').distinct().toPandas()
items = sales_train.select('item_id').distinct().toPandas()

# highest summary
summary = pd.DataFrame({
    'states': len(states),
    'stores': len(stores),
    'categories': len(categories),
    'departments': len(departments),
    'items': len(items)
}, index=[0])

calendar = calendar.withColumn('d', F.concat(F.lit('d_'), F.row_number() \
                                        .over(W.orderBy('date'))))

ids = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
long_sales_train = sales_train \
                        .unpivot(ids=ids, values=None, variableColumnName='d', valueColumnName='units')

revenue = long_sales_train \
                    .join(calendar, on='d') \
                    .join(sell_prices, on=['item_id', 'store_id', 'wm_yr_wk']) \
                    .withColumn('units', F.col('units').cast('int')) \
                    .withColumn('sell_price', F.col('sell_price').cast('double'))

