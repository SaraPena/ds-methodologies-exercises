from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    expr,
    col,
    to_timestamp,
    format_string,
    regexp_extract,
    datediff,
    current_timestamp,
    when,
    max,
    lit
)

def get_311_data():
    spark = SparkSession.builder.getOrCreate()
    print('[wrangle.py] reading case.csv')
    df = spark.read.csv('data/case.csv', header = True, inferSchema = True)
    return df.withColumnRenamed('SLA_due_date', 'case_due_date')

def handle_dtypes(df):
    print('[wrangle.py] handling data types')
    return (
        df.withColumn('case_closed', expr('case_closed == "YES"'))
        .withColumn('case_late', expr('case_late == "YES"'))
        .withColumn('council_district', col('council_district').cast('string'))
    )

def handle_dates(df):
    print('[wrangle.py] parsing dates')
    fmt = 'M/d/yy H:mm'
    return (
        df.withColumn('case_opened_date', to_timestamp('case_opened_date', fmt))
        .withColumn('case_closed_date', to_timestamp('case_closed_date', fmt))
        .withColumn('case_due_date', to_timestamp('case_due_date', fmt))
    )

def add_features(df):
    print('[wrangle.py] adding features')
    max_date = df.select(max('case_closed_date')).first()[0]
    return (
        df.withColumn('num_weeks_late', expr("num_days_late / 7 AS num_weeks_late"))
        .withColumn(
            "council_district",
            format_string("%03d", col('council_district').cast('int')),
        )
        .withColumn('zipcode', regexp_extract('request_address', r'\d+$',0))
        .withColumn('case_age', datediff(lit(max_date), 'case_opened_date'))
        .withColumn('days_to_closed', datediff('case_closed_date', 'case_opened_date'))
        .withColumn(
            "case_lifetime",
            when(expr('! case_closed'), col('case_age')).otherwise(
                col('days_to_closed')
            ),
        )
    )

def join_departments(df):
    print('[wrangle.py] joining departments')
    spark = SparkSession.builder.getOrCreate()
    dept = spark.read.csv('data/dept.csv', header = True, inferSchema = True)
    return (
        df.join(dept, 'dept_division', 'left')
        # drop all columns except for standardized name as it has much fewer unique values
        .drop(dept.dept_name)
        .withColumnRenamed('standardized_dept_name', 'department')
        # convert to a boolean
        .withColumn('dept_subject_to_SLA', col('dept_subject_to_SLA') == "YES")
    )

def wrangle_311():
    df = add_features(handle_dates(handle_dtypes(get_311_data())))
    return join_departments(df)

# case_df = wrangle_311()


