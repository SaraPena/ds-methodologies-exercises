import pandas as pd
import numpy as np
import pyspark

np.random.seed(13)


def get_panda_df():
    pandas_dataframe = pd.DataFrame(
        {
            "n": np.random.randn(20),
            "group": np.random.choice(list("xyz"), 20),
            "abool": np.random.choice([True,False], 20),
        }
    )
    return pandas_dataframe

def create_spark_df():
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(get_panda_df())
    return df

# 1. Spark Dataframe Basics
#   i. Use the starter code above to create a pandas dataframe.
#   ii. Convert the pandas dataframe to a spark dataframe. 
#       From this point forward, do all of your work with the spark dataframe, not the pandas dataframe.

df = create_spark_df()

#   iii. Show the first 3 rows of the dataframe.
df.show(3)

#   iv. Show the first 7 rows of the dataframe
df.show(7)

#   v. View a summary of the data using .describe.
df.describe().show()

#   vi. Use .select to create a new dataframe with just the 'n' and 'abool' columns. 
#       View the first 5 rows of this dataframe.

df.select('n', 'abool').show(5)

#   vii. Use .select to create a new dataframe with just the `group` and `abool` columns. 
#        View the first 5 rows of this dataframe.
df.select('group', 'abool').show(5)

#   viii. Use .select to create a new dataframe with the `group` column and the `abool` column renamed to `a_boolean_value`.
#         Show the first 3 rows. 
col1 = df.abool.alias('a_boolean_value')
df.select('group', col1).show(5)

#   ix. Use .select to create a new dataframe with the `group` column and `n` column renamed to `a_numeric_value`.
#       Show the first 6 rows of the dataframe.
col2 = df.n.alias('a_numeric_value')
df.select('group', col2).show(5)


# 2. Column Manipulation
#   i. Use the starter code above to re-create a spark dataframe. Store the spark dataframe in a variable named df
df = create_spark_df()
df

#   ii. Use .select to add 4 to the `n` column. Show the results.
df.select('n', df.n + 4).show()

#   iii. Subtract 5 from the n column and view the results.
df.select('n', df.n - 5).show()

#   iv. Multiply the n column by 2. 
#       View the results along with the original numbers.
df.select('n', df.n * 2).show()

#   v. Add a new column named `n2` that is the n value multiplied by -1. 
#      Show the first 4 rows of your dataframe. 
#      You should see the original `n` value as well as `n2`.
col1 = ((df.n*1).alias('n2'))
df.select('n', col1).show()

#   vi. Add a new column named `n3` that is the value of n squared.
#       Show the first 5 rows of your dataframe.
#       You should see both n, n2, and n3
col2 = (df.n**2).alias('n3')
df.select('n', col1, col2).show()

#   vii. What happens when you run the code below.
df.group + df.abool
# It only shows the object of the columns added together.

#   viii. What happens when you run the code below?
#         What is the difference between this and the previous code sample?
df.select(df.group + df.abool).show()
# Shows a java error.

#   ix. Try adding various other columns together. What are the results of combining the different data types?
# We can create columns with the `col` and `expr` functions from `pyspark.sql.functions` module. 

from pyspark.sql.functions import col, expr

df.show()
col('n')

avg_column = (col('n') + col('n'))/2
df.select(
    col('n'),
    avg_column.alias('avg_n')).show()

# 3. Spark SQL
#   i. Use the starter code above to recreate a spark dataframe.
df = create_spark_df()

#   ii. Turn your dataframe into a table that can be queried with spark SQL. 
#       Name the table `my_df`
#       Answer the rest of the questions in this section with a spark query (`spark.sql`) against `my_df`.
#       After each step, view  the first 7 records of the dataframe.

df.createOrReplaceTempView("my_df")
spark = pyspark.sql.SparkSession.builder.getOrCreate()


#   iii. Write a query that shows all of the columns from your dataframe.
spark.sql(
    """
    SELECT *
    FROM my_df
    """
    ).show(7)

#   iv. Write a query that shows just the `n` and `abool` columns from the dataframe.
spark.sql(
    """
    SELECT 'n', 'abool'
    FROM my_df
    """
).show(7)

#   v. Write a query that shows just the `n` and `group` columns. Rename the `group` column to `g`.
spark.sql(
    """
    SELECT n, group as g
    FROM my_df
    """
).show(7)

#   vi. Write a query that selects 'n' and creates two new columns: n2, the original n values halved, and n3: the original values minus 1.
spark.sql(
    """
    SELECT n, n/2 as n2, n-1 as n3
    FROM my_df
    """
).show(7)

#   vii. What happens if you make a SQL syntax error in your query?
spark.sql(
    """
    SELECT *,
    FROM my_df
    """
)

# 4. Type casting
#   i. Use the starter code above to re-create a spark dataframe.
df = create_spark_df()

#   ii. Use .printSchema to view the datatypes in your dataframe.
df.printSchema()

#   iii. Use .dtypes to view the datatypes in your dataframe.
df.dtypes

#   iv. What is the difference between the two code samples below?
df.abool.cast('int')
# Shows the object created

df.select(df.abool.cast('int')).show()
# performs an action of casting the values into int's and shows the values in the abool column.

#   v. Use `.select` and `.cast` to convert the abool column to an integer type. View the results.
df.select(df.abool.cast('int')).printSchema()

df.select(df.abool.cast('int')).show()

#   vi. Convert the `group` column to a integer data type, and view the results. What happens?
df.select(df.group.cast('int')).show()
# results are null
df.select('group').show()
# this group is string values.

#   vii. Convert the `n` column to a integer data type and view the results. What happens?
df.select(df.n.cast('int')).show()
df.select(df.n).show()
# the action df.n.cast('int') truncates the value to the number before the decimal.

#   viii. Convert the `abool` column to a string datatype, and view the results. What happens?
df.select(df.abool.cast('string')).show()
df.select(df.abool).show()

# 5. Built-in Functions:
#   i. Use the starter code above to re-create a spark dataframe.
df = create_spark_df()

#   ii. Import the necessary functions from pyspark.sql.functions.

from pyspark.sql.functions import concat, sum, avg, min






