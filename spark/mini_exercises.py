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

# 1. Spark Dataframe Basics
#   i. Use the starter code above to create a pandas dataframe.
#   ii. Convert the pandas dataframe to a spark dataframe. 
#       From this point forward, do all of your work with the spark dataframe, not the pandas dataframe.

spark = pyspark.sql.SparkSession.builder.getOrCreate()

df = spark.createDataFrame(get_panda_df())
df

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
df = spark.createDataFrame(get_panda_df())
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
df = spark.createDataFrame(get_panda_df())

#   ii. Turn your dataframe into a table that can be queried with spark SQL. 
#       Name the table `my_df`
#       Answer the rest of the questions in this section with a spark query (`spark.sql`) against `my_df`.
#       After each step, view  the first 7 records of the dataframe.


