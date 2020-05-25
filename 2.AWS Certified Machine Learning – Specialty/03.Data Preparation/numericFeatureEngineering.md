# Numeric Engineering

Transforming numeric values in our datasets so they are easier to work with.

## Feature Scaling

Changes numeric values so all values are on the same scale:

Normalization

For a column with large values like price of homes, it will make lowest value to 0 and highest value to 1 and scale the in-between values accordingly. Disadvantage is it can throw off the outliers from normalization.

x = x - min(x) / max(x) - Min(x)

Standardization

Puts the average value to 0, and uses z-score for remainder values.

z = x - x_mean / x_std

## Binning

Changes numeric values into groups or buckets of similar values.

Quantile Binning aims to assign the same number of features to each bin.

Example: Age os employees. Bin them as: 0-30, 30-50, 50 and above. It can lead to uneven binning.

Quantile binning: Grouping in equal bins.

Example: 0-25, 26-35, 36 and above.

# Other Feature Engineering

1. Image feature engineering

# Handling missing value

Can be represented as null, NaN, NA, None, blank etc

a) Missing at Random

b) Missing Completely at Random

c) Missing not at Random

How to handle missing values?

Supervised learning, Mean, Median, Mode, Dropping rows

Replacing data is known as data imputation

# Feature Selection

Deciding what features keep and what to remove from dataset

Principle Component Analysis (PCA) unsupervised learning algorithms selects features.

# AWS Data Preparation Tools

AWS Glue

One stop solution for ETL.

Data Source (S3,DynamoDB,RDS,Redshift,DB on EC2) --> Crawler --> Data Catalog --> Job(Scala, Python to do feature selection or any other data preparation technique) --> Output to output-source (Athena, EMR, S3, Redshift)

SageMaker

Spin-up jupyter Notebook to prepare data. 

EMR

EMR fully managed eco-system.

Athena

Serverless service. Allows to run SQL query on data in S3.

Data Pipeline

Process and move data between different AWS compute services.