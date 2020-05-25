# Data collection

Machine learning cycle:

Fetch data -> Clean Data -> Prepare Data -> Train Model -> Evaluate Model -> Deploy into Production -> Monitor and Evaluate </br>
|------ Generate Example Data ---------|    |----- Train the model -----|    |------------- Deploy the model -------------|

## What is a good dataset on which you are planning to work and build model ?

You should atleast have 10 times as many data points as the total umber of features.

| Traits of Good data | Traits of Bad data | Why |
|---------------------|--------------------|-----|
| Large dataset | Parse data set (less than 100 rows) | Generally more the data the better the predictions after applying modelling |
| Precise attribute types, feature rich | Useless attributes, not needed for solving problem at hand | Models needs to train on important feature |
| Complete fields, no missing values | Missing values, null fields | Models can skew results when data points are missing |
| Values are consistent | Inconsistent values | Models like clean and consistent data |
| Solid distribution of outcomes | Lots of positive outcomes, few negative outcomes | Models cannot learn with skewed distribution of outcomes |
| Fair sampling | Biased sampling | Models will kew results with biased data |

## Structured Data

Structured data has a defined schema and a schema is the information needed to interpret the data, including the attribute names and their assigned data types.

## Unstructured data

Unstructured data has no defined schema or structural properties. Makes up majority of data collected.

## Semi structured data

Semi structured data is to unstructured for relational data, but has some organizational structure. Usually in the form of CSV, JSON or XML.

| Database | Traditional Relational Database. <br> Transactional <br> Strict defined schema |
|----------|----------------------------------------------------------------------------------|
| Data Warehouse | Different data types are stored at one place. Data is processed before storing in the data warehouse. <br> Processing done on data import (schema-on-write). Data is stored with user in mind. <br> Ready to use with BI tools (query and analysis)|
| Data Lake | Data is dumped at one place without any prior processing. <br> Processing is done on export (schema-on-read) <br> Many different sources and formats. <br> Raw data may not be ready for use. |

## Labelled Data

Labeled data is a group of samples that have been tagged with one or more labels. Supervised learning uses labelled data. Labelled dataset will have a target variable. Examples are Spam/not Spam emails, Fraudulent/Non-fraudulent CC transaction


## Unlabelled Data

Unlabeled data is a designation for pieces of data that have not been tagged with labels identifying characteristics, properties or classifications. A dataset with no target variable is an example of unlabelled dataset. Unsupervised learning uses unlabelled data. Examples are log files, Tweets, customer information.

## Features in a data

* Categorical Features

These are quantitative and the values are discrete. Categorical features are values that are associated with a group. Examples are Yes/No values, Dog Breed, Spam feature.

* Continuos Features

These are qualitative and the values are infinite. Continuos features are values that are expressed as measurable number. Examples are Sales of a company.

## Text Data (Corpus Data)

These are datasets collected from text. Used in Natural Language Processing (NLP), speech recognition, text to speech and more. Corpus data is same thing as text data, it is referred to the dataset collected from text.

## Ground Truth

Ground truth datasets refers to factual data that has been observed or measured. This data has successfully been labeled and can be trusted as "truth" data.

<b>Amazon SageMaker Ground Truth</b> is a a tool that helps build ground truth datasets by allowing different types tagging/labelling processes.

Allows to easily create labeled data.

## Image data

Datasets with tagged images.

## Time series data

Dataset that captures changes over time.

# AWS Data Stores

It will explain different AWS services where we can get our data in AWS.

1. S3

Core AWS ML services can directly integrate with S3. We can save output to S3.

    * Files can be 0 bytes to 5 TB
    * Unlimited storage
    * Files are stored into buckets
    * Universal namespace

How to upload data to S3?

    * Just upload via AWS console, SDK or CLI.
    * AWS RDS (Amazon Aurora, MySQL, MariaDB, PostgreSQL, Oracle, SQLServer)
    * AWS DynamoDB (good for schemaless, semi-structured or no structure)
    * AWS Redshift (Fully managed petabyte data ware house service). Another important tool in RedShift is RedShift Spectrum, it allows you to query your RedShift cluster that has S3 source.

    eg. S3 -> Redshift Spectrum -> QuickSight

    * Timestream is fully managed time series database service.

    Time series data -> Timestream -> SQL Client/BI/Analytical tool

    * DocumentDB Place to migrate mongoDB data.

# AWS Migration Tools

(Exam question: If you have data in one tool and you want to migrate to another tool, what is the best AWS service)

Get data into S3 : different data migration tools

1. Data Pipeline

It allows you to process and move data between different AWS compute service and storage services. And it also allows you to transfer data from on-premise data store to AWS.

DynamoDB ------> Data pipeline -----
                                    |
RDS      ------> Data Pipeline --------> S3
                                    |
RedShift ------> Data pipeline -----

Data Pipeline have different activity objects that allows us to copy data using few different activity objects

Objects:

Copyactivity
EmrActivity
HadoopActivity
HiveActivity
HiveCopyActivity
PigActivity
RedShiftCopyActivity
ShellCommandActivity
SQLActivity (allows us to specify sql query)

Data pipeline can also be used for data transform.

2. DMS (Database Migration Service)

This service is used to transfer data between different database platform. Generally used between transfer data between 2 different relational databases, but you can transfer to S3.

On-Premise Database ------> DMS -----
                                     |
Database on EC2     ------> DMS ------->  S3
                                      |
Database on RDS     ------> DMS -------

3. AWS Glue

Its a fully managed ETL service. It can take data from any of the AWS service like S3, dynamoDB etc, use crawler and output the data to any other AWs services like Athena, EMR etc.

S3              ------                 --->    Athena
                      |                |
DynamoDB         ------                --->    EMR
                      |                |
RDS              --------->  AWS Glue -------> S3
                      |                |
Redshift         ------                --->    Redshift
                      |
Database on EC2  ------

We can use AWS glue to load data from one data source to another. We can change the output format to any of the format like csv, json, xml, parquet etc.

For the exam, Know which data migration service you will based on the give case scenario. Few example give below:

| Data source | Migration tool | Why |
|-------------|----------------|-----|
| PostgreSQL RDS instance with training data | AWS Pipeline | Specify SQLActivity query and places output into S3 |
| Unstructured log files to S3 | AWS Glue | Create custom classifier and output results into S3 |
| Clustered RedShift data | AWS Data Pipeline / AWS Glue | Use the unload command to return results of a query to CSV file in S3. </b> Create Data catalog describing data and load it into S3. |
| On-premise MySQL instance with training data. | AWS DMS | DMS can load data in CSV format onto S3 |

# AWS Helper Tools

Get data into AWS and help store in AWS. It does not fit in Data store and Data Migration Service.

## EMR (Elastic Map Reduce)

We can use it petabytes of data among distributed file system.

Fully managed hadoop cluster eco-system runs on multiple EC2. You can run many distributed workload (like Spark, Presto, MAHOUT, Hive, Jupyter Notebooks, TensorFlow, Hadoop HDFS, mxnet ML framework) between different EC2 instances.

We can use it to store mass amount of files.

## Athena

A serverless platform that allows to run query on S3.

Use AWS glue, open AWs Glue data catalog and we can use AWS Athena to run query on data stored in S3. Save output in csv file or save it in S3.

What is the difference between Redshift spectrum and Athena ?

| Redshift Spectrum | Athena |
|-------------------|--------|
| Query S3 data. <b>Must have Redshift Cluster. MAde for existing RedShift customers.</b> | Query S3 data. <b>No need for Redshift cluster. New customers quickly want to query S3 data</b> |

