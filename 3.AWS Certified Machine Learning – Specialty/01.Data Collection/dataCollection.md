## Data collection

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

These are quantitative and the values are discrete. Categorical features are values that are associated with a group.

* Continuos Features

These are qualitative and the values can be infinity. Continuos features are values that are expressed as measurable number.