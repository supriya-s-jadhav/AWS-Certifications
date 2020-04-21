# AWS Free Digital training notes

## 1. ML Building Blocks : Services & Terminology

All major frameworks:
    - Apache MXNex
    - Caffe & Caffe 2
    - Tensorflow

Fully managed path using own data:
    - Apache Spark on Amazon EMR
    - SparkML

Want to add intelligence to their application using API call:
    - Amazon Recognition (Image recognition)
    - Amazon Polly (Text to speech across many different languages)
    - Amazon Lex (Conversational interfaces, same tech as alexa)

Amazon S3 used as Data Lake
    - Data Analytics: Amazon Athena, Amazon Redshift, Amazon Redshift Spectrum

Common ML Terminology through ML Process: Training, Model, Prediction.

1. Business Problem
2. ML problem
3. Ask question like what algorithm we need to use etc.
    * Supervised Algorithm (Classification, Regression)
    * Unsupervised Learning (Clustering, Dimensionality reduction)
    * Reinforcement Learning
ML Problem definition: Key elements are Observations, Labels, Features
3. Develop Data set (Data collection/ Data integration)

Data collection services: Amazon S3, DynamoDB, Redshift, web pages

Structured, Semi-structured, Unstructured

4. Data Preparation
    * Data cleaning: Handling missing values and outliers.
        - Introduce new indicator variable to represent missing values
        - Remove the rows of missing value
        - Imputation
    * Impute missing values: Replace missing values with mean, median
    * Shuffle training data
        - Results in better model performance for certain algorithms
        - Minimizes the risk of cross validation data under representing the model data not learning from all type pf data
    * Test-validation-Train data
        - Train data 70%; Validation data: 10%; Test data: 20%
    * cross validation: 3 different types
        - Validation
        - LOOCV (Leave one out cross validation)
        - k-fold

5. Data visualization and Analysis
    * Types of visualization and analysis: Statistics, Scatter-plots, Histogram
    * Feature and target summary: Numerical (mean, median etc), Categorical(histogram).

6. Feature Engineering: Converts raw data into a higher representation
    * Numeric value binning: Example is bin age feature like 0-20 in Bin1, 21-40 in bin2, 41-60 in bin3 and 60 and above in bin4.
    * Quadratic features: Combine feature pairs. Example combine Education and Occupation feature into one feature.
    * Non-linear feature transformation
        - For numeric feature: Log, Polynomial power of target variable, feature values - may ensure a more "linear dependence" with output variable. Product/ratio of feature values.
        - Tree path features: use leaves of decision tree.
    * Domain specific transformation
        - Text features (stop words removal/stemming, lowering, punctuation removal,Cutting off very high/low percentiles, TF-IDF normalization)
        - Web page features (multiple fields of text: URL, in/out anchor text, title, frames, body, presence of certain HTML elements(tables,images), Relative style and positioning)

7. Model Training.