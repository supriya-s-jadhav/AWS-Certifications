# AWS Certified Machine Learning - Speciality Course on Linux Academy

## Machine Learning And Deep Learning Fundamentals

### 1. Machine Learning Concepts

#### 1.1 Machine Learning Lifecycle

Collect Data -> Process data -> Split data -> Train -> Test -> Deploy -> Infer (If the model requires improvement, go back to split data step) -> Prediction

- Process Data:
    * Features and labels
    * Feature engineering
    * Feature reduction
    * Encoding
    * Formatting (RecordIO)

- Split Data: Train, Validation, Test

- Deploy:

    * Host model in execution environment according to the requirements.
    * Batch. You have a more that is not labelled and you want to label it over the data. For example you have a whole list of medical records of heart disease that are not labelled and apply batch process using model to label if they are susceptible to the heart disease, using appropriate infrastructure.
    * As a service. Like an web application end point that an application can use. It can send individual records into the service and have individual prediction being made. Example: Mobile app for hot dog or not.
    * Infrastructure (eg. load balancing)

- Inference:
    * Use this model in production to make inference/predictions.
    * Making predictions/inference. And label the real world data.


#### 1.2 Supervised, Unsupervised, Reinforcement

| Supervised | Unsupervised | Reinforcement |
|------------|--------------|---------------|
| Labelled data, and try to predict/infer </bt> Numeric Data, example is giving data, analyze the data and predict future numeric value </br> Classified data, example is give kiwi and penguin picture and determine what picture it is. This is also a labelled data</br>| Look at the data and find the distinct groups, it can be 2 or more. We are trying to analyze data with 10s and 100s of different dimensions, Unsupervised algorithms find relationship between the data points. | Often used in robotics and automation. For example, a robot know to pick the things from a bucket but it does not know what all the bucket has. We give it instruction and based on what it does we give it a reward. We tell a robot to pic a penguin, it picks up a bear, we give it a Reward:-1. The robot has another go, picks a penguin and gets a Reward:+1. Its like teaching students. |
| Providing pre-labelled data, so the model can see the pattern and and infer/predict the data. | We give data to ML model, it finds the meaning and pattern in the data | There are a lot if trial-and-error method with rewards. |


#### 1.3 Optimization

[Reference](https://www.kdnuggets.com/2017/04/simple-understand-gradient-descent-algorithm.html)

```
Gradient descent is an optimization algorithm used to find values of parameters (coefficients) of a function that minimizes a cost function. It is best used when the parameters cannot be calculated analytically and must be searched for by an optimization algorithm.

Lets now go step by step to understand the Gradient Descent algorithm:

Step 1: Initialize the weights(a & b) with random values and calculate Error (SSE)

Step 2: Calculate the gradient i.e. change in SSE when the weights (a & b) are changed by a very small value from their original randomly initialized value. This helps us move the values of a & b in the direction in which SSE is minimized.

Step 3: Adjust the weights with the gradients to reach the optimal values where SSE is minimized

Step 4: Use the new weights for prediction and to calculate the new SSE

Step 5: Repeat steps 2 and 3 till further adjustments to weights doesn't significantly reduce the Error
```

Gradient descent

    * Step size sets the learning rate.
    * If the step is too large, the minimum slope may be missed. Less efficient.
    * IF the step is too small, the process will take longer. Less efficient.

Setting the learning rate or gradient descent of the algorithm is one of the tuning parameter.

Its used to optimize many different types if ML algorithms like Linear Regression, Logistic Regression, SVM

#### 1.4 Regularization

Sometimes machine learning models don't perform as expected. Regularization helps improve the models.

- Regularization through Regression. If we are dealing with the dataset that has 10s and 100s of dimension, than the model will be able to regularize the model to all of those dimensions.

1. L1 regularization or Lasso

2. L2 regularization or Ridge

We apply regularization when our model overfit. Regularization is achieved through regression.

#### 1.5 Hyperparameters

There are 2 types of parameters used when we are training our model:

1. Hyperparameters

These are external parameters that we can set when we are initiating training job.

Example of hyperparameters are:

- Learning rate:
    Determines the size of the step taken during gradient descent optimization. Its value is between 0 and 1.
- Epochs
    The number times that the algorithm will process the entire training data. Each epoch contains one or more batches. Each epoch should see the model get closer to the desired state. Common values are 10, 100, 1000 and up.
- Batch size
    The number of samples used to train at any one time.  Could be all, one or some of your data (batch, stochastic, or mini-batch). Often 32, 64 and 128 if not full or batch or stochastic. Calculable from infrastructure.

    Example: If you have many instances running and you set batch size to 1, it will send the data only to one instance.


2. Parameters

Parameters are set inside the training job as the model is trained.


#### 1.6 Validation

K-fold Cross-Validation

LOOCV : Leave one out cross validation

### 2. Data

#### 2.1 Feature Selection and Feature Engineering

Its all about selecting right data. Remove the data that is not required.

Example: We have a dataset with variable: Name, Country, Age, Height, Start sign, Liked Tea. We are trying to find out if someone will like Tea or not.

We will not need Name hence remove it, Star Sign has nothing to do with predicting if a person will like tea or not hence remove it.

If a feature has low correlation and little variance (common values across the dataset), we might decide to remove.

Missing Values and Outliers

Engineering the values To improve model accuracy and speed up training.

Creating new variable from various different variables in dataset.

Feature Selection Tips:
- Use domain knowledge to drop irrelevant features
- Drop features with low correlation to labelled data
- Drop features with low variance
- Drop features with lots of missing data

Feature Engineering Tips:
- Simplify features and remove irrelevant information
- Standardize your data ranges across features
- Transform the data to suit the model/problem

#### 2.2 PCA (Principal Component Analysis)

- PCA is an Unsupervised ML model.
- Often used as a data preprocessing step
- There can be as many PC's as features or values.
- PC1 and PC2 can be used to plot a 2D graph to show groups of features.

#### 2.3 Missing nd Unbalanced Data

Missing values

- Imputation: Replace the missing values with Mean or median, remove the row or remove the column if lot of missing values.

Unbalanced data:

- Source more data.
- Oversample minority data
- Synthesize data
- Try different types of algorithms

#### 2.4 Label and One Hot Encoding

- ML algorithms use numbers
- Use label encoding to replace string values
- ML looks for patterns and relationships
- Use one hot encoding for categorical features.

#### 2.5 Splitting and Randomization

Data collected over the time have batches of data sorted. Splitting a sorted data is dangerous.

- Always randomize the data
- Even you are unaware of data clumping
- Some algorithms will shuffle data during training but not between the training and test data sets.

#### 2.6 RecordIO

Problem explained:

We have 1000s of images, and if the model will process is a way where it takes one image, processes it, and predicts for every single image than it will be a very long and time taking process.

Solution to this problem in put all the images in one file and process that file at once. This is the concept behind RecordIO. Amazon Sagemaker processes its data in RecordIO format.

- "Pipe mode" streams data (As opposed to "File mode")
- Faster training start times and better throughput
- Most Amazon SageMaker algorithms work best with RecordIO
    * Streams data directly from S3
    * Training instances don't need a local disk copy of data.

### 3. Machine Learning Algorithms

#### 3.1 logistic Regression

Type: Supervised

Example Inference: Binary classification

Use cases: Credit Risk, Medical conditions, PErform will perform an action. Answer should be in yes or no form.

#### 3.2 Linear Regression

Type: Supervised

Example Inference: Numeric 1,2,3

Use case: Financial Forecasting, Marketing effectiveness, Risk evaluation

#### 3.3 Support Vector Mcchines

Type: Supervised

Example Inference: Classification

Use cases: Customer classification, Genomic identification

Let us suppose we have two classes in a dataset: Red and Green. Purpose of using SVM is to classify a new data point in a right class.

#### 3.4 Decision Trees

Type: Supervised

Example Inference: Binary yes and no, Numeric 1,2,3 and Classification

#### 3.5 Random Forest

Type: Supervised

Example Inference: Binary Yes or No, Numeric 1,2,3 and Classification

#### 3.6 K-means

Type: Unsupervised

Example Inference: Classification

K represent the number we provide that tells how many classes we want.

K-means algorithm does lot of iterations to categorize data point sin k classes. How do we know which one is best ? The algorithm looks at the variation of the result.

#### 3.7 K-Nearest Neighbor

Type: Supervised

Example Inference: Classification

Use Cases: Recommendation Engine, Similar articles and objects

How to select K ??
- Make k large enough to reduce the influence of outliers
- Make k small enough that classes with a small sample size don't lose influence.

#### 3.8 Latent Dirichlet Allocation algorithm (LDA)

Type: Unsupervised

Example Inference: Classification, other

Use cases: Mostly used to text classification like Topic analysis, Sentiment analysis and Automated document tagging

Text document collection are called corpus.
_______________________________________________
|    Corpus                                     |
|  | Doc1:   |    | Doc2:       |  | Doc3    |  |
|  | Storage |    |   Machine   |  | Lambda  |  |
|  |         |    |   Learning  |  |         |  |
|  |         |    |             |  |         |  |
|  |         |    |             |  |         |  |
|                                               |
|_____________________________________________  |

- Randomly assign topics to each word
- Count the topics by document

| Word | topic1 | topic2 | topic3|
|------|--------|--------|-------|
| machine learning | 22 | 33 | 43 |
| fun run | 32 | 34 | 23 |
| python | 22 | 51 | 34 |

- Count the topics by document

| document | topic2 | topic2 | topic3 |
|----------|--------|--------|--------|
| Storage | 123 | 23 | 34 |
| Machine Learning | 43 | 143 | 45 |
| Lambda | 24 | 35 | 132 |

- Reassign the words to topics

Use case: Text analysis and topic allocation

### 4. Deep Learning Algorithms



### 5. Model Performance and Optimization

### 6. Machine Learning Tools and Framework

## AWS Services

## AWS Application Services AI/ML

## Amazon Sagemaker

### 1. Introduction

### 2. Build

### 3. Train

### 4. Deploy

### 5. Security

## Other AWS Services