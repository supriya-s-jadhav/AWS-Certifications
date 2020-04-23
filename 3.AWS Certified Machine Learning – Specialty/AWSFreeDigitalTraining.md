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

7. Model Training
    * Parameter tuning: Right parameters has to be chosen for the right type of problem.
        - Loss function: That calculates how far your prediction are from ground truth values.

        Square: regression, classification
        Hinge: classification only, more robust to outliers
        Logistic: classification only, better for skewed class distributions

        - Regularization: That can increase generalization of the model to better fit the unseen data

        Prevent overfitting by constraining weights to be small.

        - Learning Parameters: USed to control how fast/slow algorithm learns

        Decaying too aggresively: algorithm never reaches optimum
        Decaying too slowly: algorithm bounces around, never converges to optimum.

8. Model Evaluation
    * Overfitting and Underfitting: Don't fit training data to obtain mazimum accuracy.
    * Bias-Variance tradeoff

    | |              |               |
    |-|--------------|---------------|
    | | Low variance | High Variance |
    | High Bias | Too far | Little close to what we might want|
    | Low bias | Exactly what we want | Far dispersed |

    * Evaluation metrics for regression: Used to evaluate how good a model is
        - RMSE Root mean Square Error: Compares true value with predicted value. The lower the better. Check on test data and not train data.
        - MAPE Mean Absolute Percent Error: Compares true value with predicted value. The lower the better. Check on test data and not train data.
        - R-square: how much better is the model compared to just picking the best constant. R-square = 1 - ( Model mean squared error / variance )
    * Evaluation Metrics (classification problems): Metrics when classification is used for predicting target class
        - Confusion metrix
        - ROC curve
        - Precision-Recall

        | | | |
        |-|-|-|
        | | Actual | Actual|
        | Predicted | True positive | False positive |
        | Predicted | False negative | True negative |

    * Precision-Recall:
        - Precision: TP / (TP + FP)
        - Recall: TP / (TP + FN)

9. Business Goal Evaluation
    * Evaluate how well different models performed and make final decision to deploy or not. Evaluation depends on:
        - Accuracy
        - Model generalization on unseen/unknown data
        - Business success criteria
    * Augmenting your data
        - Add more information
    * Prediction
        - Data distribution is important.

## 2. Developing ML Models

This AWS free digital training has below 7 modules:

1. Introduction to Amazon SageMaker
2. Introduction to Amazon SageMaker Neo
3. Machine LEarning Algorithms Explained
4. Automatic Model Tuning in Amazon SageMaker
5. Advanced Analytics with Amazon SageMaker
6. Anomaly Detection on AWS
7. Building Recommendation Systems with MXNet and GluOn

### 1. Introduction to Amazon SageMaker

Its a fully managed service that enables to quickly and easily build, train and deploy ML models.

It has 3 major components:
    - Amazon SageMaker Notebooks Service
    - Amazon SageMaker Training Service
    - Amazon SageMaker Hosting Service

### 2. Introduction to Amazon SageMaker Neo

It provides portable code. It takes model from various frameworks (TensorFlow, XGBoost, PyTorch, mxnet), compiles to one form internally, finds out best device to deploy on out of many options available.

(TensorFlow, XGBoost, PyTorch, mxnet) --> Neo Compiler Container (It converts the framework specific code to non-specific code. It will perform high level optimization) --> Shared Object Library

Benefits:

1. Provides simple, easy to use.

Key Takeaways:

1. Popular deep learning and decision tree models.

2. Apache MXNet, TensorFlow, PyTorch, XGBoost

3. Various Amazon EC2 instances and edge devices

4. Up to 2x performance speedup and 100x memory footprint reduction at no additional charge.


### 3. Machine Learning Algorithms Explained
1. Supervised Learning
    - Linear Supervised Algorithms
    - Logistic Regression
    - Amazon Sagemaker has: Linear Learner (Linear + Logistic Regression)
    - SVM
    - Decision Tree (Non-linear supervised algorithms)
    - XGBoost - Gradient Boosted Trees
    - Factorization MAchines
2. Unsupervised Learning
    - Clustering
    - Anomaly Detection
    - Random Cut Forest
    - Amazon Comprehend : Topic modelling for topics with text content.
    Library of new articles -> Amazon Comprehend (discover 8 topics) -> Output (Storm, Stock Market...Crisis, ML with weights defined)
    - K-means (Sagemaker)
    - PCA (Sagemaker)
    - LDA (Latent Dirichlet Allocation) (Sagemaker and Comprehend)
    - Random Cut forest for Anomaly Detection (Sagemaker and Kinesis Data Analytics)
    - Hot Spot Detection (Kinesis Data Analytics)
3. Reinforcement Learning
4. Deep Learning
    - Neural Networks
    input (weights(parameters)) --> Activation Function --> Output
    - Convolutional Neural Network (Specially used for image classification. Semantic segmentation, Artistic Style transfer)
    - Recurrent Neural Networks (RNN)
    - CNN for Image classification (ResNet in AWS Sagemaker)
    - RNN for text summarization, translation (seq2seq in AWS sagemaker)
    - Neural Topic Modelling (NTM)
    - Time series Prediction DeepAR Forecasting

### 4. Automatic Model Tuning in Amazon SageMaker

Build, Train, Tune and Deploy in Amazon SageMaker.
        |------ Algorithms -----------------------|
        |------ Frameworks -----------------------|
        |------ Docker ---------------------------|
        |-- Tune -|

SageMaker has custom algorithms. Its more scalable and efficient.

Pre built deep learning framework like Tensorflow, MXNet, Pytorch

Let us learn more about Amazon SageMaker's Model Tuning

Hyperparameters: Help you tune ML model to get best performance. It makes sure to get best predictive performance

Neural networks
    - Learning rate
    - Layers
    - Regularization
    - Drop-out

Trees (Decision Tree, Random Forest etc)
    - Number
    - Depth
    - Boosting step size

Clustering
    - Number
    - Initialization
    - Pre-processing

Hyperparameter space
    - Large influence on performance
    - Grows exponentially
    - Non-linear/ interact
    - Expensive evaluations

Tuning
    - Manual
        * Defaluts, guess and check
        * Experience, intuition, and heuristics
    - Brut force
        * Grid
        * Random
        * Sobol
    - Meta model (build another ML model on top of another ML model)

SageMaker's method:
    - Gaussian process regression models objective metric as a function of hyperparameters.
        * Assumes smoothness
        * Low data
        * Confidence estimates
    - Bayesian optimization decides where to search next
        * Explore ans exploit
        * Gradient free
SageMaker integration
    - Black box
        * SageMaker algos
        * Frameworks
        * Bring your own
    - Flat hyperparameters
        * Continous
        * Integer
        * Categorical
    - Objective metric logged
        * CloudWatch
        * regex
Example: hyperparatemr_tuning_mxnet_gluon

### 5. Advanced Analytics with Amazon SageMaker

1. Provide an in-depth overview of Amazon SageMaker
2. Highlight benefits of using SageMaker with Apache Spark
3. Dive deep into the SageMaker-Spark SDK
4. Articulate how to build and train ML models
5. Discuss how to create ML pipelines

Amazon SageMaker + Apache Spark overview

SageMaker
    - Dedicated training and hosting
    - Amazon SageMaker algorithms
    - GPU support
    - Single machine or distributed environment

Apache Spark:

Spark SQL, Spark Streaming, MLlib, GraphX
|----------------------------------------|
Spark (Powerful data pre-processing tool, rich ecosystem, Distributed processing)

Using Sagemaker+Spark: Allows to use any algorithm provided by both.
    - Spark runs locally on SageMaker notebooks.
    - The SageMaker-Spark SDK: i) Amazon SageMaker algorithms are compatible with Spark MLLib ii) There are Spark and Amazon SageMaker hybrid pipelines.
    - Connect a SageMaker notebook to a Spark cluster(eg. Amazon EMR)
Scala and Python SDK

Out of the box classes:
    - Model definition: SageMakerEstimator
    - Training and inference: SageMakerModel

Ready to use algorithms from Amazon SageMaker
    - K-means, XGBoost, PCA, Linear Learner Classifier, etc

Spark MLLib Concepts: DataFrame (input data), Estimator (algorithm), Model (Model created)

SageMaker-Spark SDK

Model Building and Training

Demo

### 6. Anomaly Detection on AWS

### 7. Building Recommendation Systems with MXNet and GluOn


