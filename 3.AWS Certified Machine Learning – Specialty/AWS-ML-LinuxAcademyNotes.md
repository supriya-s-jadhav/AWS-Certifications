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

```
_______________________________________________
|    Corpus                                     |
|  | Doc1:   |    | Doc2:       |  | Doc3    |  |
|  | Storage |    |   Machine   |  | Lambda  |  |
|  |         |    |   Learning  |  |         |  |
|  |         |    |             |  |         |  |
|  |         |    |             |  |         |  |
|                                               |
|_____________________________________________  |
```

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

#### 4.1 Neural Networks

Input layer --> Hidden layer --> Output

Every input neuron has weight. We multiply the value of neuron with neuron weight and add all the values we got for all the neurons. This brings the linearity in the model. Hence we add bias and Activation function to make it non-linear.

3 different types of activation function:

1. Sigmoid
2. ReLU (we will focus more on this)
3. Tanh

Loss Function: Its an optimization function to figure out better weight and biases to reduce the loss and make prediction more accurate.

Gradient Decent, Learning Rate

----> Forward propagation ---->

    Epoch (Loss Function)            Epoch is the combination of Forward and backward propagation

----> Backward propagation ---->

#### 4.2 Convolutional Neural Network

Type: Supervised

Example Inference: Classification

Use cases: Image classification, Spatial analysis

The hidden layers in Neural network are called Convolutional layers.

In CNN, it tries to identify different characteristics of an image and that is how it differentiates one image from others. For example, for a penguin image it will identify eyes, feathers, hands etc.

It uses filters. CNN comes with some pre-trained edge detection (transfer learning)

#### 4.3 Recurrent Neural Network (RNN)

Type: Supervised

Example Inference: Other

Use cases: Stock predictions, Time series data, Voice recognition (seq to seq) like translating eg. speak in english and translate to spanish.

What makes RNN different from other Neural Network ??

RNN maps the series of activities. Whatever is the output, it gives it as input back again. RNN has memory that is used to remember previous prediction and it allows to influence future prediction.

It has ability to remember a bit. <b> LSTM i.e. Long short term memory can remember a lot. </b>

### 5. Model Performance and Optimization

#### 5.1 Confusion Matrix

We will know how well the ML algorithm performing using confusion matrix.

Actual Values
| | | | |
| -|-|-|-|
| | | Like Dogs | Don't like dogs |
| Model | Like Dog | True Positive </br> Actual and predicted Like dogs | False Positive </br> Actual: does not like dog and Predicted: Like dogs |
| Prediction | Don't like dog | False Negatives </br> Actual: Likes dogs and Predicted: Does not like dogs | True Negatives </br> Actual and Predicted: Does not like dogs |

#### 5.2 Accuracy and Precision

Accuracy:

The proportion of all predictions that were correctly identified. Simplest way of looking at the performance of the model. How right is the model ?

Accuracy = TP + TN / Total

Precision: Proportion of actual positives we identified.

Precision = TP / TP + FP

Overall accuracy can sometimes be a deceptive measure because of unbalanced classes.

A general improvement to using overall accuracy is to study sensitivity and specificity separately. Sensitivity, also known as the true positive rate or recall, is the proportion of actual positive outcomes correctly identified as such. Specificity, also known as the true negative rate, is the proportion of actual negative outcomes that are correctly identified as such.

#### 5.3 Sensitivity and Specificity

A general improvement to using overall accuracy is to study sensitivity and specificity.

Positive outcome means y is 1, and
negative outcome means when y is 0.

<b>Sensitivity or Recall or TPR </b>

The number of correct positives out of the actual positive results. High sensitivity means y equals 1 implies y_hat equals 1.

Sensitivity = TP / TP + FN

Closer he sensitive value of a model to 1, the less False Negatives are.

Example:

a) In Fraud detection, we want as close Sensitivity to 1 as possible.

b) In case of plane safety, sensitivity close to 1 is important. If plan is not safe and we predicted plane is safe, is much more costly error than predicting a safe plan as malfunctioned and having it grounded.


<b>Specificity or TNR</b>

The number of correct negatives out of the actual negative results. High specificity means y equals 0 implies y_hat equals 0.

Specificity = TN / TN + FP

Example:

a) In classifying appropriate video content for kids, we want specificity value as close to 1 as possible.

b) In a capital murder criminal case, we want specificity close to 1. If we predict an innocent person as criminal and we can end up killing an innocent person.

<b>Precision or Positive Predictive value </b>

PREcision is to PREgnancy tests as reCALL is to CALL center.


This tells when you predict something positive, how many times they were actually positive. whereas,

Precision = TP / TP + FP

Having said above, in case of spam email detection, One should be okay if a spam email (positive case) left undetected and doesn't go to spam folder but, if an email is good (negative), then it must not go to spam folder. i.e. Precison is more important. (If model predicts something positive (i.e. spam), it better be spam. else, you may miss important emails).

#### 5.4 ROC/AUC

A very common approach to evaluating accuracy and F1-score is to compare them graphically by plotting both. A widely used plot that does this is the receiver operating characteristic (ROC) curve. The ROC curve plots sensitivity (TPR) versus 1 - specificity or the false positive rate (FPR).

#### 5.5 Gini Impurity

Gini Impurity = 1 - (probability of dogs)2 - (probability of cat)2

#### 5.6 F1 Score

F1 Score = 2 (Recall x Precision/ Recall + Precision)

### 6. Machine Learning Tools and Framework

#### 6.1 ML  and DL frameworks

What is the difference between frameworks and algorithms ?

1. TensorFlow

Examples of frameworks: PyTorch, mxnet, Keras, GluOn, TensorFlow, Scikit-learn

Frameworks provide lot of algorithms to work with.

tensorflow terminologies using python:

tf.get_default_graph()
graph.get_operations()

tf.Session()

2. PyTorch

3. MXNet

We can mention cpu or gpu, on which MXNet algorithm will run.

4. Sci-kit learn

## AWS Services

### 1.S3

S3

- Cost effective storage for large amounts of data.
- Structured and unstructured
- Data lake (Seems like Amazon wants people for use S3 for data lake)

2. Data Lake

- Collection of different S3 buckets.
- Destination for all data sets
- Structured data (CSV, JSON)
- Unstructured data (Text files, Images)

Advantages

- Add data from many sources
- Define the data schema at the time of analysis
- Much lower cost than data warehouse solutions
- Tolerant of low-quality data

Disadvantages

- Unsuitable for transactional systems
- Needs cataloguing before analysis

Architecture 1:
```

                                    (when Athena queries data in S3, it creates a view of data in S3 and stores it back in S3 to be sued for ML/DL tools )
Amazon Kinesis Data Firehouse -> S3 -> Athena -> ML/DL (SageMaker)
                                |___________|
                            AWS Glue (It can crawl through S3 bucket and produce a Data Catalog that Athena can use)
```

Architecture 2

```
Amazon Kinesis Data Firehouse ---> S3 ---> Athena ---> ML/DL(SageMaker)
Other                                     EMR/Spark
```
Security
- IAM USers and Roles
- Bucket Policy

Encryption
- S3 SSE
- S3 KMS

### 2. AWS Glue

When we put all the data in S3, we need some tool to catalogue the data in S3 to be able to query and use it in other tools like Athena.

- Creates catalogues of data (schema)
- Performs ETL
- Some limited ML capabilities

AWS Glue is not a database, it serves as an end point to different data sources.

### 3. Athena

- Query S3 data with SQL
- Source data from multiple S3 locations
- Save outputs to S3
- Use for data pre-processing ahead of ML

### 4. Quicksight

- Its a BI tool
- Visualiza data from many sources
    - Dashboards
    - Email Reports
    - Embedded reports
- End-user targeted

### 5. Amazon Kinesis

- Ingesting large scale data
- Lots of data from a few sources (video)
- Small amount of data from many sources (IoT)

| Amazon Kinesis | Amazon Kinesis Data Streams | Amazon Kinesis Data Firehose | Amazon Kinesis Data Analytics |
|--------|--------|---------|----------|
| Securely stream video from connected devices to AWS for analytics, ML, playback, other processing | General endpoint for ingesting large amounts of data for processing by: Kinesis data analytics, Spark on EMR, Amazon EC2, Lambda | Simple endpoint to stream data into: S3, Redshift, ElasticSearch, Splunk(third party tool) | Processing streaming data from kinesis streams or Firehose at scale using : SQL, Java libraries |

Architecture 1

IoT --> Amazon Kinesis Data Streams --> EMR/Spark --> S3

Architecture 2

Video Camera --> Amazon Kinesis Video Streams --> Amazon Rekognition Video --> Amazon Kinesis Data Streams --> AWS Lambda --> AWS SNS --> Mobile

### 6. EMR with Spark

Amazon EMR
- Managed service for hosting massively parallel compute tasks.
- Integrates with storage service S3
- Petabyte scale
- Uses 'big data' tools:
    - Spark
    - Hadoop
    - HBase

EMR (Elastic Map Reduce)

```
_________________________________
| MAster Node                    |
|  _____________  _____________  |
|  | Core Node  | | Task Node |  |
|   ------------   -----------   |
| -------------------------------|

```

Core Nodes : can add more EC2 instances gracefully there
Task Nodes: can add spot instances

Apache Spark
- Fast analytics engine
- Massively parallel compute tasks
- Deployed over clusters of resources

Variations of Spark run on:
- Amazon EMR
- Amazon SageMaker

Amazon EMR and Spark

ML
- Integrates into Amazon SageMaker
- Performs massive ETL of data into SageMAker

S3 --> EMR/Spark --> Sagemaker

### 7. EC2 for ML

EC2 instance types

- AWS EC2 instance types targeted at ML tasks:
    - Compute optimized
    - Accelerated Computing (GPU)
- The ml.*instances are not available outside of SageMaker

Amazon Machine Images(AMIs): Ubuntu, Linux, Windows
- Condo based Deep Learning AMIs
- Libraries: TensorFlow, Keras,MXNet, Gluon, PyTorch
- GPU acceleration: CUDA 8 and 9, cuDNN 6 and 7, NCCL 2.0.5 libraries, NVidia Driver 384.81

Amazon MAchine Images (AMIs)
- Deep LEarning Base AMIs
(They don't with Conda and libraries. IF you know what yo want, you have to install)

EC2 Instance type limits
- Brand new account = no ML for you
- Service limit increases take days.

Amazon SageMaker ML Instance Types

<b> AWS ML service is no longer available, it only supports the existing project. It is superseded by SageMaker</b>

## AWS Application Services AI/ML

Services used in text to speech, speech to text, image classification, text classification

### 1. Amazon Rekognition

Amazon Rekognition for Image analysis

- Image and video analysis
- PRe trained deep learning
- Simple API
- Image moderation
- Facial analysis
- Celebrity recognition
- Face comparison
- Text in image

Use cases:
- Create a filter to prevent inappropriate images being sent via a msg platform. This can include nudity or offensive text.
- Enhance metadata catalog of an image library to include the number of people in each image.
- Scan an image library to detect instances of famous people.

Amazon Rekognition for video analysis


- Image and video analysis
- PRe trained deep learning
- Simple API
- Process through Stored video (eg. stored video in S3)
- Streaming video like security camera

Architecture for stored video and streaming videos

![example for stored video](https://github.com/supriya-s-jadhav/AWS-Certifications/blob/master/3.AWS%20Certified%20Machine%20Learning%20%E2%80%93%20Specialty/Images/architecture_stored_video.png)

![example for streaming video](https://github.com/supriya-s-jadhav/AWS-Certifications/blob/master/3.AWS%20Certified%20Machine%20Learning%20%E2%80%93%20Specialty/Images/architecture_for_streaming_video.png)

-> S3 -> Rekognition

USe cases
- Detect people of interest in alive video stream for a public safety application
- Create a metadata catalog for stock video footage
- Detect offensive content with video uploaded to a social media platform

### 2. Amazon Polly

It converts text into speech. Think of it as a parrot

- Text to speech (TTS)
- Deep LEarning powered
- Simple API
- Language
- Female or male
- Custom lexicons

Use Cases
- Create accessibility tools to "read" web content
- Provide automatically generated announcements via a public address (PA) system
- Create an automated voice response (AVR) solution for a telephony system (including Amazon Connect)

### 3. Transcribe

Automatic speech to text translate.

- Automatic speech recognition
- Pre-trained deep learning
- simple API
- Language
- Custom vocabulary

Use cases
- Create a call center monitoring solution that integrates with other service to analyze caller sentiment
- Create a solution to enable text search of media with spoken words
- Provide a closed captioning solution for online video training.

### 4. Amazon Translate

We can provide different documents/strings/text in various languages and expect it to get back in different language.

- Text translation service.
- Uses pre-trained deep learning
- Simple API
- Batch (bunch of docs in S3) or real-time
- Languages
- Custom terminology

Use cases
- Enhance an online customer chat application to translate conversations in real-time
- Batch translate documents within a multilingual company
- Create a news publishing solution to convert posted stories to multiple languages

### 5. Amazon Comprehend

Its a natural language processing solution. We provide text and it returns text's analysis.

- Text analysis
- Natural language processing (NLP)
- Pre-trained deep learning
- Simple API

Features
- Keyphrase extraction
- Sentiment analysis
- Syntax analysis
- Entity recognition
- Medical Named Entity and Relationship Extraction (NERe)
- Custom entities
- Language detection
- Custom classification
- Topic modeling
- Multiple language support

Use cases
- Perform customer sentiment analysis on inbound message to support the system
- Create a system to label unstructured (clinical) data to assist in research and analysis
- Determine the topics from transcribed audio recording of company meetings.

### 6. Amazon Lex

It has a lot to do with Alexa. It is a conversation interface service. Its like Alexa or chat bot.

- Conversation interface service
- Think Alexa or chatbots
- Voice enabled or text
- Automatic speech recognition
- Natural language understanding (NLU)

Use cases
- Create a chatbot that triages customer support requests directly on the product page of a website
- Create an automated receptionist that directs people as they enter a building
- Provide an interactive voice interface to [...] application

### 7. Amazon Service Chaining

Example : Input text file and tell the sentiment in the file.

```

Text file  --> S3 --> Lambda --> Amazon Comprehend

- Lambda will invoke Amazon Comprehend service, tell it to get text file from S3, do sentiment/keyword analysis and respond back to lambda.
- It will do the next step like save the response in S3 or send an SNS notification

```
```
Bad architecture:

Text file -> S3 -> Lambda -> Amazon Translate
              |                      |
              |        |<----------->|
              <---------------------->
                       |<------------------------>  Amazon Comprehend
- Lambda will invoke Amazon translate service, it will get the text file form S3 and respond back to lambda with translated file.
- Next, lambda will invoke and send the file to Amazon comprehend and it will respond back with processed file to lambda.
- Lambda will do the next step like save in S3 or send SNS.

Lambda best practices:
- It should do only one job at a time. Im above architecture we saw it is doing multiple jobs.

Solution: Add second Lambda? Not good either. Use AWS Step Functions

AWS Step Functions: To orchestrate multiple lambda function in big architecture.
                _______________________________
                |Step Function (state machine) |
     lambda1 -->|   Lambda2 -----> Lambda3     |--> S3/SNS
       /|\      |------------------------------|
        |            /|\             /|\
        |            \|/             \|/
Text -> S3 <------  Amazon <------  Amazon
file       ------> Translate ----> Comprehend

Example: Working with audio files. Replace Translate service with Transcribe in above architecture.
                _______________________________
                |Step Function (state machine) |
     lambda1 -->|   Lambda2 -----> Lambda3     |--> S3/SNS
       /|\      |------------------------------|
        |            /|\             /|\
        |            \|/             \|/
Text -> S3 <------  Amazon <------  Amazon
file       ------> Transcribe ----> Comprehend

Above is bad architecture. Lambda2 function has to wait until Amazon Transcribe to finish and sometimes Transcribe can take long time to finish. And, the architecture starts to fall apart. IT is an asynchronous service.

Solution:
                __________________________________________________
                |Step Function (state machine)                    |
     lambda1 -->|   Lambda2 -> time -> Lambda3 -> ? -> Lambda4    |--> S3/SNS
       /|\      |-------------------------------------------------|
        |            /|\                 |/|\             /|\
        |            \|/                 | |              \|/
Text -> S3 <------  Amazon <-------------| |             Amazon
file       ------> Transcribe -------------|            Comprehend

```

## Amazon Sagemaker 3

### 1. Introduction

Its a fully managed service that covers the entire machine learning workflow to label and prepare your data, choose an algorithm, train model, tune and optimize it for deployment, make predictions and take action.

The three stages of SageMaker:

- Build
    - Preprocessing
    - Ground Truth
    - Notebooks

- Train
    - Built-in algorithms
    - Hyperparameter tuning
    - Notebooks
    - Infrastructure

- Deploy
    - Realtime
    - Batch
    - Notebooks
    - Infrastructure
    - Neo

### 2. Build

#### 2.1 Data Preprocessing

- SageMaker Components
    - SageMaker Notebooks
    - Sagemaker Algorithms

#### 2.2 Amazon Ground Truth

Build highly accurate training data sets using machine learning and reduce data labelling costs by up to 70%.

- Features
    - Reduce data labeling costs by up to 70%
    - Work with public and private human labelers
    - Define work instructions

#### 2.3 AWS SageMaker Algorithms

Where do we get built-in algorithms in AWS SageMaker

- Sources
    - SageMaket built-in algorithms
    - AWS Marketplace
    - Custom (you can build your own algorithms)

- SageMaker Built-in Algorithms
    - BlazingText
    - Image classification algorithm
    - K-means
    - KNN
    - Latent Dirichlet allocation algorithm
    - Linear learner algorithm
    - Object2Vec algorithm
    - PCA algorithm
    - Random Cut Forest algorithm
    - Sequence-to-sequence algorithm
    - XGBoost algorithm

1. BlazingText

- Type: Word2vec / Text classification (behind Amazon Comprehend)

- Use cases:
    - NLP
    - Sentiment analysis
    - Named Entity Recognition
    - Machine translation

 2. Image Classification Algorithm

 - Type: CNN (behind Amazon Rekognition)
 - Use Cases
     - Image Recognition
3. K-means

- Type : K-means, Based off web-scale k-means clustering algorithm
- Use cases:
    - Find discrete groupings within data

3. LDA
- Type: LDA algorithm (behind Amazon Comprehend)
- Use cases
    - Text analysis
    - Topic discovery

4. PCA

- Type: PCA
- Use cases
    - Reduce the dimensionality i.e. number of features

5. XGBoost
- Type: eXtreme Gradient Boosting, Gradient boosted trees algorithm
- Use cases:
    - Making predictions from tabular data

![Read AWS official documentation of SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html)

### 3. Train

Architecture of SageMaker built-in algorithms.

Amazon ECS (Elastic Container Service): It has a docker image in it.

```
____________    __________    _____________
| S3       |    | EC2    |    | ECS       |
| Training |    |  ML    |    | Docker    |
|  data    |    |        |    | Containers|
|__________|    |________|    |___________|

```

- ML Docker Container:
    - SageMAker Built-in algorithms
    - AWS deep learning containers
    - AWS Marketplace (the algorithms in marketplace are available as docker images)
    - Custom (build your own)

Where we get data from ??

S3, EFS, Amazon FSx for Lustre --> Parameter: Channel (eg. train, validation, train_lst, validation_lst, model)

When you create a training job in Amazon Sagemaker, it will manage the infrastructure but you still have to tell it what to use.

EC2 instances
- Amazon recommend:
    - m4 family, c4 family and p2 family (they have GPU and used for image classification)
- Some algorithms only support GPUs
- GPU instances are more expensive but faster

Managed Spot Training
- MAnaged spot training can optimize the cost of training models up top 90% over on-demand instances
- Checkpoints (snapshot of model state in S3)

Hyperparameter:

1. Choose an algorithm

2. Set ranges of hyperparameter like set of Epochs

3. Choose the metric to measure

Amazon SageMAker automatic Model Tuning
- works with:
    - AWS built-in algorithms
    - custom algorithms
    - SageMaker pre-built containers

- Limits
    - There are limits
    - Be mindful of EC2 resource limits

### 4. Deploy

#### 4.1 Inference pipelines

Inference data -> Model1 -> Model2 -> Model3 -> Inference or prediction

```
    Lambda
      \|/
SageMaker Endpoint
      \|/
S3 -> Model -> ECR


Batch Inference

S3 -> Batch Transform Job -> S3
        \|/
S3 -> SageMaker -> ECR

```

Steps in creating inference in model:

1. Define Model
2. Create Endpoint configurations
3. Create Endpoints
4. Batch transform jobs

#### 4.2 Accessing Inference from Apps

```
App -> AWS api/sdk -> SageMaker Endpoint
     (API Gateway->Lambda)

```
### 5. Security

- Notebook root access
- SageMaker instance profiles
- SageMaker does not support resource-based policies. For example policy to S3. It attaches roles to S3 bucket to access SageMaker.

VPC security controls
- SageMaker hosts models in a public VPC by default.
    - Create a private VPC
- Models and data stored in S3 (public internet)
    - Create an Amazon S3 VPC endpoint
    - Use a custom endpoint policy for S3
    - Encrypt the data within S3

## Other AWS Services