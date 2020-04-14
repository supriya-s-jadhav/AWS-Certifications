# Streaming Data Collection

How do we get streaming data into AWS ?

Amazon Kinesis

Easily collect, process, and analyze video and data streams in real time.

Its a family of tool. It consists of :

1. Kinesis Data Streams
2. Kinesis Data Firehose
3. Kinesis Video Streams
4. Kinesis Data Analytics

### Kinesis Data Streams

Capture, process and store data streams.

Amazon Kinesis Data Streams is a scalable and durable real-time data streaming service that can continuously capture gigabytes of data per second from hundreds of thousands of sources.

Data Producers ---> Kinesis streams (Shard) ---> Data Consumers (Processing Tools: EC2, Lambda, EMR, Kinesis Data Analytics) ---> Storage and Analyzation (S3, DynamoDB, RedShift, BI Tools)

<b>Shrads</b>

They are wrappers or containers that contains all the streaming data that we want to load to AWS. You can use 1 to 500 shards by default. If you need more, you can request to AWS.

* Each shard consists of a sequence of data records. These can be ingested at 1000 records per second.

* Default limit of 500 shards, but you can request increase to unlimited shards.

* A data record is the unit of data captured:

    a. Sequence number

    b. partition key

    c. data blob (your payload, upto 1 MB)

### Kinesis Data Firehose

Load data streams into AWS data stores.

Amazon Kinesis Data Firehose is the easiest way to capture, transform, and load data streams into AWS data stores for near real-time analytics with existing business intelligence tools.

### Kinesis Video Streams

Capture, process and store video streams.

Amazon Kinesis Video Streams makes it easy to securely stream video from connected devices to AWS for analytics, machine learning (ML), and other processing.

### Kinesis Data Analytics

Analyze data streams with SQL/java.

Amazon Kinesis Data Analytics is the easiest way to process data streams in real time with SQL or Java without having to learn new programming languages or processing frameworks.