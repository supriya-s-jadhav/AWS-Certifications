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

* Transient Data store: retention period of the data records are 24 hours. We can increase it to 7 hours.

How do we get out data into shards or kinesis data streams ??

1. Kinesis Producer Library (KPL)

Easy to use library that allows you to write to a kinesis data streams.

2. Kinesis Client library (KCL)

Integrated directly with KPL for consumer applications to consume and process data from kinesis Data Stream.

3. Kinesis API (AWS SDK)

Used for low level API operations to send records to a kinesis Data stream.

<b> When to use Kinesis Data Streams ? </b>

* Needs to be processed by consumers.

* Real time analytics

* Feed into other services in real time

* Some action needs to occur on your data

* Storing data is optional

* Data retention is important

### Use cases

1. Process and evaluate logs immediately.

Example: Analyze system and application logs continuously and process within seconds.

2. Real time dat analytics

Example: Run real-time analytics on click stream data and process it within seconds.

### Kinesis Data Firehose

Load data streams into AWS data stores.

Amazon Kinesis Data Firehose is the easiest way to capture, transform, and load data streams into AWS data stores for near real-time analytics with existing business intelligence tools.

We don't have to worry about Shards. We can use lambda for ETL, pre-processing with lambda is optional.

Data Producers -> Processing tools (optional) -> Storage (Redshift, S3->s3 event -> DynamoDB)

Difference in Kinesis Data Streams and Kinesis Data Firehose

Kinesis Data streams has shards and data retention (you can hold on to your data for 24 hours to 7 days).

With Kinesis Data firehose, you don't have to worry about shards. It is mainly use to streaming data store in storage service like S3.

When should use Kinesis Data Firehose?

1. You want to collect/store streaming data.

2. Processing is optional.

3. Final destination is S3 (or other data store)

4. Data retention is not important.


### Kinesis Video Streams

Capture, process and store video streams.

Amazon Kinesis Video Streams makes it easy to securely stream video from connected devices to AWS for analytics, machine learning (ML), and other processing.

### Kinesis Data Analytics

Analyze data streams with SQL/java.

Amazon Kinesis Data Analytics is the easiest way to process data streams in real time with SQL or Java without having to learn new programming languages or processing frameworks.