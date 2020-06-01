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

#### Use cases

1. Process and evaluate logs immediately.

Example: Analyze system and application logs continuously and process within seconds.

2. Real time data analytics

Example: Run real-time analytics on click stream data and process it within seconds.

### Kinesis Data Firehose

Load data streams into AWS data stores.

Amazon Kinesis Data Firehose is the easiest way to capture, transform, and load data streams into AWS data stores for near real-time analytics with existing business intelligence tools.

We don't have to worry about Shards. We can use lambda for ETL, pre-processing with lambda is optional.

Data Producers -> Processing tools (optional) -> Storage (Redshift, S3->s3 event -> DynamoDB)

Difference in Kinesis Data Streams and Kinesis Data Firehose

Kinesis Data streams has shards and data retention (you can hold on to your data for 24 hours to 7 days).

With Kinesis Data firehose, you don't have to worry about shards. It is mainly use to streaming data store in storage service like S3.

<b> When should use Kinesis Data Firehose? </b>

* You want to collect/store streaming data.

* Processing is optional.

* Final destination is S3 (or other data store)

* Data retention is not important.

#### Use cases

1. Stream and store data from devices

Example: Capturing imp data from IoT devices, embedded systems, consumer applications and storing it into a data lake

2. Create ETL jobs on streaming data

Example: Running ETL jobs on streaming data before data is stored into a data warehousing solution.

### Kinesis Video Streams

Capture, process and store video streams.

Amazon Kinesis Video Streams makes it easy to securely stream video from connected devices to AWS for analytics, machine learning (ML), and other processing.

Data Producers (Web camera, audio feeds, radar producer) -> Data Consumers (EC2 continous consumer, EC2 Batch Consumer) -> Storage (S3)

<b> When should use Kinesis Data Video Streams? </b>

* Needs to process real-time streaming video data (audio, images, radar)

* Batch-process and store streaming video

* Feed streaming data into other AWS services like Sage Maker

#### Use cases

1. Amazon Cloud Camera detect movement in a video camera

### Kinesis Data Analytics

Analyze data streams with SQL/java.

Amazon Kinesis Data Analytics is the easiest way to process data streams in real time with SQL or Java without having to learn new programming languages or processing frameworks.

Allows you to continuously read and process live data streaming.

Streaming input (Kinesis Data Streams, Kinesis Firehose) --> Kinesis Data Analytics --> Storage and visualization (write SQL query and output to S3)

<b> When should use Kinesis Data Video Analytics? </b>

* Run SQL queries on streaming data.

* Construct applications that provide insight on your data.

* Create metrics, dashboards, monitoring, notifications and alarms.

* Output query results into S3 (other AWS data sources)

#### Use cases

1. Responsive real-time analytics

Example: Send real-time alarms or notification when certain metrics reach predefined threshold

2. Stream ETL jobs

Example: Stream raw sensor dat athen clean, enrich, organize, and transform it before it lands into data warehouse or data lake.

The kinesis Family Use cases

| Task at hand | Which kinesis service to use | Why |
|--------------|------------------------------|-----|
| Need to stream Apache log files directly from (100) EC2 instances and store them into Redshift | Kinesis Firehose (Since no pre-processing is required) | Firehose is for easily streaming data directly to a final destination. First the data is loaded into S3, the copied into RedShift |
| Need to stream live videos coverage of a sporting event to a distribute to customers in near real-time. | Kinesis Video Streams | Kinesis Video Streams processes real-time streaming video data (audio, image, radar) and can be fed into other AWS services |
| Need to transform real-time streaming data and immediately feed into a customer ML application | Kinesis data stream | Kinesis data stream allows for streaming huge amounts of data, process/transform it, and then store it or feed into custom applications or other AWS services. |
| Need to query real-time data, create metric graphs, and store output to S3 | Kinesis Analytics | Used for running SQL queries on streaming data and store output to S3 |