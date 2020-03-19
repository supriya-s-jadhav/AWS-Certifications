# Key Points to Revise Before Exam:

There are only 3 AWS services that are global: IAM, S3 and DNS.

## Support Plans

Be able to identify which support plan is best based on the given test cases.

Example:

a) Whether you need TAM (Technical Account Manager) ?

b) Based on different service level agreements, like if you need response time of 4 hours or 1 hour etc.

## Billing Alarm

How can you get automatic notifications in different test cases example: if your account goes above $1000 ?

## IAM

When to make use of IAM (Identity Access Management) Users, Groups, Roles and Policies.

## S3

* S3 is Object-based meaning allows you to upload files (0 bytes to 5TB).
* The storage is unlimited and Files are stored in Buckets.
* S3 os universal namespace. eg: http://bucket_name.s3.amazonaws.com/filename
* S3 is a key(file name) and value(data) type.
* It has Read after Write consistency for PUTS of new objects.
* Eventual consistency for overwrite PUTS and DELETES.
* 6 different Storage classes: S3 Standard, S3 IA, S3 One Zone IA, S3 Intelligent Tiering, S3 Glacier, S3 Glacier Deep Archive.

## CloudFront

* Its amazon's CDN (Content Delivery Network). IF you turn on cloudFront, you will set up Edge locations to make data available with low latency.
* Know what is TTL ?
* READ and WRITE in Edge Locations.
* Charges for cached data in Edge Location