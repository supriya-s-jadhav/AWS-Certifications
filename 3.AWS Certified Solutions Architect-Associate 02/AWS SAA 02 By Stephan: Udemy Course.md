# AWS Certified Solutions Architect Associate C02

## 1. AWS Fundamentals

### 1.1 Regions And AZ

AWS has Regions spread globally.

Each Region has Availability zones, usually 3, minimum 2 and maximum 6. Each AZ has one or more data centers with redundant power, networking and connectivity. They are separate from each other, so that they are isolated from disasters. And, they are connected with high bandwidth, ultra-low latency networking.

### 1.2 IAM (Identity and Access Management)

IAM is global service

IAM has : Users, Groups, Roles

1. Users: Each user gets login credential and a unique login link. Users are person who work with AWS services.

2. Groups: Defined based on work role like developer, admin etc or Teams like Engineering, HR etc.

3. Roles: Roles are for internal usage and they are assigned to <b>Machines</b>

4. Policies: Policies are JSON document and they define what all the 3 of above can and cannot do.


Recommended best practices:
- Never use Root account.
- Users must be created with proper permission
- Policies are written in JSON (JavaScript Object Notation)
- Least privilege permission
- MFA set up
- One IAM Role per application
- Never write IAM credentials in code.
- Never use Root IAM credentials.

IAM Federation

IAM Federation allows users to use their corporate credentials to login to AWS account.

It uses SAML (Active Directory)

### 1.3 EC2

EC2 (virtual computer) is a virtual server that can be used to run applications in AWS. While setting up EC2, you can choose CPU type, storage, memory and networking resources. When you create an EC2 instance, you create it with an AMI.

- EC2 components:

    1. EC2 i.e. renting virtual machines
    2. EBS i.e. Storing data on virtual drives
    3. ELB i.e. Distributing load across machines
    4. ASG i.e. Scaling the services using an auto-scaling group

How to connect to EC2 instance running different OS?

<b>SSH</b>

Use SSH to connect with Mac, Linux and Windows OS greater than 10.

Use Putty to connect to Windows OS

Use EC2 instance connect to connect to any OS.

```
                                  | (Security Group: SSH-Port 22) |
Personal laptop/computer <------> |     EC2 (Linux OS)            |
                                  |     Public IP                 |
```

We make changes to what traffic to allow in and out to EC2 in Security Group.

- Security Groups

Firewalls in Cloud

One of the important topics of network security in AWS is Security Groups. They control how traffic is allowed into or out of the EC2 machines.

- They regulate:
    - Access to Ports
    - Authorized IP ranges : IPv4 and IPv6
    - Control incoming traffic i.e. inbound network
    - Control outgoing traffic i.e. outbound network

- Bonus points:
    - One Security Groups can be attached to multiple instances
    - One EC2 instance can have multiple Security Groups too
    - Security Groups are region/VPC specific. IF you launch a new instance in a different region, you need to create a new SG.
    - SG is not an app sitting on instance, it lives outside the instance.
    - It is good to maintain one separate security group for SSH access.

- Popular/known issues and root causes
    - IF your application is not accessible meaning its timed out, then its a security group issue
    - If you get a "connection refused" error, it means your application is not launched or the error is at application side
    - All inbound traffic is blocked by default
    - All outbound traffic is authorized by default

1. Allow from anywhere : SSH-type TCP-protocol 22-Port Range 0.0.0.0/0-Source

### 1.4 Private and Public IP(IPv4)

Networking has two sorts if IPs which are IPv4 and IPv6. This learning will include only IPv4, it is the most common format used online. The total possible IPv4 allowed are 3.7B:

```
[0-255].[0-255].[0-255].[0-255]
```

IPv6 are used in IoT

- Public IP
    - Public IP means the machine can be identified on the internet (WWW)
    - Must be unique across the whole web
    - It can be geo-located easily

- Private IP
    - Private IP means the machine can only be identified on a private network only
    - The IP must be unique across the private network
    - Two different private networks (two companies) can have the same IPs
    - Machines connect to WWW using a NAT + internet gateway (a proxy)


### Useful links to official AWS site :

[1. IAM](https://aws.amazon.com/iam/)