# AWS Global Infrastructure

The AWS Global Infrastructure consists of 22 launched regions, 5 announced regions around the world with 69 availability zones as per Feb 2020.

Region
------------------------------------------|
|                                         |
|  |--------------|    |--------------|   |
|  | Availability |    | Availability |   |
|  |    Zone      |    |     Zone     |   |
|  |  |--------|  |    |  |--------|  |   |
|  |  |  Data  |  |    |  |  Data  |  |   |
|  |  | Center |  |    |  | Center |  |   |
|  |  |        |  |    |  |        |  |   |
|  |  |--------|  |    |  |--------|  |   |
|  |--------------|    |--------------|   |
|-----------------------------------------|

## 1. Regions

A region is a physical location around the world. Each AWS region consists of multiple, isolated and physically separate Availability Zone's within a geographic region.


## 2. Availability Zones

An Availability zone is one or more discrete data centers with redundant power, networking and connectivity in an AWS region. All Az's in an AWS Region are interconnected. All traffic between Az's is encrypted.

## 3. Data Centers

Inside an Availability zone is/are one or more Data Centers, which contain the physical servers that run AWS resources.