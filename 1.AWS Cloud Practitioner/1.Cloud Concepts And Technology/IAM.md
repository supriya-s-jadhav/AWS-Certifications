# IAM (Identity Access Management)

IAM is the service where AWS user accounts and their access to various AWS services is managed. The common use cases of IAM is tp manage:

1. Users
2. Groups
3. Access policies
4. Roles
5. User Credentials
6. User Password policies
7. MFA i.e. Multi-Factor Authentication
8. API keys for programmatic access

The user created when you create a new AWS account is called the root user. By default, the root user has full administrative rights. Any new or additional users created in the AWS account are created with no access to any AWS resources. The only granted access is the ability to login.

For a user to access any AWS service, permission must be granted to that user, which is managed in/by IAM.

## IAM Best Practices

AWS best practices are the guidelines that recommend settings, configurations, and architecture for maintaining a high level of security, accessibility and efficiency.

When a new AWS root account is created, it is best practice to complete the tasks listed in IAM under security status, including:

1. Delete your root access keys
2. Activate MFA on your root account
3. Create individual IAM users
4. USe groups to assign permissions
5. Apply an IAM password policy


