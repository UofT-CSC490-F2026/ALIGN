# ALIGN infrastructure

Account: **183754987292**. Region: **us-east-2 (Ohio)**.
Terraform: **1.16.x**. AWS provider: **6.x**, exact version recorded in each root's
committed .terraform.lock.hcl. AWS credentials come from the local CLI profile.

## Scope and ownership

PR 1 creates the state bootstrap and documents the shared interface.
PR 2 implements modules/network. PR 3 wires dev and prod to the module.
Environment folders are scaffolding until PR 3; do not apply them yet.

- bootstrap/: independent root owning the state bucket and lock table.
- modules/network/: reusable networking owned by Jinhao.
- envs/dev/ and envs/prod/: separate root modules and separate remote state.
- Future modules/compute and modules/database consume network outputs.

One shared state per environment; never create a separate state for each teammate.
Keep the default Terraform workspace. Separate directories and backend keys
already isolate environments.

## Local verification (no AWS deployment)

Run from the repository root:

```powershell
terraform fmt -check -recursive infra
terraform -chdir=infra/bootstrap init -backend=false
terraform -chdir=infra/bootstrap validate
```

Commit provider lock files. Never commit state, saved plans, credentials, or secrets.
The committed tfvars and backend files contain non-secret account configuration.

## First bootstrap deployment

Only one team member performs this initial deployment. First check that another
teammate has not already created the bucket/table. If they exist, obtain the
existing state and join it; do not create competing local state or blindly import.

```powershell
$env:AWS_PROFILE = "align"
aws sts get-caller-identity
terraform -chdir=infra/bootstrap init
terraform -chdir=infra/bootstrap plan '-var-file=bootstrap.tfvars' '-out=bootstrap.tfplan'
terraform -chdir=infra/bootstrap apply bootstrap.tfplan
terraform -chdir=infra/bootstrap output
```

Confirm account 183754987292. Review the plan before applying. Expected resources:
one S3 bucket, its public-access block, ownership controls, versioning, encryption,
HTTPS-only policy, and one DynamoDB table. Nothing in this root creates EC2 or RDS.

The bucket name is align-tfstate-183754987292-us-east-2. S3 names are globally unique;
if unavailable, agree on a replacement and update all backend configurations.

## Move bootstrap state to S3

After the first apply, allow S3 versioning to propagate before the first state
upload (AWS recommends waiting 15 minutes after initially enabling versioning).

```powershell
Copy-Item infra/bootstrap/backend.tf.example infra/bootstrap/backend.tf
terraform -chdir=infra/bootstrap init -migrate-state '-backend-config=backend.hcl'
terraform -chdir=infra/bootstrap plan '-var-file=bootstrap.tfvars'
```

Answer yes to migrating the existing local state. Do not use -reconfigure for this
first migration: the local state must be copied. Confirm the plan shows no changes
and the remote object exists before retiring any local state backup.

backend.tf is ignored intentionally: a fresh clone must remain usable for the
first bootstrap. The example and this explicit migration procedure are committed.

## Joining an already bootstrapped account

After the owner confirms state has been migrated:

```powershell
$env:AWS_PROFILE = "align"
Copy-Item infra/bootstrap/backend.tf.example infra/bootstrap/backend.tf
terraform -chdir=infra/bootstrap init '-backend-config=backend.hcl'
terraform -chdir=infra/bootstrap plan '-var-file=bootstrap.tfvars'
```

Do not perform a local-state apply on a second machine.

## State and locking

All roots use the same bucket, with these separate keys:

| Root | Key |
|---|---|
| Bootstrap | align/bootstrap/terraform.tfstate |
| Dev | align/dev/terraform.tfstate |
| Prod | align/prod/terraform.tfstate |

Backends enable native S3 locking and the requested DynamoDB lock table.
DynamoDB locking is deprecated, but HashiCorp supports both settings together for
migration. Everyone must use the committed settings. A future coordinated change
can retire the DynamoDB setting/table; do not change locking per developer.

Backend access needs S3 ListBucket, GetObject/PutObject on the state's key, and
GetObject/PutObject/DeleteObject on its .tflock key, plus DynamoDB
DescribeTable/GetItem/PutItem/DeleteItem on the lock table. These files do not
grant IAM access; team identities need those permissions separately.
The bootstrap creator also needs permissions to create/manage the declared resources.

## Naming and future module contract

Use snake_case Terraform identifiers. AWS resource names follow
align-{environment}-{component}. Tags: Project=align, Environment=dev/prod/bootstrap,
ManagedBy=Terraform. Credentials must not be module inputs.

Shared inputs: project, environment, aws_region, aws_account_id.
Network inputs: vpc_cidr, availability_zones, public_subnet_cidrs,
private_subnet_cidrs, allowed_ssh_cidrs, db_port (PostgreSQL: 5432).
Future network outputs: vpc_id, public_subnet_ids, private_subnet_ids,
ec2_security_group_id, rds_security_group_id.

Preserve subnet list order matching availability_zones. Environment roots will
pass module.network outputs directly to compute/database modules and expose the
same outputs for inspection. Never hardcode deployed IDs into configuration.

## Part 5 boundary

Production destruction targets envs/prod only. The state bucket and lock table
belong to bootstrap state and must survive. prevent_destroy on both resources and
force_destroy=false on the bucket add guards, but do not prevent deletion through
AWS console/API or after removing the resource configuration.

Application data buckets and retained RDS snapshots require their own backup and
recovery plan. Terraform state is not a backup of database rows or experiment logs.

## References

- [S3 backend and locking](https://developer.hashicorp.com/terraform/language/backend/s3)
- [S3 versioning propagation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html)
