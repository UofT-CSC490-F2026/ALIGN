# Add Terraform foundation and remote-state bootstrap

ALIGN needs state storage that survives the production disaster-recovery demo.
This change introduces an independent bootstrap root for account 183754987292 in
us-east-2, plus the dev/prod backend configurations and the future network module
interface. No environment resources are deployed by this PR.

## Changes

- Versioned, AES256-encrypted state bucket with public access blocked, ACLs disabled,
  and an HTTPS-only policy.
- On-demand DynamoDB lock table with LockID string key; backend files enable both
  the required table and native S3 locking. DynamoDB deprecation is documented.
- Account allowlist and prevent_destroy guards on the bucket and table.
- Separate bootstrap/dev/prod state keys and first-deployment/migration instructions.
- Terraform 1.16.x constraint and AWS provider 6.67.0 lock file.
- Git exclusions for state, plans, provider downloads, and local secrets.
- Documented naming, variables, module ownership, and future output interface.

## Verification

- Terraform init downloaded the signed AWS provider successfully.
- Terraform validate passed.
- AWS STS confirmed the intended account and IAM user jinhao.
- Terraform plan succeeded: **7 add, 0 change, 0 destroy**.
- No terraform apply has been run; provisioning permissions and bucket availability
  are not proven by this initial plan. Remote-state migration is pending deployment.

## Review order

1. bootstrap/main.tf: resources and protection settings.
2. bootstrap/providers.tf and bootstrap.tfvars: account and region.
3. bootstrap/backend.hcl and envs/*/backend.hcl: independent state keys.
4. README.md: deployment, migration, and teammate handoff.

The network implementation and environment wiring follow in PRs 2 and 3.
Existing a2/ pipeline files are outside this change.
