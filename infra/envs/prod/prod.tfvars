# This filename is not auto-loaded by Terraform; pass -var-file when planning.
# Run from infra/envs/prod with your AWS credentials configured:
# terraform init -backend-config=backend.hcl
# terraform plan -var-file=prod.tfvars -out=prod.tfplan
# Review the plan before applying; the saved plan includes these variables.
# terraform apply prod.tfplan

project              = "align"
environment          = "prod"
aws_account_id       = "183754987292"
aws_region           = "us-east-2"
vpc_cidr             = "10.20.0.0/16"
availability_zones   = ["us-east-2a", "us-east-2b"]
public_subnet_cidrs  = ["10.20.1.0/24", "10.20.2.0/24"]
private_subnet_cidrs = ["10.20.11.0/24", "10.20.12.0/24"]
# Session Manager is the intended access method; keep inbound SSH closed.
# Compute must configure SSM Agent, its instance role, and outbound connectivity.
allowed_ssh_cidrs = []
db_port           = 5432
