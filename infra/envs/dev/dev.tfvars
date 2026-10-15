# This filename is not auto-loaded by Terraform; pass -var-file when planning.
# Run from infra/envs/dev with your AWS credentials configured:
# terraform init -backend-config=backend.hcl
# terraform plan -var-file=dev.tfvars -out=dev.tfplan
# Review the plan before applying; the saved plan includes these variables.
# terraform apply dev.tfplan

project              = "align"
environment          = "dev"
aws_account_id       = "710809468662"
aws_region           = "us-east-2"
vpc_cidr             = "10.10.0.0/16"
availability_zones   = ["us-east-2a", "us-east-2b"]
public_subnet_cidrs  = ["10.10.1.0/24", "10.10.2.0/24"]
private_subnet_cidrs = ["10.10.11.0/24", "10.10.12.0/24"]
# Session Manager is the intended access method; keep inbound SSH closed.
# Compute must configure SSM Agent, its instance role, and outbound connectivity.
allowed_ssh_cidrs = []
db_port           = 5432
