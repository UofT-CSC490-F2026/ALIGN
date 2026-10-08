project              = "align"
environment          = "dev"
aws_account_id       = "183754987292"
aws_region           = "us-east-2"
vpc_cidr             = "10.10.0.0/16"
availability_zones   = ["us-east-2a", "us-east-2b"]
public_subnet_cidrs  = ["10.10.1.0/24", "10.10.2.0/24"]
private_subnet_cidrs = ["10.10.11.0/24", "10.10.12.0/24"]
# Session Manager is the intended access method; keep inbound SSH closed.
# Compute must configure SSM Agent, its instance role, and outbound connectivity.
allowed_ssh_cidrs = []
db_port           = 5432
