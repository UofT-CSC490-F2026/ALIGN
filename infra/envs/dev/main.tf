module "network" {
  source               = "../../modules/network"
  project              = var.project
  environment          = var.environment
  aws_region           = var.aws_region
  vpc_cidr             = var.vpc_cidr
  availability_zones   = var.availability_zones
  public_subnet_cidrs  = var.public_subnet_cidrs
  private_subnet_cidrs = var.private_subnet_cidrs
  allowed_ssh_cidrs    = var.allowed_ssh_cidrs
  db_port              = var.db_port
}
