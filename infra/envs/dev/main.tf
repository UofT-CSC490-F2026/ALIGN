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

module "database" {
  source = "../../modules/database"

  project     = var.project
  environment = var.environment
  aws_region  = var.aws_region

  private_subnet_ids    = module.network.private_subnet_ids
  rds_security_group_id = module.network.rds_security_group_id

  db_name = "align"
  db_port = var.db_port

  instance_class        = "db.t4g.micro"
  allocated_storage     = 20
  backup_retention_days = 7

  multi_az            = false
  deletion_protection = false
}

module "storage" {
  source = "../../modules/storage"

  project        = var.project
  environment    = var.environment
  aws_region     = var.aws_region
  aws_account_id = var.aws_account_id

  backup_bucket_name = "${var.project}-${var.environment}-backup-${var.aws_account_id}"
}
