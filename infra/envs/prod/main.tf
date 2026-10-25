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
  backup_retention_days = 1

  multi_az                  = false
  deletion_protection       = var.db_deletion_protection
  snapshot_identifier       = var.db_snapshot_identifier
  final_snapshot_identifier = var.db_final_snapshot_identifier
}

module "storage" {
  source = "../../modules/storage"

  project        = var.project
  environment    = var.environment
  aws_region     = var.aws_region
  aws_account_id = var.aws_account_id

  backup_bucket_name = "${var.project}-${var.environment}-backup-${var.aws_account_id}"
  lake_force_destroy = var.lake_force_destroy
}

module "compute" {
  source = "../../modules/compute"

  project     = var.project
  environment = var.environment
  aws_region  = var.aws_region

  subnet_id             = module.network.public_subnet_ids[0]
  ec2_security_group_id = module.network.ec2_security_group_id
  lake_bucket_name      = module.storage.lake_bucket_name
  lake_bucket_arn       = module.storage.lake_bucket_arn
  db_secret_arn         = module.database.db_master_secret_arn
  ami_id                = "ami-0d31a22b09ceec4db"
  instance_type         = "t3.micro"
  root_volume_size      = 20
  arbiter_enabled       = false
  smoke_test_enabled    = false
}
