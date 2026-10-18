locals {
  name = "${var.project}-${var.environment}"

  tags = {
    project   = var.project
    env       = var.environment
    ManagedBy = "Terraform"
  }

  # RDS requires a name for the final snapshot when skip_final_snapshot=false.
  # A later destroy/recreate may require a new identifier if this snapshot exists.
  final_snapshot_identifier = var.final_snapshot_identifier != null ? var.final_snapshot_identifier : "${local.name}-postgres-final"
}

# The network module already owns the VPC and its private subnets.
resource "aws_db_subnet_group" "this" {
  region      = var.aws_region
  name        = "${local.name}-db-subnets"
  description = "Private subnets for ALIGN ${var.environment} RDS"
  subnet_ids  = var.private_subnet_ids

  tags = merge(local.tags, { Name = "${local.name}-db-subnets" })
}

# Managed PostgreSQL. EC2-to-RDS port 5432 is already allowed by the
# existing network module's RDS security group. Do not create another one.
resource "aws_db_instance" "this" {
  region     = var.aws_region
  identifier = "${local.name}-postgres"

  engine         = "postgres"
  engine_version = var.postgres_major_version
  instance_class = var.instance_class

  db_name  = var.db_name
  username = var.master_username
  port     = var.db_port

  # RDS generates the master password and stores it in Secrets Manager;
  # no password is supplied to or stored in Terraform state.
  manage_master_user_password = true

  allocated_storage = var.allocated_storage
  storage_type      = "gp3"
  storage_encrypted = true

  db_subnet_group_name   = aws_db_subnet_group.this.name
  vpc_security_group_ids = [var.rds_security_group_id]
  publicly_accessible    = false
  multi_az               = var.multi_az

  backup_retention_period    = var.backup_retention_days
  copy_tags_to_snapshot      = true
  auto_minor_version_upgrade = true
  apply_immediately          = false

  # Keep a final snapshot; deletion protection can be explicitly disabled
  # before the course teardown via an environment-level variable override.
  skip_final_snapshot       = false
  final_snapshot_identifier = local.final_snapshot_identifier
  deletion_protection       = var.deletion_protection

  tags = merge(local.tags, { Name = "${local.name}-postgres" })
}
