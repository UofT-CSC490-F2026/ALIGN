# Mocked plan tests: no AWS API calls and no actual RDS resources created.
mock_provider "aws" {}

variables {
  project               = "align"
  environment           = "dev"
  aws_region            = "us-east-2"
  private_subnet_ids    = ["subnet-0123456789abcdef0", "subnet-0123456789abcdef1"]
  rds_security_group_id = "sg-0123456789abcdef0"
}

run "secure_dev_configuration" {
  command = plan

  assert {
    condition     = toset(aws_db_subnet_group.this.subnet_ids) == toset(var.private_subnet_ids)
    error_message = "DB subnet group must use the two network-provided subnet IDs."
  }

  assert {
    condition = (
      aws_db_instance.this.db_subnet_group_name == aws_db_subnet_group.this.name &&
      toset(aws_db_instance.this.vpc_security_group_ids) == toset([var.rds_security_group_id])
    )
    error_message = "RDS must use the subnet group and existing network security group."
  }

  assert {
    condition     = aws_db_instance.this.publicly_accessible == false && aws_db_instance.this.storage_encrypted == true
    error_message = "RDS must be private and encrypt storage."
  }

  assert {
    condition = (
      aws_db_instance.this.manage_master_user_password == true &&
      aws_db_instance.this.backup_retention_period == 7 &&
      aws_db_instance.this.skip_final_snapshot == false
    )
    error_message = "RDS must manage its password, retain seven days of backups, and take a final snapshot."
  }

  assert {
    condition = (
      aws_db_instance.this.engine == "postgres" &&
      aws_db_instance.this.engine_version == "16" &&
      aws_db_instance.this.instance_class == "db.t4g.micro" &&
      aws_db_instance.this.multi_az == false &&
      aws_db_instance.this.deletion_protection == false
    )
    error_message = "Development database settings do not match the expected baseline."
  }
}

run "production_requires_deletion_protection" {
  command = plan

  variables {
    environment = "prod"
  }

  assert {
    condition = (
      aws_db_instance.this.deletion_protection == true &&
      aws_db_instance.this.identifier == "align-prod-postgres"
    )
    error_message = "Prod RDS must have a distinct identifier and deletion protection."
  }
}

run "reject_duplicate_private_subnets" {
  command = plan
  variables { private_subnet_ids = ["subnet-0123456789abcdef0", "subnet-0123456789abcdef0"] }
  expect_failures = [var.private_subnet_ids]
}

run "reject_single_private_subnet" {
  command = plan
  variables { private_subnet_ids = ["subnet-0123456789abcdef0"] }
  expect_failures = [var.private_subnet_ids]
}

run "reject_zero_backup_retention" {
  command = plan
  variables { backup_retention_days = 0 }
  expect_failures = [var.backup_retention_days]
}

run "reject_too_little_gp3_storage" {
  command = plan
  variables { allocated_storage = 10 }
  expect_failures = [var.allocated_storage]
}

run "reject_invalid_database_port" {
  command = plan
  variables { db_port = 65536 }
  expect_failures = [var.db_port]
}
