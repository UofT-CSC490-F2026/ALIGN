mock_provider "aws" {}
variables {
  project               = "align"
  environment           = "prod"
  aws_region            = "us-east-2"
  private_subnet_ids    = ["subnet-0123456789abcdef0", "subnet-0123456789abcdef1"]
  rds_security_group_id = "sg-0123456789abcdef0"
}

run "restore_from_explicit_snapshot" {
  command = plan
  variables {
    environment         = "prod"
    snapshot_identifier = "a2-snapshot-before-disaster"
  }
  assert {
    condition = (
      aws_db_instance.this.snapshot_identifier == "a2-snapshot-before-disaster" &&
      aws_db_instance.this.manage_master_user_password &&
      aws_db_instance.this.deletion_protection &&
      !aws_db_instance.this.publicly_accessible &&
      aws_db_instance.this.storage_encrypted &&
      !aws_db_instance.this.skip_final_snapshot
    )
    error_message = "Restore must select the snapshot and preserve password management, private access and deletion safeguards."
  }
}
run "reject_empty_snapshot" {
  command = plan
  variables { snapshot_identifier = "" }
  expect_failures = [var.snapshot_identifier]
}
