# Run: terraform test -var-file=prod.tfvars
# Mocked configuration regression, not a comparison against live AWS state.
mock_provider "aws" {
  mock_data "aws_s3_bucket" {
    defaults = { arn = "arn:aws:s3:::align-prod-backup-710809468662" }
  }
}

override_module {
  target  = module.compute
  outputs = { instance_id = "i-0123456789abcdef0" }
}

run "canonical_prod_retains_recovery_settings" {
  command = plan
  assert {
    condition = (
      var.db_snapshot_identifier == "a2-snapshot-before-disaster" &&
      var.db_deletion_protection &&
      var.db_final_snapshot_identifier == "align-prod-final-after-restore-20261022-01" &&
      !var.lake_force_destroy
    )
    error_message = "prod.tfvars alone must retain the restored DB source and safeguards."
  }
}

run "reject_empty_prod_snapshot" {
  command = plan
  variables { db_snapshot_identifier = "" }
  expect_failures = [var.db_snapshot_identifier]
}
