bucket              = "align-tfstate-183754987292-us-east-2"
key                 = "align/backup/terraform.tfstate"
region              = "us-east-2"
allowed_account_ids = ["183754987292"]
encrypt             = true
use_lockfile        = true
dynamodb_table      = "align-terraform-locks"
