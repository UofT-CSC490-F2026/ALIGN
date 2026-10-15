bucket              = "align-tfstate-710809468662-us-east-2"
key                 = "align/prod/terraform.tfstate"
region              = "us-east-2"
allowed_account_ids = ["710809468662"]
encrypt             = true
use_lockfile        = true
dynamodb_table      = "align-terraform-locks"
