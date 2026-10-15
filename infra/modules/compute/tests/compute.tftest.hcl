# These are fictitious fixtures used only with the mocked provider.
# All runs use plan; no AWS credentials or cloud resource changes are involved.
mock_provider "aws" {
  override_during = plan

  mock_data "aws_partition" {
    defaults = { partition = "aws" }
  }

  mock_data "aws_ami" {
    defaults = {
      id                  = "ami-00000000000000000"
      architecture        = "x86_64"
      name                = "al2023-ami-2023.0.00000000.0-kernel-6.1-x86_64"
      root_device_type    = "ebs"
      root_device_name    = "/dev/xvda"
      virtualization_type = "hvm"
      block_device_mappings = [{
        device_name = "/dev/xvda"
        ebs         = { volume_size = "8" }
      }]
    }
  }

  mock_data "aws_ec2_instance_type" {
    defaults = {
      supported_architectures        = ["x86_64"]
      supported_root_device_types    = ["ebs"]
      supported_virtualization_types = ["hvm"]
    }
  }

  mock_resource "aws_instance" {
    defaults = { id = "i-00000000000000000" }
  }

  mock_resource "aws_ecr_repository" {
    defaults = {
      arn            = "arn:aws:ecr:us-east-2:000000000000:repository/mock-arbiter"
      repository_url = "000000000000.dkr.ecr.us-east-2.amazonaws.com/mock-arbiter"
    }
  }

  mock_data "aws_ecr_repository" {
    defaults = {
      arn            = "arn:aws:ecr:us-east-2:000000000000:repository/mock-existing"
      repository_url = "000000000000.dkr.ecr.us-east-2.amazonaws.com/mock-existing"
    }
  }

  mock_resource "aws_iam_role" {
    defaults = { arn = "arn:aws:iam::000000000000:role/mock-compute" }
  }

  mock_resource "aws_secretsmanager_secret" {
    defaults = { arn = "arn:aws:secretsmanager:us-east-2:000000000000:secret:mock-llm-api-key-ABC123" }
  }
}

variables {
  project               = "align"
  environment           = "dev"
  aws_region            = "us-east-2"
  subnet_id             = "subnet-00000000000000000"
  ec2_security_group_id = "sg-00000000000000000"
  lake_bucket_name      = "align-dev-test-lake"
  lake_bucket_arn       = "arn:aws:s3:::align-dev-test-lake"
  db_secret_arn         = "arn:aws:secretsmanager:us-east-2:000000000000:secret:mock-database-ABC123"
  ami_id                = "ami-00000000000000000"
  instance_type         = "t3.micro"
  root_volume_size      = 20
}

run "scoped_workload_permissions" {
  command = plan

  assert {
    condition = (
      length(jsondecode(aws_iam_role.compute.assume_role_policy).Statement) == 1 &&
      jsondecode(aws_iam_role.compute.assume_role_policy).Statement[0].Effect == "Allow" &&
      toset(jsondecode(aws_iam_role.compute.assume_role_policy).Statement[0].Action) == toset(["sts:AssumeRole"]) &&
      jsondecode(aws_iam_role.compute.assume_role_policy).Statement[0].Principal == { Service = "ec2.amazonaws.com" }
    )
    error_message = "Only the EC2 service may assume the compute role."
  }

  assert {
    condition = (
      length(jsondecode(aws_iam_role_policy.data_access.policy).Statement) == 3 &&
      toset(flatten([for statement in jsondecode(aws_iam_role_policy.data_access.policy).Statement : statement.Action])) == toset([
        "s3:ListBucket", "s3:GetObject", "s3:PutObject", "secretsmanager:GetSecretValue",
      ]) &&
      alltrue([for statement in jsondecode(aws_iam_role_policy.data_access.policy).Statement :
        statement.Effect == "Allow" && !can(statement.NotAction) && !can(statement.NotResource) && (
          toset(statement.Action) == toset(["s3:ListBucket"]) ? toset(statement.Resource) == toset([var.lake_bucket_arn]) :
          toset(statement.Action) == toset(["s3:GetObject", "s3:PutObject"]) ? toset(statement.Resource) == toset(["${var.lake_bucket_arn}/*"]) :
          toset(statement.Action) == toset(["secretsmanager:GetSecretValue"]) ? toset(statement.Resource) == toset([var.db_secret_arn, aws_secretsmanager_secret.llm_api_key.arn]) :
          false
        )
      ])
    )
    error_message = "Workload access must be limited to lake listing/read/write and reads of exactly the database and LLM secrets."
  }

  assert {
    condition = (
      aws_iam_instance_profile.compute.role == aws_iam_role.compute.name &&
      aws_iam_role_policy.data_access.role == aws_iam_role.compute.id &&
      aws_iam_role_policy_attachment.ssm.role == aws_iam_role.compute.name &&
      aws_iam_role_policy_attachment.ssm.policy_arn == "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
    )
    error_message = "The instance profile and scoped workload/Session Manager policies must belong to the compute role."
  }

  assert {
    condition = (
      aws_iam_role.compute.name == "align-dev-compute" &&
      aws_iam_instance_profile.compute.name == "align-dev-compute" &&
      aws_iam_role_policy.data_access.name == "align-dev-data-access" &&
      aws_secretsmanager_secret.llm_api_key.name == "align-dev-llm-api-key" &&
      alltrue([for tags in [aws_iam_role.compute.tags, aws_iam_instance_profile.compute.tags, aws_secretsmanager_secret.llm_api_key.tags] :
        tags.project == "align" && tags.env == "dev" && tags.ManagedBy == "Terraform"
      ])
    )
    error_message = "Compute resources must follow shared environment naming and tagging conventions."
  }

  assert {
    condition = (
      aws_secretsmanager_secret.llm_api_key.region == var.aws_region &&
      aws_secretsmanager_secret.llm_api_key.recovery_window_in_days == 30 &&
      aws_secretsmanager_secret.llm_api_key.kms_key_id == null &&
      output.iam_role_arn == aws_iam_role.compute.arn &&
      output.llm_secret_arn == aws_secretsmanager_secret.llm_api_key.arn
    )
    error_message = "The LLM entry must use the configured region, default encryption, a 30-day recovery window, and correct role/secret ARN outputs."
  }
}

run "customer_managed_database_key_is_scoped" {
  command = plan
  variables {
    db_secret_kms_key_arn = "arn:aws:kms:us-east-2:000000000000:key/00000000-0000-0000-0000-000000000000"
  }

  assert {
    condition = (
      length(jsondecode(aws_iam_role_policy.data_access.policy).Statement) == 4 &&
      length([for statement in jsondecode(aws_iam_role_policy.data_access.policy).Statement : statement if contains(statement.Action, "kms:Decrypt")]) == 1 &&
      alltrue([for statement in jsondecode(aws_iam_role_policy.data_access.policy).Statement :
        statement.Effect == "Allow" &&
        toset(statement.Action) == toset(["kms:Decrypt"]) &&
        toset(statement.Resource) == toset([var.db_secret_kms_key_arn]) &&
        statement.Condition.StringEquals == {
          "kms:ViaService"                  = "secretsmanager.us-east-2.amazonaws.com"
          "kms:EncryptionContext:SecretARN" = var.db_secret_arn
        }
        if contains(statement.Action, "kms:Decrypt")
      ])
    )
    error_message = "KMS decryption must be restricted to the supplied key and database secret through Secrets Manager."
  }
}

run "prod_names_are_isolated" {
  command = plan
  variables { environment = "prod" }

  assert {
    condition = (
      aws_iam_role.compute.name == "align-prod-compute" &&
      aws_iam_instance_profile.compute.name == "align-prod-compute" &&
      aws_iam_role_policy.data_access.name == "align-prod-data-access" &&
      aws_secretsmanager_secret.llm_api_key.name == "align-prod-llm-api-key" &&
      aws_ecr_repository.arbiter[0].name == "align-prod-arbiter" &&
      aws_instance.compute.tags.Name == "align-prod-compute" &&
      aws_instance.compute.root_block_device[0].tags.Name == "align-prod-compute-root" &&
      alltrue([for tags in [aws_iam_role.compute.tags, aws_iam_instance_profile.compute.tags, aws_secretsmanager_secret.llm_api_key.tags, aws_instance.compute.tags, aws_instance.compute.root_block_device[0].tags] :
        tags.env == "prod" && tags.project == "align"
      ])
    )
    error_message = "Production names and tags must distinguish its role, profile, policy, secret, instance, and disk from dev."
  }
}

run "ec2_bootstrap_contract" {
  command = plan

  assert {
    condition = (
      aws_instance.compute.region == var.aws_region &&
      aws_instance.compute.ami == var.ami_id &&
      aws_instance.compute.instance_type == var.instance_type &&
      aws_instance.compute.subnet_id == var.subnet_id &&
      aws_instance.compute.vpc_security_group_ids == toset([var.ec2_security_group_id]) &&
      aws_instance.compute.iam_instance_profile == aws_iam_instance_profile.compute.name &&
      aws_instance.compute.associate_public_ip_address
    )
    error_message = "EC2 must use the supplied AMI/type/network and compute instance profile, with explicit public-IP assignment."
  }

  assert {
    condition = (
      aws_instance.compute.metadata_options[0].http_endpoint == "enabled" &&
      aws_instance.compute.metadata_options[0].http_tokens == "required" &&
      aws_instance.compute.metadata_options[0].http_put_response_hop_limit == 2 &&
      aws_instance.compute.metadata_options[0].instance_metadata_tags == "disabled"
    )
    error_message = "EC2 must require IMDSv2 while permitting token responses across the Docker bridge."
  }

  assert {
    condition = (
      aws_instance.compute.root_block_device[0].encrypted &&
      aws_instance.compute.root_block_device[0].volume_type == "gp3" &&
      aws_instance.compute.root_block_device[0].volume_size == var.root_volume_size &&
      aws_instance.compute.root_block_device[0].delete_on_termination &&
      aws_instance.compute.tags.Name == "align-dev-compute" &&
      alltrue([for tags in [aws_instance.compute.tags, aws_instance.compute.root_block_device[0].tags] :
        tags.env == "dev" && tags.project == "align" && tags.ManagedBy == "Terraform"
      ])
    )
    error_message = "EC2 and its encrypted gp3 root disk must follow the requested size, lifecycle, and shared tags."
  }

  assert {
    condition = (
      aws_instance.compute.user_data_replace_on_change &&
      contains([local.bootstrap_user_data, sha1(local.bootstrap_user_data)], aws_instance.compute.user_data) &&
      output.instance_id == aws_instance.compute.id
    )
    error_message = "EC2 must receive the checked-in bootstrap, replace on bootstrap changes, and expose its instance ID."
  }
}

run "image_delivery_and_disabled_defaults" {
  command = plan

  assert {
    condition = (
      length(aws_ecr_repository.arbiter) == 1 &&
      aws_ecr_repository.arbiter[0].name == "align-dev-arbiter" &&
      aws_ecr_repository.arbiter[0].image_tag_mutability == "IMMUTABLE" &&
      !aws_ecr_repository.arbiter[0].force_delete &&
      aws_ecr_repository.arbiter[0].encryption_configuration[0].encryption_type == "AES256" &&
      aws_ecr_repository.arbiter[0].image_scanning_configuration[0].scan_on_push &&
      !local.runtime_config.service_enabled && !local.runtime_config.smoke_enabled &&
      local.runtime_config.image == "" && local.smoke_cron == "# ALIGN smoke tests are disabled.\n"
    )
    error_message = "ECR must use immutable tags, encryption and scanning; application workloads must remain disabled until configured."
  }

  assert {
    condition = (
      length(jsondecode(aws_iam_role_policy.ecr_pull.policy).Statement) == 2 &&
      alltrue([for statement in jsondecode(aws_iam_role_policy.ecr_pull.policy).Statement :
        statement.Effect == "Allow" && (
          toset(statement.Action) == toset(["ecr:GetAuthorizationToken"]) ? statement.Resource == "*" :
          toset(statement.Action) == toset(["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]) ? toset(statement.Resource) == toset([output.ecr_repository_arn]) : false
        )
      ]) && output.ecr_repository_url == aws_ecr_repository.arbiter[0].repository_url
    )
    error_message = "EC2 may authenticate to ECR and pull from exactly the selected repository, with no image publishing permissions."
  }
}

run "use_existing_ecr_repository" {
  command = plan
  variables { existing_ecr_repository_name = "mock-existing" }

  assert {
    condition     = length(aws_ecr_repository.arbiter) == 0 && output.ecr_repository_arn == data.aws_ecr_repository.existing[0].arn
    error_message = "An existing repository must be consumed rather than recreated."
  }
}

run "dedicated_dev_smoke_schedule" {
  command = plan
  variables {
    smoke_test_enabled   = true
    arbiter_image_digest = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    smoke_test_command   = ["mock-smoke"]
    smoke_test_cpus      = 0.5
    smoke_test_memory_mb = 128
  }

  assert {
    condition = (
      local.runtime_config.smoke_enabled && !local.runtime_config.service_enabled &&
      strcontains(local.smoke_cron, "CRON_TZ=America/Toronto\n0 3 * * * root /usr/bin/python3 /opt/align/run_container.py smoke\n") &&
      local.runtime_config.smoke_timeout_seconds == 900 &&
      local.runtime_config.environment.ALIGN_DB_SECRET_ARN == var.db_secret_arn &&
      local.runtime_config.environment.ALIGN_LLM_SECRET_ARN == output.llm_secret_arn
    )
    error_message = "The dedicated smoke deployment must receive its daily schedule, bounded timeout, and secret ARNs only."
  }
}

run "reject_service_and_smoke_on_same_instance" {
  command = plan
  variables {
    arbiter_enabled      = true
    smoke_test_enabled   = true
    arbiter_image_digest = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    smoke_test_command   = ["mock-smoke"]
    smoke_test_cpus      = 0.5
    smoke_test_memory_mb = 128
  }
  expect_failures = [var.arbiter_enabled]
}

run "reject_smoke_on_prod" {
  command = plan
  variables {
    environment          = "prod"
    smoke_test_enabled   = true
    arbiter_image_digest = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
    smoke_test_command   = ["mock-smoke"]
    smoke_test_cpus      = 0.5
    smoke_test_memory_mb = 128
  }
  expect_failures = [var.smoke_test_enabled]
}

run "reject_arm_instance_type" {
  command = plan
  variables { instance_type = "t4g.micro" }
  override_data {
    target = data.aws_ec2_instance_type.selected
    values = {
      supported_architectures        = ["arm64"]
      supported_root_device_types    = ["ebs"]
      supported_virtualization_types = ["hvm"]
    }
  }
  expect_failures = [aws_instance.compute]
}

run "reject_other_operating_system" {
  command = plan
  override_data {
    target = data.aws_ami.selected
    values = {
      id                  = "ami-00000000000000000"
      architecture        = "x86_64"
      name                = "mock-ubuntu-image"
      root_device_type    = "ebs"
      root_device_name    = "/dev/xvda"
      virtualization_type = "hvm"
      block_device_mappings = [{
        device_name = "/dev/xvda"
        ebs         = { volume_size = "8" }
      }]
    }
  }
  expect_failures = [aws_instance.compute]
}

run "reject_disk_smaller_than_ami" {
  command = plan
  variables { root_volume_size = 4 }
  expect_failures = [aws_instance.compute]
}

run "reject_wildcard_lake" {
  command = plan
  variables { lake_bucket_arn = "arn:aws:s3:::*" }
  expect_failures = [var.lake_bucket_arn]
}

run "reject_mismatched_lake_name" {
  command = plan
  variables { lake_bucket_name = "different-test-lake" }
  expect_failures = [var.lake_bucket_arn]
}

run "reject_wildcard_database_secret" {
  command = plan
  variables { db_secret_arn = "arn:aws:secretsmanager:us-east-2:000000000000:secret:*" }
  expect_failures = [var.db_secret_arn]
}

run "reject_database_secret_in_other_region" {
  command = plan
  variables { db_secret_arn = "arn:aws:secretsmanager:us-east-1:000000000000:secret:mock-database-ABC123" }
  expect_failures = [var.db_secret_arn]
}

run "reject_wildcard_database_key" {
  command = plan
  variables { db_secret_kms_key_arn = "arn:aws:kms:us-east-2:000000000000:key/*" }
  expect_failures = [var.db_secret_kms_key_arn]
}

run "reject_database_key_in_other_region" {
  command = plan
  variables { db_secret_kms_key_arn = "arn:aws:kms:us-east-1:000000000000:key/00000000-0000-0000-0000-000000000000" }
  expect_failures = [var.db_secret_kms_key_arn]
}
