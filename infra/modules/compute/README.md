# Compute module

This module defines EC2, Docker bootstrap, its instance role/profile, an LLM secret
entry, and ECR image delivery. It consumes networking, lake storage, and database
secret outputs. Environment roots inherit the AWS provider into this module.

EC2 runs a published `linux/amd64` image. It does not clone Git or build the arbiter.
The application and Dockerfile belong to the application owner. Terraform does
not build or publish an image or create a registry publishing identity.

## Naming separate compute deployments

`deployment_suffix` optionally distinguishes multiple compute module instances in
the same environment. It defaults to `""`, preserving all existing names. A
non-empty suffix accepts 1-20 lowercase letters, digits or hyphens, starting and
ending with a letter or digit.

For `project = "align"` and `environment = "dev"`:

| Suffix | IAM role / instance profile | Managed ECR repository | LLM secret entry |
| --- | --- | --- | --- |
| `""` (default) | `align-dev-compute` | `align-dev-arbiter` | `align-dev-llm-api-key` |
| `"service"` | `align-dev-service-compute` | `align-dev-service-arbiter` | `align-dev-service-llm-api-key` |
| `"smoke"` | `align-dev-smoke-compute` | `align-dev-smoke-arbiter` | `align-dev-smoke-llm-api-key` |

The suffix also applies to owned inline policy names and EC2/root-volume Name
tags. The required `project` and `env` tags retain their original values. The
maximum generated IAM role name is 54 characters, within the
[64-character IAM limit](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_iam-quotas.html).

Choose unique suffixes across module instances and state files, and keep them
stable after deployment. Changing a suffix renames resources and can require
replacement, including the secret and managed ECR repository. Existing images or
a secret recovery window can complicate replacement; review the plan first.

Each module instance still defines one EC2 instance. A second host requires a
second module call in an environment root; a suffix alone does not create one.
In dev, one module call can run both the regular arbiter and an offline nightly
smoke container. Keep dev and prod in separate environment states; smoke remains
forbidden in prod. A second compute module is not needed for nightly testing.
The suffix remains optional for other independently named deployments. To consume
an already-created shared image repository, use `existing_ecr_repository_name`;
the suffix does not rename or recreate that repository.

## Deployment account

The dev/prod roots now target AWS account `710809468662` in `us-east-2`.
Nothing was deployed to the previous account, so this is preparation for the first
deployment; no existing compute resources or Terraform state need transferring.

The environment root's AWS provider authenticates to the deployment account and
checks its `allowed_account_ids`. Changing an account ID in configuration does not
authenticate your local AWS CLI/Terraform session. Compute inherits that provider
and reads its account with `aws_caller_identity`; no account ID is hardcoded into
the module. IAM, ECR and the LLM secret entry are created in that account.

The database secret and any supplied KMS key must belong to that same account;
Terraform rejects mismatches before attaching the workload policy. Lake IAM
permissions also require `aws:ResourceAccount` to match the provider's account,
because an S3 bucket ARN contains no account ID. This condition applies only to
the scoped lake grants. Use networking/storage/database outputs from this
deployment when wiring the module, and publish the image to its ECR repository.
See [AWS's bucket-owner condition documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/amazon-s3-policy-keys.html).

The environment files are named `dev.tfvars` and `prod.tfvars`, so pass the matching
`-var-file` explicitly when planning from an environment root. Those files are not
automatically loaded. The state bucket is now
`align-tfstate-710809468662-us-east-2`; its bootstrap and locking resources must be
ready before initializing the dev/prod backends. This module configures neither
the backend nor the local AWS login.

## Image delivery and enabling workloads

By default, Terraform creates an encrypted, environment-prefixed ECR repository
with immutable tags and scan-on-push. Set `existing_ecr_repository_name` to consume
a repository already managed by the team instead. EC2 receives pull permissions;
the developer or CI identity that publishes images needs its own push permissions.

Both `arbiter_enabled` and `smoke_test_enabled` default to false. The ECR repository
can therefore be provisioned before an application image exists. After an approved
deployment creates the repository, the application owner builds/tests/pushes the
image. Supply its `sha256:...` digest before enabling a workload in a later approved
deployment. No image URI, digest, credentials, or application command is invented.

The normal service uses the image ENTRYPOINT/CMD and `--restart unless-stopped`.
`arbiter_command` optionally replaces CMD with an argument list; it is not a shell
command. `arbiter_port` optionally publishes that port only on host `127.0.0.1` for
SSM port forwarding. Public application ingress remains the networking owner's
responsibility. No writable host volume or Docker socket is mounted.

Service image preparation retries failed ECR authentication, Docker login and pull
commands up to five total attempts, with 5/10/20/40-second backoffs and a shared
15-minute preparation deadline. Each attempt refreshes ECR authentication.
Missing executables, Python configuration errors and interruption signals fail
immediately. Command failures can include permanent permission/image errors; after
the attempt or time limit, startup fails and requires investigation. Container
creation/start is attempted once, avoiding ambiguous duplicate launches. A failed
first boot is recorded in the bootstrap/service logs; it does not automatically
rerun cloud-init. Existing containers retain Docker's restart policy.

## Nightly smoke job

Set both `arbiter_enabled` and `smoke_test_enabled` for one shared **dev** host.
The regular container stays running; a separate, short-lived smoke container runs
nightly. Both flags still default to false. Smoke-only dev deployments are also
supported. This mode tests the image offline with fixtures packaged in it; it
does not test real database, S3 or LLM connections.

Before enabling smoke, provide `smoke_test_command` as arguments accepted by the
image ENTRYPOINT and measured `smoke_test_cpus` / `smoke_test_memory_mb` limits,
plus explicit `smoke_test_host_memory_reserve_mb` and
`smoke_test_host_disk_reserve_mb` thresholds. Shared operation also requires
`arbiter_memory_mb`, a hard service memory limit with swap disabled. Choose it
above measured service peaks; a limit that is too small can kill the service.
Terraform checks that container memory budgets plus the host reserve fit the
selected instance, that smoke CPU leaves capacity for other work, and that the
free-disk reserve is smaller than the root disk. These checks do not measure
application requirements or establish an appropriate production instance size.
For example, `["smoke"]` is appropriate only if the real image supports that
subcommand; it is not implemented by this infrastructure.

The default schedule is daily at 03:00 America/Toronto with a 900-second timeout.
Bootstrap keeps `/etc/cron.d/align-smoke` inactive while writing runtime assets and
starting the regular service. Only after successful service startup does it
publish the complete cron file and enable cron. Startup failure leaves the job
inactive. On a shared host, each smoke run also verifies the service's label,
pinned image, running state and configured memory cap. A stopped or missing
service blocks the test; the smoke path never starts, restarts or replaces it.

Shared smoke jobs inspect and reuse that service's cached image. They do not
authenticate to ECR, pull, build or prune images, and container creation uses
`--pull never`. A smoke-only host still prepares its pinned image from ECR before
running the offline container. While holding its nonblocking lock, the runner
checks Linux `MemAvailable` and free space on Docker's default `/var/lib/docker`
filesystem. Insufficient headroom produces a logged failure without starting a
test. The runner records the test's exit status and removes only its uniquely
named container, including on timeout or SIGTERM/SIGINT cancellation.

Smoke containers have no network (`--network none`), published ports, host mounts
or Docker socket. Their root filesystem is read-only; `/tmp` is a size-bounded,
non-executable tmpfs. Images with Dockerfile `VOLUME` declarations are rejected
before container creation because those declarations otherwise create writable
anonymous host volumes. CPU/memory/swap limits and lower CPU shares bound test
resource use; PID/thread count defaults to 64 (`smoke_test_pids_limit`) and `/tmp`
defaults to 32 MiB (`smoke_test_tmpfs_mb`, no larger than the smoke memory cap).
Capabilities are dropped and privilege escalation is disabled. Adjust resource
limits to the actual test image before enabling it; fixtures must fit these rules.

While holding the lock, the job first lists **all** containers labelled
`align.workload=smoke`, before any image preparation. Any leftover
container, running or stopped, blocks the new run with a logged failure. A Docker
listing error also blocks the run. This catches containers left after interrupted
runners or failed cleanup; the runner does not automatically delete previous jobs.
An operator must review those containers and explicitly remove obsolete smoke
containers before nightly execution resumes. Inspect them with
`docker ps -a --filter label=align.workload=smoke`. The check and lock apply only
to this host; they do not isolate shared application data or coordinate other hosts.

The runner is `/opt/align/run_container.py`; its non-secret configuration is
`/etc/align/config.json`. Logs are `/var/log/align-service.log` and
`/var/log/align-smoke.log`, rotated by the runner. Docker container logs also have
size limits. The application must avoid printing secret values to its own logs.

## Runtime configuration contract

The regular service must support these environment variables or provide an adapter:

- `AWS_REGION` and `AWS_DEFAULT_REGION`
- `ALIGN_ENVIRONMENT`
- `ALIGN_LAKE_BUCKET`
- `ALIGN_DB_SECRET_ARN` and `ALIGN_LLM_SECRET_ARN`

Additional **non-secret** settings may use `container_environment`. Managed settings
take precedence. The image retrieves secret values with its AWS SDK and instance
role. No secret values or static AWS access keys belong in Terraform, user data,
image layers, or these environment settings. ECR login credentials exist only in
the runner's temporary directory while preparing the image.

Offline smoke receives only `smoke_test_environment` plus managed
`ALIGN_ENVIRONMENT`, `ALIGN_SMOKE_OFFLINE=true`, `ALIGN_RUN_ID` and
`ALIGN_OUTPUT_PREFIX=smoke/<run-id>/`. Service environment settings, lake and secret
references are not automatically copied. The test command must avoid initializing
live clients and must use packaged fixture data, stdout/stderr and bounded `/tmp`
scratch space. No test results are uploaded to S3 in this offline mode.

Both containers still share a kernel, Docker daemon, physical disk and EC2 instance
role. Offline networking blocks normal access to metadata credentials and external
services from smoke, but this is not a boundary for hostile code or a kernel escape.
Use trusted images and lightweight checks. Resource limits and headroom checks
reduce interference; they cannot guarantee zero latency impact or protect against
every host/daemon failure. Real AWS integration tests need a separate explicit
design for test data, credentials and quotas.

## Validation and deployment dependencies

Formatting, Terraform validation, shell/Python syntax, and mocked plans can be
checked without arbiter code. The opt-in Docker integration suite below exercises
the runtime helpers with a small fixture image. ECR,
instance IAM, networking, and bootstrap require a separately approved dev test.
The repository currently contains no real arbiter image.

Run the command-mocked runtime tests locally without Docker or AWS credentials:
`python3 -B -m unittest discover -s infra/modules/compute/tests -p 'test_*.py' -v`.
They cover preparation retries/deadlines, shared routing, host/service guards,
offline settings and bootstrap ordering/failures;
the Docker integration tests are skipped unless explicitly enabled.

### Local Docker integration tests

`tests/docker` is a test-only BusyBox image with a long-running service and smoke
commands that pass, fail, or wait. It contains no arbiter code or credentials and
is not an image to publish or deploy. The suite builds it for `linux/amd64`, matching
the EC2 architecture, and calls the production helpers and `main()` with real Docker.
No AWS credentials or teammate resources are needed.

Use a dedicated, empty local Docker engine. The suite requires an explicit Docker
context with a Unix socket and refuses to start if that engine has any containers.
It uses a temporary Docker client configuration, selects the socket for every
command, and restores the process environment afterward. It never changes the
default Docker context. Cleanup checks this run's unique image label and image ID;
it removes only its fixture containers and image, without pruning other resources.

For an Apple Silicon Mac, Docker CLI, Buildx and Colima can provide the local
engine. Colima runs Docker inside a Linux VM. Install the tools if needed:

```sh
brew install docker colima docker-buildx
```

Start a dedicated VM without sharing the home directory, forwarding the SSH agent,
or switching the default Docker context:

```sh
colima start align-integration --vm-type vz --arch aarch64 --cpus 2 --memory 2 --disk 10 --runtime docker --mount none --ssh-agent=false --ssh-config=false --activate=false --template=false
```

From the repository root, explicitly enable the integration suite:

```sh
ALIGN_RUN_DOCKER_INTEGRATION=1 ALIGN_DOCKER_TEST_CONTEXT=colima-align-integration python3 -B -m unittest discover -s infra/modules/compute/tests -p 'test_docker_integration.py' -v -f
```

The selected engine must support running `linux/amd64` images, including emulation
on an ARM host. The first run downloads the public base image and builds the fixture.
Tests check service start/restart and idempotence, offline restrictions, resource
settings, exit codes, timeout cleanup and orphan rejection. Shared-mode `main()`
also runs through success, failure, timeout, real concurrent-process locking and
termination cleanup, plus missing/stopped/wrong-image service and
insufficient-headroom failures. The regular
fixture container's ID and start time must be preserved. Only log/lock/temp paths,
Linux memory availability and ECR service-image preparation are adapted for this
local fixture; Docker execution and file locking remain real. Stop only this VM
when finished:

```sh
colima stop align-integration
```

Recorded local run on 2026-10-16: **all 14 integration tests passed** using a
`linux/amd64` fixture on Apple Silicon, Colima 0.10.3, Docker CLI 29.9.1,
Docker Engine 29.5.2, and Buildx 0.38.0. The suite removed its fixture containers
and image after execution. All 33 offline bootstrap/runtime tests and 39 mocked
Terraform plan tests also passed; Terraform validation and formatting checks passed.

These are local runtime integration tests. They do not test ECR authentication or
pulling, IAM, EC2 user data, cron scheduling, AWS networking, S3, database access, or
the actual arbiter. They do not measure resource-limit enforcement under stress or
actual Linux/EC2 host capacity. Bootstrap unit tests execute its embedded Python
with filesystem and command adapters, not cloud-init or systemd. These tests do not
establish that shared hosting has zero performance impact. The real application
image still needs application integration tests, and AWS behavior
still needs a separately approved dev deployment.

Before deployment, obtain the pinned AL2023 x86_64 AMI, instance type and disk size;
wire this account's networking/lake outputs; obtain its database secret ARN and any
customer KMS key ARN/key-policy access; configure AWS authentication and state
access; and arrange secure population of the LLM secret before the application runs.

Confirm who publishes the image, its entrypoint and test arguments, and whether the
offline checks use fixture data and fit the measured host/container budgets. The
module is not yet wired into either environment root. All applies require Jerry's explicit
approval. Changing bootstrap/runtime configuration plans EC2 replacement, deleting
its root disk; persistent application data belongs in the lake or database.
