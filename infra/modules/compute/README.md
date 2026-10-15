# Compute module

This module defines EC2, Docker bootstrap, its instance role/profile, an LLM secret
entry, and ECR image delivery. It consumes networking, lake storage, and database
secret outputs. Environment roots inherit the AWS provider into this module.

EC2 runs a published `linux/amd64` image. It does not clone Git or build the arbiter.
The application and Dockerfile belong to the application owner. No Docker image
or registry publishing identity is created by these source files.

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

## Nightly smoke job

Use a dedicated dev/test deployment. Service and smoke modes cannot both be enabled
on one instance, and smoke is rejected for prod. This prevents the job from managing
the normal arbiter container, but separate hosts alone do not isolate database
state, S3 objects, or shared LLM quotas: the team must supply test dependencies too.

Before enabling smoke, provide `smoke_test_command` as arguments accepted by the
image ENTRYPOINT and measured `smoke_test_cpus` / `smoke_test_memory_mb` limits.
For example, `["smoke"]` is appropriate only if the real image supports that
subcommand; it is not implemented by this infrastructure.

The default schedule is daily at 03:00 America/Toronto with a 900-second timeout.
Bootstrap installs cronie and writes `/etc/cron.d/align-smoke`. The job holds a
nonblocking lock, refreshes ECR authentication, pulls the pinned image, and starts a
uniquely named container with CPU/memory limits. It records the application's exit
status and removes only that run's container, including on timeout. It never
updates, stops, or replaces `align-arbiter`. Images are not automatically pruned.

The runner is `/opt/align/run_container.py`; its non-secret configuration is
`/etc/align/config.json`. Logs are `/var/log/align-service.log` and
`/var/log/align-smoke.log`, rotated by the runner. Docker container logs also have
size limits. The application must avoid printing secret values to its own logs.

## Runtime configuration contract

The image must support these environment variables or provide an adapter:

- `AWS_REGION` and `AWS_DEFAULT_REGION`
- `ALIGN_ENVIRONMENT`
- `ALIGN_LAKE_BUCKET`
- `ALIGN_DB_SECRET_ARN` and `ALIGN_LLM_SECRET_ARN`
- `ALIGN_RUN_ID` and `ALIGN_OUTPUT_PREFIX` for smoke runs

Additional **non-secret** settings may use `container_environment`. Managed settings
take precedence. The image retrieves secret values with its AWS SDK and instance
role. No secret values or static AWS access keys belong in Terraform, user data,
image layers, or these environment settings. ECR login credentials exist only in
the runner's temporary directory while preparing the image.

## Validation and deployment dependencies

Formatting, Terraform validation, shell/Python syntax, and mocked plans can be
checked without arbiter code. Docker behavior can later be exercised with a small
fixture image. Actual Docker execution requires a running Docker engine; ECR,
instance IAM, networking, and bootstrap require a separately approved dev test.
The repository currently contains no real arbiter image.

Before deployment, obtain the pinned AL2023 x86_64 AMI, instance type and disk size;
wire the networking/lake outputs; obtain the database secret ARN and any customer
KMS key ARN/key-policy access; configure AWS authentication and state access; and
arrange secure population of the LLM secret before the application runs.

Confirm who publishes the image, its entrypoint and test arguments, and whether the
dev deployment and its dependencies are isolated from regular use. The module is
not yet wired into either environment root. All applies require Jerry's explicit
approval. Changing bootstrap/runtime configuration plans EC2 replacement, deleting
its root disk; persistent application data belongs in the lake or database.
