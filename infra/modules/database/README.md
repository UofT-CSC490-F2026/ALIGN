# ALIGN PostgreSQL database and AWS RDS

This directory contains local Docker PostgreSQL, versioned Flyway SQL migrations,
and the reusable AWS RDS Terraform module.

## Schema and migration layout

- **No database initialization scripts are mounted under `docker-entrypoint-initdb.d`.**
  Docker now only starts PostgreSQL; **Flyway explicitly applies versioned migrations**.
- `migrations/V1__schema.sql` and `V2__views.sql` preserve your tested 9-table/2-view design.
- `migrations/V3__monitoring_event_index.sql` adds one non-destructive index and comment,
  proving upgrades work without recreating the volume.
- Fixtures are separate (`fixtures/003_seed.sql`), **never included in Flyway migrations**.
- `flyway_schema_history` is an additional internal bookkeeping table, not a 10th ALIGN table.
- `scripts/verify_schema.sql` checks structure regardless of data; `verify_fixtures.sql`
  checks optional, explicitly synthetic data.

## Local PostgreSQL setup

Run these commands from `infra/modules/database`. Keeping the same Compose project
directory preserves its existing local database volume. For a **fresh** database,
Flyway installs V1, V2, and V3; for an **already Flyway-managed** database,
`migrate` applies only outstanding changes. Do not automatically baseline an
unmanaged, manually initialized database; review its schema separately.

```powershell
docker compose up -d postgres

# Fresh databases have zero domain tables; Flyway will install V1 -> V2 -> V3.
docker compose run --rm migrate migrate

docker compose run --rm migrate validate

docker compose exec -T postgres psql -U align_dev -d align -f /checks/verify_schema.sql

# OPTIONAL local-only demo data; never run this against RDS.
docker compose exec -T postgres psql -U align_dev -d align -f /fixtures/003_seed.sql

docker compose exec -T postgres psql -U align_dev -d align -f /checks/verify_fixtures.sql
```

### Isolated upgrade smoke test — preserve data across migrations

**The strongest automated lifecycle test** creates an *isolated* temporary Compose
project using host port 5434. It applies V1 and V2, inserts fixtures, applies V3
without dropping the database volume, re-checks all fixture results, and migrates
once more as a no-op. Your normal database (`localhost:5433`) is unaffected:

```powershell
.\scripts\fresh_demo_test.ps1
```

The isolated project is stopped afterward, but its volume is retained so nothing is
deleted. To run it from scratch again, remove **only** the isolated test project
volume intentionally after confirming the project name `align_phase1_smoke`.

After migrating, run `docker compose run --rm migrate migrate` **again**.
It should report that the schema is already up to date. Then check:

```powershell
docker compose run --rm migrate info
docker compose exec -T postgres psql -U align_dev -d align -c "SELECT count(*) AS tasks_remaining FROM tasks;"
docker compose exec -T postgres psql -U align_dev -d align -c "SELECT indexname FROM pg_indexes WHERE tablename='events' AND indexname='idx_events_run_event_type';"
```

Task records persist, and the new index is available, with **no `down -v`**.

## Making the next schema change

1. Add `migrations/V4__short_description.sql` (only the incremental SQL change).
2. Run `docker compose run --rm migrate migrate`.
3. Run `docker compose run --rm migrate validate` and the structure tests.
4. Do NOT edit an already-applied V1/V2/V3 migration. Flyway checks checksums of
   applied SQL scripts and will reject unexpected changes.
5. Do NOT modify the database manually if that change should be tracked in Git.

Example V4 for future work (not included):

```sql
ALTER TABLE events ADD COLUMN ingestion_source TEXT;
```

Plan any nontrivial data migration and take a backup first. `flyway repair` should
not be used to mask migration problems, and `flyway clean` is disabled.

## Notes for your current repository / later AWS

- These files are **local development configuration**; credentials are hard-coded
  solely for local Docker. Never point this Compose file at RDS.
- When RDS is ready, connect a separately configured Flyway client from an authorized
  environment to apply the same `migrations/` files. Do not run `fixtures/` there.
- Terraform provisions servers and networking; **Flyway maintains SQL schema state**.
- Optional synthetic fixtures do **not** represent real ALIGN benchmark results.
- The existing local Docker volume is left untouched by the migration process.
- This Compose file supports `ALIGN_PG_HOST_PORT` (default 5433) for safely running
  isolated test projects on a different host port.
- V1/V2 reference the exact prior v3 schema/views (with their external BEGIN/COMMIT
  wrappers removed so Flyway owns the transaction).

## Common issues

| Symptom | What it means / what to do |
|---|---|
| `Found non-empty schema(s) without schema history table` | Unmanaged legacy schema; inspect it separately before choosing a migration/baseline strategy. |
| `relation "tasks" already exists` | Tables already exist; stop and investigate Flyway history and schema compatibility before retrying. |
| `Validate failed: migration checksum mismatch` | An applied file changed; restore that migration or add a new version. |
| `Table count = 10` | 9 ALIGN domain tables + Flyway's own `flyway_schema_history`. |
| `Port 5433 already allocated` | Another Compose project/container uses the same local port. Stop that container, not the volume. |
| `fixture already present` | The seed is intentionally local-only and idempotent. |

```powershell
# Helpful diagnostics
docker compose ps
docker compose logs postgres
docker compose run --rm migrate info
```

Do not use `docker compose down -v` unless intentionally destroying **throwaway** data.

## Milestone 2: RDS integration

Terraform creates one `aws_db_subnet_group` and one `aws_db_instance`. RDS creates
and maintains its master credential in Secrets Manager; Terraform exposes only
the secret ARN, never its password. No additional VPC, subnet, security group,
ingress rule, or independently managed password secret is created here.

`infra/envs/dev/main.tf` instantiates this module with the inherited root AWS
provider, project/environment/region variables, both
`module.network.private_subnet_ids`, and `module.network.rds_security_group_id`.
The subnet IDs must be distinct and belong to two different availability zones in
the same VPC; mocked tests cannot verify their actual AWS placement. The shared
`var.db_port` keeps the database listener and network rule consistent. Network
output references establish implicit dependencies.

Dev uses PostgreSQL 16, `db.t4g.micro`, 20 GiB gp3 encrypted storage, seven-day
automated backups, and Single-AZ placement. Public access is disabled. Module
outputs are `db_address`, `db_port`, `db_name`, `db_instance_identifier`,
`db_master_secret_arn`, and `db_subnet_group_name`; dev re-exports the first five
alongside its existing network outputs. An ARN is metadata, not a credential.

### Local Terraform validation (no deployment)

Use Terraform `~> 1.16.0` and AWS provider `~> 6.0`. From this module directory:

```powershell
terraform init -backend=false
terraform fmt -check -recursive
terraform validate
terraform test
```

Tests use a mocked AWS provider and plan-only runs, including production deletion
protection and invalid-input cases. No AWS credentials are needed for those tests;
initialization requires registry access or a populated provider cache. Then, from
`infra/envs/dev`:

```powershell
terraform init -backend=false
terraform fmt -check
terraform validate
```

These checks establish configuration validity, not deployability. Do not use an
empty-state or offline plan as a deployment decision. Before a future authorized
deployment, initialize the team's shared remote backend with its approved backend
configuration and credentials for the allowed account in `us-east-2`, then review
an AWS-backed plan. Confirm PostgreSQL 16 / `db.t4g.micro` regional availability,
quotas, subnet AZ placement, existing resources/state, and current pricing. This
milestone does not run apply or destroy or change shared state.

### Access and schema migrations after deployment

Local Docker PostgreSQL is a separate development database on localhost (default
host port 5433) with local-only credentials. RDS is private, uses managed credentials,
and is not initialized by Docker or Terraform SQL scripts. Keep the existing
Compose configuration unchanged and never point it at RDS.

Use an authorized EC2 host attached to the network module's EC2 security group,
with routing and DNS access to the RDS endpoint. Session Manager requires the SSM
agent, instance IAM permissions, operator session permissions, and connectivity to
SSM endpoints (via permitted egress or VPC endpoints). Compute/SSM setup is a team
prerequisite, not provisioned by this database module. No public RDS access or
additional database ingress is needed. A workstation may use an authorized Session
Manager remote-host port-forwarding session through that EC2 host.

Run a separately configured Flyway client on the authorized host (or through the
approved tunnel), using the existing `migrations/` directory and an RDS JDBC URL.
Require TLS with server certificate and hostname verification using the AWS RDS
CA bundle; tunneled clients must preserve verification of the RDS hostname. Retrieve
credentials at runtime with narrowly scoped Secrets Manager access (and KMS access
if applicable), without logging or committing secret values. Prefer a dedicated
migration role once bootstrap is complete. Set `baselineOnMigrate=false` and
`cleanDisabled=true`; for a fresh RDS database run `info`, `migrate`, then `validate`
and the schema verification SQL. Do not baseline a fresh database, replay local
fixtures, edit existing migrations, or use the local Compose credentials. Terraform
owns infrastructure state; Flyway owns schema history.

### Data protection and costs

Automated backups retain seven days by default (validated range 1-35). Normal
deletion requires a final snapshot; the default name is
`<project>-<environment>-postgres-final`. Before a later deletion after recreation,
choose a new `final_snapshot_identifier` if that name already exists. Production
always enables RDS deletion protection even when the optional flag is false;
disabling it requires an intentional code change and team review. Dev can opt into
protection too. Final snapshots do not replace a tested restore procedure.

RDS incurs instance, storage, backup/snapshot, Secrets Manager, and potentially
transfer and supporting EC2/endpoint costs. Retained final snapshots continue to
cost money after instance deletion. Single-AZ and the micro instance reduce dev
cost but provide no standby failover. Confirm current pricing and burstable-instance
CPU credit charges with the team; do not assume free-tier eligibility. Production
availability decisions and any network/SSM changes require teammate coordination;
the production root is unchanged in this milestone.
