# ALIGN database

PostgreSQL 16, Flyway migrations, and local development tests.

- `migrations/V1__schema.sql`: 10 tables, constraints, indexes
- `migrations/V2__views.sql`: run/task evaluation and safety metrics
- `fixtures/seed.sql`: synthetic test data, **never load into AWS RDS**
- `scripts/verify_schema.sql`, `verify_fixtures.sql`: SQL assertions
- `scripts/fresh_demo_test.ps1`: clean-install test in an isolated Docker project
- `docs/`: physical ERD and metric definitions

## Fresh local setup

From `infra/modules/database`:

```powershell
docker compose up -d postgres
docker compose run --rm migrate migrate
docker compose run --rm migrate validate
docker compose exec -T postgres psql -U align_dev -d align -f /checks/verify_schema.sql
```

Optional local-only synthetic data:

```powershell
docker compose exec -T postgres psql -U align_dev -d align -f /fixtures/seed.sql
docker compose exec -T postgres psql -U align_dev -d align -f /checks/verify_fixtures.sql
```

Run `./scripts/fresh_demo_test.ps1` for an isolated repeatable test. It resets
**only** its own `align_phase1_smoke` volume.

**Existing development databases:** Before deployment, we consolidated the original
V1–V4 into two migrations. Old local Flyway volumes will fail checksum validation.
Only if its data is disposable, reset the **normal dev volume** with
`docker compose down --volumes` from this directory, then repeat the fresh setup.
**That command deletes the local database and its data; back up anything needed.**
The smoke test never resets the normal development volume.

## AWS boundary

Terraform provisions private RDS; Flyway will run separately from an authorized
network connection. Never use the local Compose credentials against AWS and never
load synthetic test fixtures into a live experiment database.

After the initial deployment, introduce future schema changes as
`V3__description.sql` rather than editing existing migrations.
