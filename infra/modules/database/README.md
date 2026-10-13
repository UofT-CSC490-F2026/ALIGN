# ALIGN — Phase 1: Safe, repeatable PostgreSQL schema migrations

**Target folder:** `ALIGN/infra/modules/database/` on branch `a2`. This package is a
**drop-in replacement for the local Docker/SQL setup**; it does not modify Terraform,
AWS, or your GitHub branch. Designed from the tested `align_postgres_v3` package.

> The linked GitHub repository's `a2` contents could not be read with the current
> GitHub connection (404/not accessible). Review differences in your local branch
> before overwriting any existing files. No remote repository files were changed.

## What changed from your working v3

- **No database initialization scripts are mounted under `docker-entrypoint-initdb.d`.**
  Docker now only starts PostgreSQL; **Flyway explicitly applies versioned migrations**.
- `migrations/V1__schema.sql` and `V2__views.sql` preserve your tested 9-table/2-view design.
- `migrations/V3__monitoring_event_index.sql` adds one non-destructive index and comment,
  proving upgrades work without recreating the volume.
- Fixtures are separate (`fixtures/003_seed.sql`), **never included in Flyway migrations**.
- `flyway_schema_history` is an additional internal bookkeeping table, not a 10th ALIGN table.
- `scripts/verify_schema.sql` checks structure regardless of data; `verify_fixtures.sql`
  checks optional, explicitly synthetic data.

## IMPORTANT: choose ONE path below

Keep working in your original `ALIGN/infra/modules/database` folder. Do **NOT**
rename the folder before `docker compose up -d postgres` if you want Docker Compose
 to reuse its existing `database_align_pgdata` volume.

### Path A — your **existing populated v3 database** (the one you already tested)

Open PowerShell in `ALIGN\infra\modules\database`. Copy this package's contents
into that folder, replacing its `docker-compose.yml` and `README.md`, and **adding**
`migrations/`, `fixtures/`, and the new verification scripts. The old `sql/` directory
can stay temporarily, but it is no longer used by Compose. Do not delete Docker volumes.

```powershell
# Start/reuse the existing PostgreSQL volume; leaves existing fixture records untouched
docker compose up -d postgres

# Mandatory safety check of the pre-Flyway v3 schema
docker compose exec -T postgres psql -U align_dev -d align -f /checks/verify_legacy_v3.sql

# ONE TIME ONLY: tell Flyway that the existing schema is equivalent to V1 and V2.
# This does not execute V1 or V2, nor does it erase existing rows.
docker compose run --rm migrate -baselineVersion=2 baseline

# Apply ONLY V3 and later changes. The command is safe to repeat.
docker compose run --rm migrate migrate

docker compose run --rm migrate validate

docker compose exec -T postgres psql -U align_dev -d align -f /checks/verify_schema.sql

docker compose exec -T postgres psql -U align_dev -d align -f /checks/verify_fixtures.sql
```

**Never run `baseline` against a fresh empty database.** On your existing database,
run it only after the v3 preflight succeeds. `baseline` trusts that your old schema
matches V1/V2 — it does not replay or checksum them. If you already changed the
schema manually, compare it to those files and investigate before baselining.

### Path B — a **fresh local database**

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

### Phase 1 acceptance test — update the schema without touching the volume

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

After completing Path A or B, run `docker compose run --rm migrate migrate` **again**.
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
| `Found non-empty schema(s) without schema history table` | Existing v3 volume; perform Path A preflight and explicit baseline. |
| `relation "tasks" already exists` | You ran fresh migrations against an existing schema; stop and use Path A instead. |
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
