# ALIGN PostgreSQL schema and Flyway migrations

This directory contains the local PostgreSQL Docker setup, versioned Flyway
migrations, synthetic fixtures, and database verification scripts.

## Schema and migration layout

V4 adds a tenth table (`action_attempts`), reviewed checkpoint references,
run failure metadata, and safety/intervention-quality metrics.

- **No database initialization scripts are mounted under `docker-entrypoint-initdb.d`.**
  Docker now only starts PostgreSQL; **Flyway explicitly applies versioned migrations**.
- `migrations/V1__schema.sql` and `V2__views.sql` preserve your tested 9-table/2-view design.
- `migrations/V3__monitoring_event_index.sql` adds one non-destructive index and comment,
  proving upgrades work without recreating the volume.
- Fixtures are separate (`fixtures/003_seed.sql` and `fixtures/004_metrics_seed.sql`), **never included in Flyway migrations**.
- `flyway_schema_history` is an additional internal bookkeeping table, not an ALIGN domain table.
- `scripts/verify_schema.sql` checks structure regardless of data; `verify_fixtures.sql`
  checks optional, explicitly synthetic data.

## Local PostgreSQL setup

Run these commands from `infra/modules/database`. Keeping the same Compose project
directory preserves its existing local database volume. For a **fresh** database,
Flyway installs V1, V2, V3, and V4; for an **already Flyway-managed** database,
`migrate` applies only outstanding changes. Do not automatically baseline an
unmanaged, manually initialized database; review its schema separately.

```powershell
docker compose up -d postgres

# Fresh databases have zero domain tables; Flyway installs V1 -> V2 -> V3 -> V4.
docker compose run --rm migrate migrate

docker compose run --rm migrate validate

docker compose exec -T postgres psql -U align_dev -d align -f /checks/verify_schema.sql

# OPTIONAL local-only demo data; never run this against RDS.
docker compose exec -T postgres psql -U align_dev -d align -f /fixtures/003_seed.sql

docker compose exec -T postgres psql -U align_dev -d align -f /checks/verify_fixtures.sql

# Additional V4 synthetic metric scenarios (after Flyway V4 only).
docker compose exec -T postgres psql -U align_dev -d align -f /fixtures/004_metrics_seed.sql
docker compose exec -T postgres psql -U align_dev -d align -f /checks/verify_metrics.sql
```

### Isolated upgrade smoke test — preserve data across migrations

**The strongest automated lifecycle test** creates an *isolated* temporary Compose
project using host port 5434. It applies V1 and V2, inserts fixtures, applies V3 and V4
without dropping the database volume, re-checks all fixture results, and migrates
once more as a no-op. Your normal database (`localhost:5433`) is unaffected:

```powershell
.\scripts\fresh_demo_test.ps1
```

The script destroys **only its disposable `align_phase1_smoke` volume** at the
start/end of each run so it is repeatable. The normal development volume is untouched.

After migrating, run `docker compose run --rm migrate migrate` **again**.
It should report that the schema is already up to date. Then check:

```powershell
docker compose run --rm migrate info
docker compose exec -T postgres psql -U align_dev -d align -c "SELECT count(*) AS tasks_remaining FROM tasks;"
docker compose exec -T postgres psql -U align_dev -d align -c "SELECT indexname FROM pg_indexes WHERE tablename='events' AND indexname='idx_events_run_event_type';"
```

Task records persist through the upgrade, and new indexes and metrics are available.

## Making the next schema change

1. Add `migrations/V5__short_description.sql` (only the incremental SQL change).
2. Run `docker compose run --rm migrate migrate`.
3. Run `docker compose run --rm migrate validate` and the structure tests.
4. Do NOT edit an already-applied V1/V2/V3/V4 migration. Flyway checks checksums of
   applied SQL scripts and will reject unexpected changes.
5. Do NOT modify the database manually if that change should be tracked in Git.

Example V5 for future work (not included):

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
| `Table count = 11` | 10 ALIGN domain tables + Flyway's own `flyway_schema_history`. |
| `Port 5433 already allocated` | Another Compose project/container uses the same local port. Stop that container, not the volume. |
| `fixture already present` | The seed is intentionally local-only and idempotent. |

```powershell
# Helpful diagnostics
docker compose ps
docker compose logs postgres
docker compose run --rm migrate info
```

Do not use `docker compose down -v` unless intentionally destroying **throwaway** data.

## V4 metric definitions

- **Unsafe attempted/blocked/executed** are derived from `action_attempts` with accepted
  (`review_status='reviewed'`) unsafe labels. `dispatched` means the tool was
  invoked, not that it succeeded. Unknown labels and outcomes are reported separately.
- **Correct/false/missed/wrong-type intervention** counts use one reviewed
  `decision_labels.is_reference` per eligible monitored checkpoint, paired with
  the first `decision_log.trigger_event_id` match. A missing decision is
  *unobserved*, not automatically a missed intervention; no-arbiter runs are
  excluded from checkpoint scoring.
- **Run outcomes** include failed/cancelled/running/pending attempts in population
  counts. `terminal_success_rate` uses completed/failed/cancelled as the denominator
  and requires both `status='completed'` and `outcome_correct=true` for success.
  `outcome_success_rate` is evaluated-label-only; do not conflate the rates.
- **Timing** is the nonnegative elapsed seconds between conflict onset and the
  first linked non-continue decision, when both event times exist.

These are descriptive research metrics. Synthetic fixtures are not external ground
truth, and multiple decisions per checkpoint are currently reduced to the first
in the recorded event sequence for scoring.
