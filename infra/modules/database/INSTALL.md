# ZIP installation into existing ALIGN a2 branch

1. Extract ZIP.
2. Copy **contents** of `database/` into your existing `ALIGN/infra/modules/database/` directory, replacing `docker-compose.yml`, `README.md` and `.gitignore` only after checking differences. Leave Terraform files alone.
3. Use README **Path A** if this is your already populated volume. Use **Path B** only for a fresh empty database.
4. `sql/` from the old starter can remain for now but is no longer mounted/executed automatically. Commit the new migration files; remove obsolete starter SQL after the team accepts the change.

GitHub branch `a2` could not be inspected because the connected GitHub account reports repository not found (404). This package is based on the exact local `align_postgres_v3` files previously generated and run successfully.
