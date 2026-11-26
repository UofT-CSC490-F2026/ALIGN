# ALIGN

The ALIGN runner coordinates coding agents in isolated workspaces with a shared
Git broker and pre-execution arbitration. See the [runner quickstart](runner/QUICKSTART.md).

## Test coverage

[Coverage workflow and PR reports](https://github.com/UofT-CSC490-F2026/ALIGN/actions/workflows/coverage.yml)

<!-- runner-coverage:start -->

| Scope | Covered | Total | Coverage |
| --- | ---: | ---: | ---: |
| Executable lines | 1682 | 1682 | 100.00% |
| Branches | 564 | 564 | 100.00% |

<!-- runner-coverage:end -->

The report measures production Python in `runner/` and the A2 ingestion and
evaluation modules changed by this stack. Each PR check publishes a summary
and JSON artifact. The check requires 100% line and branch coverage. Successful
pushes to `final-project` refresh the README.

```powershell
py -m pip install -r requirements-coverage.txt
py -m pytest scripts/tests --cov --cov-branch --cov-report=term-missing --cov-fail-under=100
```

For the same integration tests as CI, point `ALIGN_TEST_DATABASE_URL` at a
disposable PostgreSQL database with the repository migrations applied. Pull the
pinned Click image from the workflow and set `ALIGN_TEST_DOCKER_IMAGE` to its
digest reference. The integration tests use local Docker and do not require
model inference. Without these fixtures, pytest reports the corresponding skips.
