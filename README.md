# ALIGN

The ALIGN runner coordinates coding agents in isolated workspaces with a shared
Git broker and pre-execution arbitration. See the [runner quickstart](runner/QUICKSTART.md).

## Test coverage

[Coverage workflow and PR reports](https://github.com/UofT-CSC490-F2026/ALIGN/actions/workflows/coverage.yml)

<!-- project-coverage:start -->

| Scope | Covered | Total | Coverage |
| --- | ---: | ---: | ---: |
| Executable lines | 2638 | 2638 | 100.00% |
| Branches | 833 | 834 | 99.88% |

<!-- project-coverage:end -->

The table shows line and branch coverage for the runner, pipeline scripts, EC2
container runner and infrastructure data-path helpers. PR checks publish a
summary and JSON report. Pushes to `a5` and `final-project` refresh the table
when the suite passes. A separate check enforces 100% line coverage.

```sh
python -m pip install -r requirements-coverage.txt
python -m pytest scripts/tests infra/modules/compute/tests infra/tests/e2e --cov -ra --cov-report=term-missing
```

For the same integration tests as CI, point `ALIGN_TEST_DATABASE_URL` at a
disposable PostgreSQL database with the repository migrations applied. Pull the
pinned Click image from the workflow and set `ALIGN_TEST_DOCKER_IMAGE` to its
digest reference. The integration tests use local Docker and do not require
model inference. Without these fixtures, pytest reports the corresponding skips.
