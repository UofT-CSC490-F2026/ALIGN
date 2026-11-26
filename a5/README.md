# A5 coverage baseline

On application commit `e3dcfb3`, the existing tests exercised **2,110 of
2,638 executable lines (79.98%)** and **662 of 834 branches (79.38%)**.
The local Python 3.11 run passed 228 tests and skipped 20 integration tests:
three requiring PostgreSQL, three requiring the pinned CooperBench Docker
image, and fourteen requiring a dedicated empty Docker engine for compute.

From the repository root, install `requirements-coverage.txt` and run:

```sh
python -m pytest scripts/tests infra/modules/compute/tests infra/tests/e2e \
  --cov -ra --cov-report=term-missing
```

The scope includes the runner, six pipeline scripts, the compute container
runner and two infrastructure data-path helpers. The project workflow also generates
JSON for the per-file results. GitHub Actions supplies PostgreSQL and the pinned
Click image. The dedicated-engine compute tests remain opt-in because they
require an empty engine. Compare measurements made with the same fixtures.
