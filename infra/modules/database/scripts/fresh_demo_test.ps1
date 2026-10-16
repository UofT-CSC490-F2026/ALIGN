# Run in ALIGN/infra/modules/database (PowerShell with Docker Desktop running).
# Uses an isolated, disposable Compose project and host port 5434.
# Does NOT alter your normal database project or its Docker volume.
$ErrorActionPreference = 'Stop'
$env:ALIGN_PG_HOST_PORT = '5434'
$project = 'align_phase1_smoke'
function Run-Compose {
    param([string[]]$ComposeArgs)
    & docker compose -p $project @ComposeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose failed: $($ComposeArgs -join ' ')"
    }
}
try {
    Write-Host 'Starting isolated PostgreSQL test on 127.0.0.1:5434'
    # This project is exclusively for smoke tests; remove ONLY its own old test volume.
    Run-Compose -ComposeArgs @('down', '--volumes', '--remove-orphans')
    Run-Compose -ComposeArgs @('up', '-d', 'postgres')
    # Start with V1 + V2 only, simulating the state before our additive change.
    Run-Compose -ComposeArgs @('run', '--rm', 'migrate', '-target=2', 'migrate')
    Run-Compose -ComposeArgs @('exec', '-T', 'postgres', 'psql', '-U', 'align_dev', '-d', 'align', '-f', '/checks/verify_base_schema.sql')
    Run-Compose -ComposeArgs @('exec', '-T', 'postgres', 'psql', '-U', 'align_dev', '-d', 'align', '-f', '/fixtures/003_seed.sql')
    Run-Compose -ComposeArgs @('exec', '-T', 'postgres', 'psql', '-U', 'align_dev', '-d', 'align', '-f', '/checks/verify_fixtures.sql')
    # This updates the schema, preserving the local dataset and volume.
    Run-Compose -ComposeArgs @('run', '--rm', 'migrate', 'migrate')
    Run-Compose -ComposeArgs @('run', '--rm', 'migrate', 'validate')
    # Check the original V2 fixture remains intact following the V3/V4 upgrade.
    Run-Compose -ComposeArgs @('exec', '-T', 'postgres', 'psql', '-U', 'align_dev', '-d', 'align', '-f', '/checks/verify_fixtures.sql')
    Run-Compose -ComposeArgs @('exec', '-T', 'postgres', 'psql', '-U', 'align_dev', '-d', 'align', '-f', '/fixtures/004_metrics_seed.sql')
    Run-Compose -ComposeArgs @('exec', '-T', 'postgres', 'psql', '-U', 'align_dev', '-d', 'align', '-f', '/checks/verify_schema.sql')
    Run-Compose -ComposeArgs @('exec', '-T', 'postgres', 'psql', '-U', 'align_dev', '-d', 'align', '-f', '/checks/verify_metrics.sql')
    Run-Compose -ComposeArgs @('run', '--rm', 'migrate', 'migrate')  # no-op; repeatable
    Run-Compose -ComposeArgs @('exec', '-T', 'postgres', 'psql', '-U', 'align_dev', '-d', 'align', '-f', '/fixtures/004_metrics_seed.sql')
    Run-Compose -ComposeArgs @('exec', '-T', 'postgres', 'psql', '-U', 'align_dev', '-d', 'align', '-f', '/checks/verify_metrics.sql')
    Write-Host 'PASS: fresh initialization, V2-to-V4 upgrade, preserved data, safety metrics, repeat run.'
} finally {
    # Only this disposable smoke-test volume is destroyed; normal dev DB is untouched.
    Run-Compose -ComposeArgs @('down', '--volumes', '--remove-orphans')
    Remove-Item Env:ALIGN_PG_HOST_PORT -ErrorAction SilentlyContinue
}
