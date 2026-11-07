# Fresh-install regression test, from infra/modules/database (requires Docker).
# Only the disposable align_phase1_smoke project is reset, never the normal dev DB.
$ErrorActionPreference = 'Stop'
$env:ALIGN_PG_HOST_PORT = '5434'
$project = 'align_phase1_smoke'

function Run-Compose {
    param([string[]]$ComposeArgs)
    & docker compose -p $project @ComposeArgs
    if ($LASTEXITCODE -ne 0) { throw "docker compose failed: $($ComposeArgs -join ' ')" }
}

try {
    Run-Compose -ComposeArgs @('down', '--volumes', '--remove-orphans')
    Run-Compose -ComposeArgs @('up', '-d', 'postgres')
    Run-Compose -ComposeArgs @('run', '--rm', 'migrate', 'migrate')
    Run-Compose -ComposeArgs @('run', '--rm', 'migrate', 'validate')
    Run-Compose -ComposeArgs @('exec', '-T', 'postgres', 'psql', '-U', 'align_dev', '-d', 'align', '-f', '/checks/verify_schema.sql')
    Run-Compose -ComposeArgs @('exec', '-T', 'postgres', 'psql', '-U', 'align_dev', '-d', 'align', '-f', '/fixtures/seed.sql')
    Run-Compose -ComposeArgs @('exec', '-T', 'postgres', 'psql', '-U', 'align_dev', '-d', 'align', '-f', '/checks/verify_fixtures.sql')
    Run-Compose -ComposeArgs @('run', '--rm', 'migrate', 'migrate')  # no-op repeat
    Run-Compose -ComposeArgs @('exec', '-T', 'postgres', 'psql', '-U', 'align_dev', '-d', 'align', '-f', '/fixtures/seed.sql')
    Run-Compose -ComposeArgs @('exec', '-T', 'postgres', 'psql', '-U', 'align_dev', '-d', 'align', '-f', '/checks/verify_fixtures.sql')
    Write-Host 'PASS: initial schema, fixtures, metrics, and repeatability.'
} finally {
    Run-Compose -ComposeArgs @('down', '--volumes', '--remove-orphans')
    Remove-Item Env:ALIGN_PG_HOST_PORT -ErrorAction SilentlyContinue
}
