# Regenerate the ctags index for symbol lookup (see CLAUDE.md tokens-first workflow).
# Requires universal-ctags on PATH. Run from the repo root: pwsh scripts/tags.ps1
$ErrorActionPreference = "Stop"

$ctags = Get-Command ctags -ErrorAction SilentlyContinue
if ($null -eq $ctags) {
    Write-Error "universal-ctags not found on PATH. Install it (e.g. winget install universal-ctags.ctags) and retry."
    exit 1
}

$root = Split-Path -Parent $PSScriptRoot

& $ctags.Source -R `
    --languages=Python `
    --python-kinds=-i `
    --exclude='.venv' `
    --exclude='.agent-workspace' `
    --exclude='*.egg-info' `
    -f (Join-Path $root 'tags') `
    (Join-Path $root 'direbm') `
    (Join-Path $root 'experiments') `
    (Join-Path $root 'scripts')

Write-Host "tags written to $(Join-Path $root 'tags')"
