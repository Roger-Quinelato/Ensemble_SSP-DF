# cleanup_outputs.ps1
# Remove pastas outputs_* da raiz e mantém apenas as N runs mais recentes em outputs/
#
# Uso:
#   .\cleanup_outputs.ps1            -> mantém as 3 ultimas runs
#   .\cleanup_outputs.ps1 -Keep 5   -> mantém as 5 ultimas runs
#   .\cleanup_outputs.ps1 -DryRun   -> mostra o que seria deletado sem deletar

param(
    [int]$Keep = 3,
    [switch]$DryRun
)

$root = Split-Path $MyInvocation.MyCommand.Path

Write-Host "=== SSP-DF Cleanup de Outputs ===" -ForegroundColor Cyan

# 1. Deletar pastas outputs_* na RAIZ (geradas por testes ad-hoc do Codex)
$stray = Get-ChildItem $root -Directory | Where-Object { $_.Name -like "outputs_*" }
if ($stray.Count -gt 0) {
    Write-Host "" 
    Write-Host "[Raiz] Pastas outputs_* encontradas: $($stray.Count)" -ForegroundColor Yellow
    $stray | ForEach-Object {
        if ($DryRun) {
            Write-Host "  [DRY-RUN] Deletaria: $($_.Name)"
        } else {
            Remove-Item $_.FullName -Recurse -Force
            Write-Host "  Deletado: $($_.Name)" -ForegroundColor Red
        }
    }
} else {
    Write-Host ""
    Write-Host "[Raiz] Nenhuma pasta outputs_* encontrada. OK." -ForegroundColor Green
}

# 2. Dentro de outputs/, manter apenas as N runs mais recentes
$outputsDir = Join-Path $root "outputs"
if (Test-Path $outputsDir) {
    $runs = Get-ChildItem $outputsDir -Directory |
            Where-Object { $_.Name -match "^\d{8}_\d{6}$" } |
            Sort-Object LastWriteTime -Descending

    $toDelete = $runs | Select-Object -Skip $Keep

    if ($toDelete.Count -gt 0) {
        Write-Host ""
        Write-Host "[outputs/] Runs antigas ($($toDelete.Count) de $($runs.Count)):" -ForegroundColor Yellow
        $toDelete | ForEach-Object {
            if ($DryRun) {
                Write-Host "  [DRY-RUN] Deletaria run: $($_.Name)"
            } else {
                Remove-Item $_.FullName -Recurse -Force
                Write-Host "  Deletado run: $($_.Name)" -ForegroundColor Red
            }
        }
        Write-Host "  Mantidos: $Keep runs mais recentes." -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "[outputs/] $($runs.Count) runs dentro do limite de $Keep. OK." -ForegroundColor Green
    }
} else {
    Write-Host ""
    Write-Host "[outputs/] Pasta outputs/ nao encontrada." -ForegroundColor Gray
}

Write-Host ""
if ($DryRun) {
    Write-Host "Concluido. (DRY-RUN - nada foi deletado)" -ForegroundColor Cyan
} else {
    Write-Host "Concluido." -ForegroundColor Cyan
}
