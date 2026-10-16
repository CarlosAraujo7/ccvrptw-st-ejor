# Caso São Paulo v2 (dados públicos), 14/10/2026, diário Entrada 94.
# 560 execuções (8 instâncias × 7 métodos × 10 sementes) → avaliação fora da amostra → refinamento
# (partidas BUF0.95+C, SIM+C e CAL) → tabela do artigo. A referência PyVRP do caso (limitada por TEMPO)
# fica para quando a máquina estiver ociosa (há processos de outros projetos rodando).
# Como rodar: powershell -ExecutionPolicy Bypass -File .\pesquisa\caso_sp\rodar_caso_sp_v2.ps1

$ErrorActionPreference = "Continue"
$projeto = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$env:PYTHONIOENCODING = "utf-8"
Set-Location $projeto
$log = Join-Path $projeto ("pesquisa\caso_sp\log_caso_sp_v2_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Write-Host "Log completo em: $log"
Add-Type -Name Energia -Namespace CCVRPSP -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[CCVRPSP.Energia]::SetThreadExecutionState([uint32]2147483649) | Out-Null

python pesquisa\caso_sp\rodar_caso_sp.py --rodar --workers 10 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\caso_sp\rodar_caso_sp.py --analisar --workers 10 2>&1 | Tee-Object -FilePath $log -Append
foreach ($partida in @("C-BUF95C", "C-SIMC", "CAL")) {
    python pesquisa\refinamento_simulacao.py --caso-sp --res-dir results_c_sp --niveis MEDIUM HIGH --tamanhos 050 100 `
      --sementes 0 1 2 3 4 5 6 7 8 9 --partida $partida --saida pesquisa\resultados_caso_sp\refinamento 2>&1 | Tee-Object -FilePath $log -Append
}
python pesquisa\analisar_refinamento.py --dir pesquisa\resultados_caso_sp\refinamento 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\caso_sp\tabela_caso_sp.py 2>&1 | Tee-Object -FilePath $log -Append
Write-Host ""
Write-Host "Caso SP v2 concluído (fim do caso SP)"
