# Medicoes com a maquina livre (15/10/2026, diario Entrada 96; o pesquisador autorizou encerrar os
# processos de outros projetos).
#  1. Tabela 8: microbenchmark de tempo (216 execucoes HIGH, 6 metodos, 10 processos, ordem sorteada)
#  2. Referencia PyVRP do caso SP (8 instancias x 3 sementes, limite de tempo por resolucao)
#  3. Referencia PyVRP do benchmark refeita com a maquina livre (a anterior foi arquivada em
#     resultados_pyvrp\pyvrp_20261010_maquina_compartilhada.csv)
# Como rodar: powershell -ExecutionPolicy Bypass -File .\pesquisa\rodar_medicoes_maquina_livre.ps1

$ErrorActionPreference = "Continue"
$projeto = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONIOENCODING = "utf-8"
Set-Location $projeto
$log = Join-Path $projeto ("pesquisa\log_medicoes_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Write-Host "Log completo em: $log"
Add-Type -Name Energia -Namespace CCVRPMED -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[CCVRPMED.Energia]::SetThreadExecutionState([uint32]2147483649) | Out-Null

python pesquisa\tempo_microbenchmark.py --rodar --tabela 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\referencia_pyvrp.py --caso-sp --workers 6 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\referencia_pyvrp.py --workers 6 2>&1 | Tee-Object -FilePath $log -Append
Write-Host ""
Write-Host "Medicoes concluidas (fim das medicoes)"
