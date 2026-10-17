# Teste de escala (17/10/2026, diário Entrada 114): motor (432 execuções) + PyVRP (216) + refinamento.
# Como rodar: powershell -ExecutionPolicy Bypass -File .\pesquisa\escala\rodar_escala.ps1
$ErrorActionPreference = "Continue"
$projeto = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$env:PYTHONIOENCODING = "utf-8"
Set-Location $projeto
$log = Join-Path $projeto ("pesquisa\escala\log_escala_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Write-Host "Log completo em: $log"
Add-Type -Name Energia -Namespace CCVRPESC -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[CCVRPESC.Energia]::SetThreadExecutionState([uint32]2147483649) | Out-Null
"INICIO ESCALA $(Get-Date)" | Tee-Object -FilePath $log -Append
python pesquisa\escala\rodar_escala.py --motor --pyvrp --workers 10 2>&1 | Tee-Object -FilePath $log -Append
$env:CCVRP_SP_INST_DIR = Join-Path $projeto "instances\escala"
foreach ($partida in @("CAL", "C-ACALC-B", "C-SIMC")) {
    python pesquisa\refinamento_simulacao.py --caso-sp --res-dir results_c_escala --niveis MEDIUM HIGH --tamanhos 200 400 `
      --sementes 0 1 2 --partida $partida --saida pesquisa\resultados_escala\refinamento 2>&1 | Tee-Object -FilePath $log -Append
}
Remove-Item Env:\CCVRP_SP_INST_DIR
"FIM DO TESTE DE ESCALA $(Get-Date)" | Tee-Object -FilePath $log -Append
