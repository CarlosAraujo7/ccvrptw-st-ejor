# Calibração dentro de uma busca (ACAL), teste (17/10/2026, diário Entrada 109): 950 execuções + avaliação.
# Como rodar: powershell -ExecutionPolicy Bypass -File .\pesquisa\rodar_acal_teste.ps1
$ErrorActionPreference = "Continue"
$projeto = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONIOENCODING = "utf-8"
Set-Location $projeto
$log = Join-Path $projeto ("pesquisa\log_acal_teste_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Write-Host "Log completo em: $log"
Add-Type -Name Energia -Namespace CCVRPACAL -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[CCVRPACAL.Energia]::SetThreadExecutionState([uint32]2147483649) | Out-Null
"INICIO ACAL TESTE $(Get-Date)" | Tee-Object -FilePath $log -Append
python pesquisa\rodar_acal.py --fase teste --variante B --workers 10 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\avaliar_acal.py --fase teste --variante B 2>&1 | Tee-Object -FilePath $log -Append
"FIM DO TESTE ACAL $(Get-Date)" | Tee-Object -FilePath $log -Append
