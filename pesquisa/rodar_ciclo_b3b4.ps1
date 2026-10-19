# Ciclos B3 e B4 (18/10/2026, diário Entrada 118): simheurística sobre o PyVRP (PY-SIM) e sementes 3-9 na escala.
# Como rodar: powershell -ExecutionPolicy Bypass -File .\pesquisa\rodar_ciclo_b3b4.ps1
# Retomável: cada etapa pula o que já está gravado.
$ErrorActionPreference = "Continue"
$projeto = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONIOENCODING = "utf-8"
Set-Location $projeto
$log = Join-Path $projeto ("pesquisa\log_ciclo_b3b4_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Write-Host "Log completo em: $log"
Add-Type -Name Energia -Namespace CCVRPB3B4 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[CCVRPB3B4.Energia]::SetThreadExecutionState([uint32]2147483649) | Out-Null

"INICIO B4 (PY-SIM) $(Get-Date)" | Tee-Object -FilePath $log -Append
python pesquisa\pyvrp_sim.py --rodar --conjuntos bench sp escala --workers 10 2>&1 | Tee-Object -FilePath $log -Append

"INICIO B3 (sementes 3-9 na escala) $(Get-Date)" | Tee-Object -FilePath $log -Append
python pesquisa\escala\rodar_escala.py --motor --pyvrp --sementes 3 4 5 6 7 8 9 --workers 8 2>&1 | Tee-Object -FilePath $log -Append

$env:CCVRP_SP_INST_DIR = Join-Path $projeto "instances\escala"
foreach ($regra in @("pontual", "lcb")) {
    foreach ($partida in @("CAL", "C-ACALC-B", "C-SIMC")) {
        python pesquisa\refinamento_simulacao.py --caso-sp --res-dir results_c_escala --niveis MEDIUM HIGH --tamanhos 200 400 `
          --sementes 0 1 2 3 4 5 6 7 8 9 --partida $partida --regra $regra `
          --saida pesquisa\resultados_escala\refinamento 2>&1 | Tee-Object -FilePath $log -Append
    }
}
Remove-Item Env:\CCVRP_SP_INST_DIR
"FIM DO CICLO B3-B4 $(Get-Date)" | Tee-Object -FilePath $log -Append
