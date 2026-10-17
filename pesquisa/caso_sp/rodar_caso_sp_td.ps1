# Caso São Paulo com tempos de viagem dependentes do horário (16/10/2026, diário Entrada 101).
# Desenho fixado ANTES de rodar (diário Entrada 101):
#   1. SPT (instances/caso_sp_td): 16 instâncias × 7 métodos × 10 sementes = 1.120 execuções, planejadas e
#      avaliadas com os tempos dependentes do horário; avaliação fora da amostra (1.000 cenários comuns);
#      refinamento (partidas BUF0.95+C, SIM+C e CAL).
#   2. SPM (instances/caso_sp_media_operacao): as mesmas 1.120 execuções com tempos estáticos no fator médio
#      das horas de operação (o melhor planejamento estático sem o horário); avaliação fora da amostra.
#   Depois (comparar_td.py): planos de SP (média anual), SPM e SPP (pico) executados no mundo dependente do
#   horário, contra os planejados com o horário.
# Como rodar: powershell -ExecutionPolicy Bypass -File .\pesquisa\caso_sp\rodar_caso_sp_td.ps1

$ErrorActionPreference = "Continue"
$projeto = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$env:PYTHONIOENCODING = "utf-8"
Set-Location $projeto
$log = Join-Path $projeto ("pesquisa\caso_sp\log_caso_sp_td_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Write-Host "Log completo em: $log"
Add-Type -Name Energia -Namespace CCVRPSPTD -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[CCVRPSPTD.Energia]::SetThreadExecutionState([uint32]2147483649) | Out-Null

$env:CCVRP_SP_VARIANTE = "td"
"INICIO SPT $(Get-Date)" | Tee-Object -FilePath $log -Append
python pesquisa\teste_tempo_dependente.py 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\caso_sp\rodar_caso_sp.py --rodar --workers 10 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\caso_sp\rodar_caso_sp.py --analisar --workers 10 2>&1 | Tee-Object -FilePath $log -Append
foreach ($partida in @("C-BUF95C", "C-SIMC", "CAL")) {
    python pesquisa\refinamento_simulacao.py --caso-sp --res-dir results_c_sp_td --niveis MEDIUM HIGH --tamanhos 050 100 `
      --sementes 0 1 2 3 4 5 6 7 8 9 --partida $partida --saida pesquisa\resultados_caso_sp_td\refinamento 2>&1 | Tee-Object -FilePath $log -Append
}
python pesquisa\analisar_refinamento.py --dir pesquisa\resultados_caso_sp_td\refinamento 2>&1 | Tee-Object -FilePath $log -Append
"FIM SPT $(Get-Date)" | Tee-Object -FilePath $log -Append

$env:CCVRP_SP_VARIANTE = "media"
"INICIO SPM $(Get-Date)" | Tee-Object -FilePath $log -Append
python pesquisa\caso_sp\rodar_caso_sp.py --rodar --workers 10 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\caso_sp\rodar_caso_sp.py --analisar --workers 10 2>&1 | Tee-Object -FilePath $log -Append
"FIM DO CASO TD $(Get-Date)" | Tee-Object -FilePath $log -Append
Remove-Item Env:\CCVRP_SP_VARIANTE
