# Ciclo de 15/10/2026 (analise de fraquezas; aprovado pelo pesquisador: "Pode seguir com o 2 e o 1,
# inclui o 3 tambem"). Diario Entrada 97.
#  A2. Pipeline PyVRP + buffer + certificacao + refinamento (limitado por TEMPO: roda primeiro e sozinho)
#  A1. Caso SP ampliado para 16 instancias (amostras SP3 e SP4: 560 execucoes novas)
#  A3. Caso SP planejado com o pico da manha (SPP1, SPP2: 560 execucoes)
#  Analises: avaliacao fora da amostra, refinamento do caso SP (16 instancias), tabelas, estresse,
#  comparacao do pico, PyVRP do caso SP (16 instancias) e analise do pipeline.
# Resultados anteriores do caso SP (8 instancias) arquivados em pesquisa\resultados_caso_sp\*_8instancias*.
# Como rodar: powershell -ExecutionPolicy Bypass -File .\pesquisa\rodar_ciclo_20261015.ps1

$ErrorActionPreference = "Continue"
$projeto = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONIOENCODING = "utf-8"
Set-Location $projeto
$log = Join-Path $projeto ("pesquisa\log_ciclo_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Write-Host "Log completo em: $log"
Add-Type -Name Energia -Namespace CCVRPCIC -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[CCVRPCIC.Energia]::SetThreadExecutionState([uint32]2147483649) | Out-Null

# A2 - pipeline PyVRP (sozinho na maquina)
python pesquisa\pipeline_pyvrp_buffer.py --rodar --workers 10 2>&1 | Tee-Object -FilePath $log -Append

# A1 - SP3 e SP4 (o --resume do runner pula as 560 execucoes de SP1 e SP2)
$env:CCVRP_SP_VARIANTE = ""
python pesquisa\caso_sp\rodar_caso_sp.py --rodar --workers 10 2>&1 | Tee-Object -FilePath $log -Append
# A3 - variante do pico
$env:CCVRP_SP_VARIANTE = "pico"
python pesquisa\caso_sp\rodar_caso_sp.py --rodar --workers 10 2>&1 | Tee-Object -FilePath $log -Append

# avaliacao fora da amostra
$env:CCVRP_SP_VARIANTE = ""
python pesquisa\caso_sp\rodar_caso_sp.py --analisar --workers 10 2>&1 | Tee-Object -FilePath $log -Append
$env:CCVRP_SP_VARIANTE = "pico"
python pesquisa\caso_sp\rodar_caso_sp.py --analisar --workers 10 2>&1 | Tee-Object -FilePath $log -Append

# refinamento e tabelas do caso SP (16 instancias)
$env:CCVRP_SP_VARIANTE = ""
foreach ($partida in @("C-BUF95C", "C-SIMC", "CAL")) {
    python pesquisa\refinamento_simulacao.py --caso-sp --res-dir results_c_sp --niveis MEDIUM HIGH --tamanhos 050 100 `
      --sementes 0 1 2 3 4 5 6 7 8 9 --partida $partida --saida pesquisa\resultados_caso_sp\refinamento 2>&1 | Tee-Object -FilePath $log -Append
}
python pesquisa\analisar_refinamento.py --dir pesquisa\resultados_caso_sp\refinamento 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\caso_sp\tabela_caso_sp.py 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\caso_sp\estresse_pico.py 2>&1 | Tee-Object -FilePath $log -Append
$env:CCVRP_SP_VARIANTE = "pico"
python pesquisa\caso_sp\tabela_caso_sp.py 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\caso_sp\comparar_pico.py 2>&1 | Tee-Object -FilePath $log -Append

# PyVRP do caso SP (16 instancias) e analise do pipeline
$env:CCVRP_SP_VARIANTE = ""
python pesquisa\referencia_pyvrp.py --caso-sp --workers 6 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\pipeline_pyvrp_buffer.py --analisar 2>&1 | Tee-Object -FilePath $log -Append
"FIM DO CICLO" | Tee-Object -FilePath $log -Append
