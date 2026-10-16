# Etapas encadeadas DEPOIS da execução principal (11/10/2026, diário Entradas 88-89).
# Espera a execução principal (rodar_execucao_c_v2.ps1) terminar — a última etapa dela,
# empacotar_latex_c.py, escreve "integridade" no log — e então roda:
#   1. Refinamento por simulação nos resultados v2 (CONJUNTO DE TESTE; o método foi desenvolvido
#      nos resultados v1 arquivados): partidas C-BUF95C, C-SIMC e CAL; MEDIUM e HIGH; n = 25/50/100;
#      sementes 0-9 → pesquisa\resultados_refinamento\
#   2. Caso realista em São Paulo: 560 execuções (results_c_sp\), avaliação fora da amostra,
#      refinamento e referência PyVRP → pesquisa\resultados_caso_sp\
# Estimativa: ≈ 3 h depois do fim da execução principal.
# Como rodar (PowerShell, na raiz do projeto):
#   powershell -ExecutionPolicy Bypass -File .\pesquisa\rodar_pos_execucao.ps1

$ErrorActionPreference = "Continue"
$projeto = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONIOENCODING = "utf-8"
Set-Location $projeto
$log = Join-Path $projeto ("pesquisa\log_pos_execucao_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Write-Host "Log completo em: $log"

# Mantém o Windows acordado enquanto este script roda (não altera o plano de energia).
Add-Type -Name Energia -Namespace CCVRPPOS -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[CCVRPPOS.Energia]::SetThreadExecutionState([uint32]2147483649) | Out-Null

$principal = Get-ChildItem pesquisa -Filter "log_execucao_c_v2_*.txt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $principal) { Write-Host "ABORTADO: log da execução principal não encontrado."; exit 1 }
Write-Host ("Aguardando o fim da execução principal ({0})..." -f $principal.Name)
while (-not (Select-String -Path $principal.FullName -Pattern "integridade" -Quiet)) {
    Start-Sleep -Seconds 300
    # o log mais recente pode mudar se a execução principal for reiniciada (Entrada 90)
    $principal = Get-ChildItem pesquisa -Filter "log_execucao_c_v2_*.txt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
}
Write-Host "Execução principal concluída. Iniciando as etapas seguintes."

# 1 — refinamento nos resultados v2 (conjunto de teste)
foreach ($partida in @("C-BUF95C", "C-SIMC", "CAL")) {
    python pesquisa\refinamento_simulacao.py --res-dir results_c --niveis MEDIUM HIGH --tamanhos 025 050 100 `
      --sementes 0 1 2 3 4 5 6 7 8 9 --partida $partida --saida pesquisa\resultados_refinamento 2>&1 | Tee-Object -FilePath $log -Append
}
python pesquisa\analisar_refinamento.py --dir pesquisa\resultados_refinamento 2>&1 | Tee-Object -FilePath $log -Append

# 2 — caso São Paulo: RETIRADO deste encadeamento em 13/10/2026 (diário Entrada 92). As instâncias do caso
#     estão sendo reconstruídas com dados públicos (porte das lojas pelo CNPJ, regras de circulação etc.);
#     o caso roda depois, à mão, com as instâncias novas. Comandos originais, para referência:
# python pesquisa\caso_sp\rodar_caso_sp.py --rodar --workers 10 2>&1 | Tee-Object -FilePath $log -Append
# python pesquisa\caso_sp\rodar_caso_sp.py --analisar --workers 10 2>&1 | Tee-Object -FilePath $log -Append
# foreach ($partida in @("C-BUF95C", "C-SIMC", "CAL")) {
#     python pesquisa\refinamento_simulacao.py --caso-sp --res-dir results_c_sp --niveis MEDIUM HIGH --tamanhos 050 100 `
#       --sementes 0 1 2 3 4 5 6 7 8 9 --partida $partida --saida pesquisa\resultados_caso_sp\refinamento 2>&1 | Tee-Object -FilePath $log -Append
# }
# python pesquisa\analisar_refinamento.py --dir pesquisa\resultados_caso_sp\refinamento 2>&1 | Tee-Object -FilePath $log -Append
# python pesquisa\referencia_pyvrp.py --caso-sp --workers 6 2>&1 | Tee-Object -FilePath $log -Append

Write-Host ""
Write-Host "Concluído: pesquisa\resultados_refinamento\"
