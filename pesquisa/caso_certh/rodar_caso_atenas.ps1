# Caso de Atenas com dados operacionais (16/10/2026, diário Entrada 103).
# Espera o fim do lote dependente do horário (rodar_caso_sp_td.ps1) e então roda:
#   9 instâncias × 7 métodos × 10 sementes = 630 execuções, avaliação fora da amostra (1.000 cenários comuns)
#   e refinamento (partidas BUF0.95+C, SIM+C e CAL).
# Precisa do perfil climático de Atenas (pesquisa/clima/perfis_16716199999.json).
# Como rodar: powershell -ExecutionPolicy Bypass -File .\pesquisa\caso_certh\rodar_caso_atenas.ps1

$ErrorActionPreference = "Continue"
$projeto = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$env:PYTHONIOENCODING = "utf-8"
Set-Location $projeto
$log = Join-Path $projeto ("pesquisa\caso_certh\log_caso_atenas_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Write-Host "Log completo em: $log"
Add-Type -Name Energia -Namespace CCVRPATH -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[CCVRPATH.Energia]::SetThreadExecutionState([uint32]2147483649) | Out-Null

# espera o lote dependente do horário terminar (o marcador vai para o log dele em UTF-16)
"ESPERANDO o fim do lote dependente do horário $(Get-Date)" | Tee-Object -FilePath $log -Append
while ($true) {
    $logs = Get-ChildItem (Join-Path $projeto "pesquisa\caso_sp\log_caso_sp_td_*.txt") -ErrorAction SilentlyContinue
    if (-not $logs) { break }      # nenhum lote dependente do horário neste computador: segue direto
    if (Select-String -Path $logs.FullName -Pattern "FIM DO CASO TD" -Encoding Unicode -Quiet) { break }
    Start-Sleep -Seconds 120
}
if (-not (Test-Path (Join-Path $projeto "pesquisa\clima\perfis_16716199999.json"))) {
    "ABORTADO: falta o perfil climático de Atenas (pesquisa\clima\perfis_16716199999.json)" | Tee-Object -FilePath $log -Append
    exit 1
}

$env:CCVRP_SP_VARIANTE = "atenas"
"INICIO ATENAS $(Get-Date)" | Tee-Object -FilePath $log -Append
python pesquisa\teste_tempo_dependente.py 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\caso_sp\rodar_caso_sp.py --rodar --workers 10 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\caso_sp\rodar_caso_sp.py --analisar --workers 10 2>&1 | Tee-Object -FilePath $log -Append
foreach ($partida in @("C-BUF95C", "C-SIMC", "CAL")) {
    python pesquisa\refinamento_simulacao.py --caso-sp --res-dir results_c_atenas --niveis DATA `
      --tamanhos 063 066 067 069 074 075 077 078 084 --sementes 0 1 2 3 4 5 6 7 8 9 --partida $partida `
      --saida pesquisa\resultados_caso_atenas\refinamento 2>&1 | Tee-Object -FilePath $log -Append
}
python pesquisa\analisar_refinamento.py --dir pesquisa\resultados_caso_atenas\refinamento 2>&1 | Tee-Object -FilePath $log -Append
"FIM DO CASO ATENAS $(Get-Date)" | Tee-Object -FilePath $log -Append
Remove-Item Env:\CCVRP_SP_VARIANTE
