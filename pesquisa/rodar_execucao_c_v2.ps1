# Reexecução completa do Caminho C, desenho "caminho 1" (10/10/2026, diário Entradas 83-87;
# aprovado pelo pesquisador: "Vamos com o caminho 1, incluindo a sensibilidade opcional").
#
# Desenho:
#   - Experimento principal: carga refrigerada HOMOGÊNEA (toda a carga segue a categoria
#     dominante; src\ccvrp_c.py, CINETICA_POR_CLIENTE = False), como na versão de conferência.
#   - Multicompartimento (B2): carga MISTA, cada produto com a própria cinética, comparada com
#     uma base de compartimento único com a MESMA carga mista (variantes *-MIX).
#   - Motor novo: eliminação de rotas na busca local (ELIMINAR_ROTAS = True) e busca local por
#     variação de custo (BUSCA_LOCAL_DELTA = True; mesmas decisões, cerca de 2x mais rápido).
#
# Resultados antigos ARQUIVADOS, não apagados: results_c_v1_cindom\, results_c_extra_v1_cindom\,
# results_c_b2_v1_motorantigo\, pesquisa\resultados_*_v1_cindom\, pesquisa\log_test_split_ls_v1_cindom.txt
#
# Fases, todas com --resume (se cair, rode de novo e ele continua de onde parou):
#   Fase 1  — núcleo, α = 0,90, 54 instâncias × 10 sementes (4.860 execuções)
#   Fase 2  — α = 0,95, 5 sementes (1.080)
#   Fase 3  — B2 com zonas, Z ∈ {2,3}, 5 sementes (1.620)
#   Fase 3b — B2, base de compartimento único com carga mista (810)
#   Fase 3c — B2, sensibilidade de γ e (m_L, m_U) em metade e dobro, n = 50, 3 sementes (432)
#   Fase 3d — SIM+C com 100 cenários na busca, n = 50, MEDIUM e HIGH, 5 sementes (60)
#   Fase 3e — correlação dentro da busca, ρ = 0,5 (120)
#   Fase 4  — validação contra o ótimo exato (test_split_ls.py)
#   Fase 5  — análises, tabelas, figuras e pacote LaTeX
#   Estimativa total ≈ 33 h com 10 workers; com 6 workers (Entrada 90), ≈ 55 h ou mais se a máquina for usada em paralelo (fases 1-2 ≈ 20 h; fase 3 ≈ 6 h; 3b ≈ 3 h; 3c-3e ≈ 2 h; 4 ≈ 0,3 h; 5 ≈ 2-3 h),
#   com busca local em 50% dos filhos (p_ls = 0,5), eliminação de rotas e busca local delta (Entrada 87).
#
# Como rodar (PowerShell, na raiz do projeto):
#   powershell -ExecutionPolicy Bypass -File .\pesquisa\rodar_execucao_c_v2.ps1

$ErrorActionPreference = "Continue"
$projeto = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONIOENCODING = "utf-8"
$log = Join-Path $projeto ("pesquisa\log_execucao_c_v2_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Set-Location $projeto
Write-Host "Log completo em: $log"

# Mantém o Windows acordado enquanto ESTE script roda (SetThreadExecutionState, o mesmo recurso
# usado por players de vídeo): não altera o plano de energia e deixa de valer quando o processo
# termina. ES_CONTINUOUS | ES_SYSTEM_REQUIRED = 0x80000001 = 2147483649. Fechar a tampa ainda suspende.
Add-Type -Name Energia -Namespace CCVRP -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[CCVRP.Energia]::SetThreadExecutionState([uint32]2147483649) | Out-Null

# Trava 1: o motor tem de estar na configuração aprovada.
foreach ($par in @(@("CINETICA_POR_CLIENTE", "False"), @("ELIMINAR_ROTAS", "True"), @("BUSCA_LOCAL_DELTA", "True"))) {
    if (-not (Select-String -Path "src\ccvrp_c.py" -Pattern ("^{0} = {1}" -f $par[0], $par[1]) -Quiet)) {
        Write-Host ("ABORTADO: src\ccvrp_c.py não está com {0} = {1}." -f $par[0], $par[1]); exit 1
    }
}
# Trava 2: o --resume pularia execuções já existentes. Se houver numa pasta de saída alguma
# execução SEM a marca do motor novo, aborta (seriam resultados antigos misturados).
foreach ($pasta in @("results_c", "results_c_extra", "results_c_cor", "results_c_b2", "results_c_b2_sens")) {
    if (Test-Path $pasta) {
        $antigas = Get-ChildItem -Path $pasta -Recurse -Filter "*_metrics.json" |
            Where-Object { -not (Select-String -Path $_.FullName -Pattern '"busca_local_delta": true' -Quiet) }
        if ($antigas) {
            Write-Host "ABORTADO: $pasta contém execuções do motor antigo (ex.: $($antigas[0].Name))."
            Write-Host "Arquive essa pasta antes de rodar (nada foi executado)."
            exit 1
        }
    }
}

# Fase 1 — núcleo
python src\experiment_runner.py --algorithms C-DET C-DETC C-BUF90 C-BUF70C C-BUF80C C-BUF90C C-BUF95C C-SIMC C-MOEADC `
  --seeds 10 --workers 10 --resume --results-dir results_c 2>&1 | Tee-Object -FilePath $log -Append

# Fase 2 — nível de serviço 0,95
python src\experiment_runner.py --algorithms C-SIMC95 C-BUF90C95 C-BUF95C95 C-BUF99C95 `
  --seeds 5 --workers 10 --resume --results-dir results_c 2>&1 | Tee-Object -FilePath $log -Append

# Fase 3 — B2 com zonas (só as tarefas com zonas)
python src\experiment_runner.py --algorithms C-DET C-BUF90C C-SIMC --zones 2 3 --sem-base `
  --seeds 5 --workers 10 --resume --results-dir results_c_b2 --results-b2-dir results_c_b2 2>&1 | Tee-Object -FilePath $log -Append

# Fase 3b — B2, base de compartimento único com carga mista
python src\experiment_runner.py --algorithms C-DET-MIX C-BUF90C-MIX C-SIMC-MIX `
  --seeds 5 --workers 10 --resume --results-dir results_c_b2 2>&1 | Tee-Object -FilePath $log -Append

# Fase 3c — B2, sensibilidade (n = 50, 3 sementes, só tarefas com zonas)
python src\experiment_runner.py --algorithms C-SIMC-G05 C-SIMC-G2 C-SIMC-KM05 C-SIMC-KM2 --zones 2 3 --sem-base `
  --sizes 50 --seeds 3 --workers 10 --resume --results-dir results_c_b2_sens --results-b2-dir results_c_b2_sens 2>&1 | Tee-Object -FilePath $log -Append

# Fase 3d — tamanho da amostra de busca (100 cenários), n = 50, MEDIUM e HIGH
python src\experiment_runner.py --algorithms C-SIMC100 --sizes 50 --uncertainty MEDIUM HIGH `
  --seeds 5 --workers 10 --resume --results-dir results_c_extra 2>&1 | Tee-Object -FilePath $log -Append

# Fase 3e — correlação dentro da busca (pesquisa\correlacao_busca.py; src\ não é alterado)
python pesquisa\correlacao_busca.py --rodar --workers 10 2>&1 | Tee-Object -FilePath $log -Append

# Fase 4 — validação contra o ótimo exato (cmd mantém o log em UTF-8, que o gerador de tabelas lê)
cmd /c "python -X utf8 pesquisa\test_split_ls.py > pesquisa\log_test_split_ls.txt 2>&1"
Get-Content pesquisa\log_test_split_ls.txt -Encoding UTF8 | Tee-Object -FilePath $log -Append

# Fase 5 — análises (ordem importa: analisar_c cria os *_oosC.json usados pelas demais)
python pesquisa\analisar_c.py --dirs results_c --out pesquisa\resultados_c --workers 10 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\analisar_c.py --dirs results_c_b2 --out pesquisa\resultados_c_b2 --workers 10 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\analisar_c.py --dirs results_c_b2_sens --out pesquisa\resultados_c_b2_sens --workers 10 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\parte_b_caracterizacao.py --dirs results_c --algs C-DET C-BUF90C C-SIMC 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\robustez_revisor.py 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\escolha_decisor.py 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\comparar_simc100.py 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\equivalencia.py 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\correlacao_busca.py --analisar --workers 10 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\referencia_pyvrp.py --so-gap 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\conformidade_legal_b2.py 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\sensibilidade_b2.py 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\gerar_tabelas_c.py 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\gerar_tab_robustez.py 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\gerar_figuras_c.py 2>&1 | Tee-Object -FilePath $log -Append
python pesquisa\empacotar_latex_c.py 2>&1 | Tee-Object -FilePath $log -Append

Write-Host ""
Write-Host "Concluído. Relatórios: pesquisa\resultados_c\, resultados_c_b2\, resultados_c_b2_sens\, resultados_parte_b\, resultados_robustez\, resultados_pyvrp\"
Write-Host "As tabelas do LaTeX foram regeneradas; o TEXTO do manuscrito ainda precisa ser atualizado com os números novos."
