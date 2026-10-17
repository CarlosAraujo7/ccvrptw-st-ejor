#!/usr/bin/env python3
"""
Resumo do teste da calibração dentro de uma busca (ACAL), 17/10/2026, diário Entrada 110.
Por conjunto (benchmark v2, SP, SP com horário, Atenas), pareado por instância (média das sementes 0-9) contra SIM+C
(mesmo orçamento: uma busca): ACAL e ACAL+R (refinamento da Seção 4.4, < 1 s por plano) — custo, perda de qualidade,
CO2; Hodges-Lehmann e Wilcoxon com Holm sobre os seis testes do conjunto; % de execuções cujo plano cumpre α.
No benchmark, ACAL+R usa as 360 execuções de incerteza média e alta (como o refinamento do artigo).
Saídas: pesquisa/resultados_acal/relatorio_resumo_acal.txt e latex_ejor_c/tables/tab_acal.tex
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibracao_preditiva_base as b
import avaliar_acal as aa

BASE = b.BASE; ALPHA = b.ALPHA
OUT = os.path.join(BASE, 'pesquisa', 'resultados_acal')
REF = {'bench': 'refinamento_bench', 'SP': 'refinamento_sp', 'SPT': 'refinamento_sp_td', 'ATH': 'refinamento_atenas'}
ROT = {'bench': 'Benchmark', 'SP': r'S\~ao Paulo', 'SPT': r'S\~ao Paulo, time-dependent', 'ATH': 'Athens'}


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj


def main():
    rel = open(os.path.join(OUT, 'relatorio_resumo_acal.txt'), 'w', encoding='utf-8')

    def log(t=''):
        print(t, flush=True); rel.write(t + '\n')
    log('ACAL (uma busca) e ACAL+R contra SIM+C (uma busca), pareado por instância; HL [Holm sobre 6 testes por conjunto]')
    linhas_tab = []
    for c in ('bench', 'SP', 'SPT', 'ATH'):
        R = b.carregar(aa.ORIG[c])
        E = aa.escolhas(aa.ACAL[c], 'C-ACALC-B')
        sim = {k: d.get('C-SIMC') for k, d in R.items()}
        ref = {}
        arq = os.path.join(OUT, REF[c], 'refinamento_C-ACALC-B.json')
        if os.path.exists(arq):
            for x in json.load(open(arq, encoding='utf-8')):
                ref[(x['instance'], x['seed'])] = dict(f1=x['fin_f1'], q_oos=x['fin_q_oos'], f3=x['fin_f3'], p_on=x['fin_p_oos'])
        metodos = {'ACAL': E, 'ACAL+R': ref}
        testes = []; cumpre = {}
        for nm, M in metodos.items():
            ks = [k for k in M if M[k] and sim.get(k)]
            cumpre[nm] = 100 * np.mean([M[k]['p_on'] >= ALPHA for k in ks]) if ks else float('nan')
            cumpre['SIM+C' if nm == 'ACAL' else 'SIM+C (mesmas execuções do +R)'] = 100 * np.mean([sim[k]['p_on'] >= ALPHA for k in ks]) if ks else float('nan')
            insts = sorted({i for i, _ in ks})
            for k_, rot in (('f1', 'custo'), ('q_oos', 'perda'), ('f3', 'CO2')):
                dif = []
                for i in insts:
                    a = np.mean([M[k][k_] for k in ks if k[0] == i]); s = np.mean([sim[k][k_] for k in ks if k[0] == i])
                    dif.append(100 * (a - s) if k_ == 'q_oos' else 100 * (a / s - 1))
                p = wilcoxon(dif).pvalue if any(abs(v) > 1e-12 for v in dif) else 1.0
                testes.append((nm, rot, b.hl(dif), p, len(dif)))
        adj = holm(np.array([t[3] for t in testes]))
        log(f'\n=== {aa.NOMES[c]} ===  cumpre: ' + ', '.join(f'{k} {v:.1f}%' for k, v in cumpre.items()))
        res = {}
        for (nm, rot, e, p, n), pa in zip(testes, adj):
            log(f'  {nm:7s} {rot:6s} n={n:2d}  HL {e:+7.3f}{"pp" if rot == "perda" else "% "}  p={p:.3g}  p_Holm={pa:.3g}')
            res[(nm, rot)] = (e, pa)
        def cel(nm, rot):
            e, pa = res[(nm, rot)]
            return f'{e + 0.0:+.2f}' + ('$^*$' if pa < 0.05 else '')
        linhas_tab.append(f"{ROT[c]} & {cumpre['SIM+C']:.0f} & {cumpre['ACAL']:.0f} & {cel('ACAL', 'custo')} & {cel('ACAL', 'perda')} & "
                          f"{cel('ACAL', 'CO2')} & {cumpre.get('ACAL+R', float('nan')):.0f} & {cel('ACAL+R', 'custo')} & "
                          f"{cel('ACAL+R', 'perda')} & {cel('ACAL+R', 'CO2')} " + r'\\')
    rel.close()
    L = [r'\begin{table}[htbp]', r'\centering',
         r'\caption{Calibration within a single search (\mACAL{}) and its refinement (+R) against \mSIM{} at equal computational '
         r'budget (one search each), on the test runs (seeds 0--9; 1{,}000 common scenarios). Comply: \% of runs whose chosen '
         r'plan meets $\alpha=0.90$. Differences: Hodges--Lehmann estimate of the paired difference per instance in cost (\%), '
         r'quality loss (percentage points) and CO$_2$ (\%); $^*$: $p_{\mathrm{Holm}}<0.05$ over the six tests of each set. '
         r'On the benchmark, +R uses the 360 runs with medium or high uncertainty.}',
         r'\label{tab:acal}', r'\footnotesize', r'\setlength{\tabcolsep}{3.5pt}',
         r'\begin{tabular}{@{}l rr rrr r rrr@{}}', r'\toprule',
         r' & \multicolumn{2}{c}{Comply (\%)} & \multicolumn{3}{c}{\mACAL{} $-$ \mSIM{}} & & \multicolumn{3}{c}{\mACAL{}+R $-$ \mSIM{}} \\',
         r'\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(l){8-10}',
         r'Set & \mSIM{} & \mACAL{} & Cost & Loss & CO$_2$ & Comply & Cost & Loss & CO$_2$ \\', r'\midrule']
    L += linhas_tab + [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(os.path.join(BASE, 'pesquisa', 'latex_ejor_c', 'tables', 'tab_acal.tex'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('tabela: tab_acal.tex')


if __name__ == '__main__':
    main()
