#!/usr/bin/env python3
"""
Equivalência entre buffers e simulação (08/10/2026, diário Entrada 84).

Crítica de revisor que este script responde: "não significativo" não prova
"equivalente". Para cada comparação pareada por instância (média das sementes, como
nas Tabelas 6 e 7), calcula a diferença relativa d_i (%) e reporta:
  - o estimador de Hodges–Lehmann (mediana das médias de Walsh) e o IC de 95%
    associado ao teste de Wilcoxon (aproximação normal; n = 54 instâncias);
  - para o CUSTO, o teste de equivalência TOST a 5%: equivalente se o IC de 90% cabe
    em (−Δ, +Δ), com Δ = 0,5%.
Escolha das margens (testada nos dados arquivados da v1, results_c_v1_cindom):
  - custo, Δ = 0,5%: a ordem de grandeza do próprio preço da confiabilidade (mediana
    +0,24%). Com Δ = 1%, até métodos significativamente diferentes (CAL, MOEA/D)
    seriam declarados equivalentes, ou seja, a margem não discriminaria;
  - hipervolume: SEM teste de equivalência. Na v1, até o MOEA/D, claramente pior
    (r = −0,65), fica dentro de ±4%, então nenhuma margem é defensável. Reporta-se só o
    IC, e a alegação é a mais fraca: "nenhuma diferença detectável".
Entrada: pesquisa/resultados_c/por_execucao.csv (analisar_c.py).
Saída: pesquisa/resultados_robustez/relatorio_equivalencia.txt, equivalencia.json e
       latex_ejor_c/tables/tab_equivalencia.tex (omitida com --sem-tabela).
Uso: python pesquisa/equivalencia.py [--csv caminho] [--out pasta] [--sem-tabela]
"""
import argparse, csv, json, os
from collections import defaultdict
import numpy as np
from scipy.stats import norm

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
TAB = os.path.join(BASE, 'pesquisa', 'latex_ejor_c', 'tables', 'tab_equivalencia.tex')
MARGEM = {'cm': 0.5, 'hv': None}
NOME = {'C-BUF90': r'\mBUFnc{0.90}', 'C-BUF90C': r'\mBUF{0.90}', 'C-BUF95C': r'\mBUF{0.95}',
        'C-CAL-BEST': r'\mCAL{}', 'C-MOEADC': r'\mMOEAD{}', 'C-DETC': r'\mDETC{}',
        'C-BUF95C95': r'\mBUF{0.95}', 'C-CAL95-BEST': r'\mCAL{}'}
PLANO = [('cm', 0.90, 'C-SIMC', ['C-BUF95C', 'C-BUF90C', 'C-CAL-BEST', 'C-MOEADC']),
         ('cm', 0.95, 'C-SIMC95', ['C-BUF95C95', 'C-CAL95-BEST']),
         ('hv', 0.90, 'C-SIMC', ['C-BUF95C', 'C-BUF90C', 'C-BUF90', 'C-CAL-BEST', 'C-MOEADC'])]


def hodges_lehmann(d, nivel):
    """Estimador HL e IC bilateral (nível dado) pelos quantis das médias de Walsh."""
    d = np.asarray(d, float); n = len(d)
    w = np.sort((d[:, None] + d[None, :])[np.triu_indices(n)] / 2.0)
    M = len(w)
    z = norm.ppf(1 - (1 - nivel) / 2)
    C = max(int(np.floor(M / 2 - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24))), 1)
    return float(np.median(w)), float(w[C - 1]), float(w[M - C])


def comparar(x, base, campo):
    ks = sorted(k for k in x if k in base and base[k] > 0)
    d = np.array([100 * (x[k] / base[k] - 1) for k in ks])
    hl, lo95, hi95 = hodges_lehmann(d, 0.95)
    _, lo90, hi90 = hodges_lehmann(d, 0.90)
    m = MARGEM[campo]
    eq = None if m is None else bool(lo90 > -m and hi90 < m)
    return dict(n=len(ks), hl=hl, ic95=(lo95, hi95), ic90=(lo90, hi90), equivalente=eq, margem=m)


def texto_eq(eq, sim='EQUIVALENTE', nao='não demonstrada'):
    return '--' if eq is None else (sim if eq else nao)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.join(BASE, 'pesquisa', 'resultados_c', 'por_execucao.csv'))
    ap.add_argument('--out', default=os.path.join(BASE, 'pesquisa', 'resultados_robustez'))
    ap.add_argument('--sem-tabela', action='store_true')
    a = ap.parse_args()
    R = list(csv.DictReader(open(a.csv, encoding='utf-8')))
    coluna = {'cm': 'custo_min_cumpre', 'hv': 'hv_cumpre'}

    def por_inst(alg, campo):
        d = defaultdict(list)
        for r in R:
            v = r[coluna[campo]]
            if r['algorithm'] == alg and v not in ('', 'nan') and np.isfinite(float(v)):
                d[r['instance']].append(float(v))
        return {k: float(np.mean(v)) for k, v in d.items()}

    os.makedirs(a.out, exist_ok=True)
    rel = open(os.path.join(a.out, 'relatorio_equivalencia.txt'), 'w', encoding='utf-8')

    def log(t=''):
        print(t, flush=True); rel.write(t + '\n')

    log('EQUIVALÊNCIA (diferença relativa pareada por instância, %; HL = Hodges–Lehmann)')
    res = []
    for campo, alpha, ref, algs in PLANO:
        base = por_inst(ref, campo)
        mg = f'margem ±{MARGEM[campo]:.1f}%' if MARGEM[campo] else 'sem teste de equivalência'
        tit = 'Custo mínimo que cumpre' if campo == 'cm' else 'Hipervolume das que cumprem'
        log(f'\n{tit}, α={alpha:.2f}, referência {ref}, {mg}')
        for alg in algs:
            x = por_inst(alg, campo)
            if not x:
                continue
            c = comparar(x, base, campo)
            c.update(campo=campo, alpha=alpha, alg=alg, ref=ref)
            res.append(c)
            log(f"  {alg:13s} n={c['n']:2d}  HL {c['hl']:+6.2f}  IC95 [{c['ic95'][0]:+6.2f}, {c['ic95'][1]:+6.2f}]  "
                f"IC90 [{c['ic90'][0]:+6.2f}, {c['ic90'][1]:+6.2f}]  {texto_eq(c['equivalente'])}")
    rel.close()
    json.dump(res, open(os.path.join(a.out, 'equivalencia.json'), 'w', encoding='utf-8'), indent=1)
    if a.sem_tabela:
        return

    lin = []
    for campo, alpha in (('cm', 0.90), ('cm', 0.95), ('hv', 0.90)):
        rr = [c for c in res if c['campo'] == campo and c['alpha'] == alpha]
        if not rr:
            continue
        tit = 'Cost of the cheapest compliant solution' if campo == 'cm' else 'Hypervolume of the compliant solutions'
        mg = (r', margin $\pm' + f'{MARGEM[campo]:.1f}' + r'\%$') if MARGEM[campo] else ', no equivalence margin'
        lin.append(r'\multicolumn{5}{@{}l}{\textit{' + tit + r', $\alpha=' + f'{alpha:.2f}' + '$' + mg + r'}} \\')
        for c in rr:
            lin.append(f"{NOME[c['alg']]} & {c['n']} & {c['hl']:+.2f} & [{c['ic95'][0]:+.2f}, {c['ic95'][1]:+.2f}] & "
                       + texto_eq(c['equivalente'], 'yes', 'no') + r' \\')
    tex = (r"""\begin{table}[htbp]
\centering
\caption{Equivalence with \mSIM{}: Hodges--Lehmann estimate of the paired relative difference per instance (\%; mean over seeds, as in Tables~\ref{tab:cost} and~\ref{tab:hv}), its 95\% confidence interval and, for cost, the two one-sided tests (TOST) at the 5\% level, which declare equivalence when the 90\% interval lies within the margin. No margin is set for hypervolume, whose relative differences are small for all methods.}
\label{tab:equiv}
\small
\begin{tabular}{@{}l r r c c@{}}
\toprule
Method & $n$ & HL (\%) & 95\% CI (\%) & Equivalent \\
\midrule
""" + '\n'.join(lin) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    open(TAB, 'w', encoding='utf-8').write(tex)


if __name__ == '__main__':
    main()
