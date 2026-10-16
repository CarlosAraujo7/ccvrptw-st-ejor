#!/usr/bin/env python3
"""
Gera latex_ejor_c/tables/tab_robustez.tex (08/10/2026, diário Entrada 81) a partir de
  pesquisa/resultados_robustez/relatorio_robustez.txt  (robustez_revisor.py: δ e correlação)
  pesquisa/resultados_robustez/decisor.json             (escolha_decisor.py: escolha do decisor)
Rode os dois scripts antes deste.
"""
import json, os, re

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
RR = os.path.join(BASE, 'pesquisa', 'resultados_robustez')
OUT = os.path.join(BASE, 'pesquisa', 'latex_ejor_c', 'tables', 'tab_robustez.tex')
METODOS = [('C-DETC', r'\mDETC{}'), ('C-BUF90C', r'\mBUF{0.90}'), ('C-BUF95C', r'\mBUF{0.95}'),
           ('CAL', r'\mCAL{}'), ('C-SIMC', r'\mSIM{}'), ('C-MOEADC', r'\mMOEAD{}')]

rob = {}
nivel = None
for linha in open(os.path.join(RR, 'relatorio_robustez.txt'), encoding='utf-8'):
    t = linha.strip()
    if t in ('MEDIUM', 'HIGH'):
        nivel = t; continue
    m = re.match(r'(C-\w+)\s+ind_0:\s*([\d.]+)%.*ind_15:\s*([\d.]+)%.*cor_5:\s*([\d.]+)%', t)
    if m and nivel:
        rob[(m.group(1), nivel)] = {'d0': float(m.group(2)), 'd15': float(m.group(3)), 'cor': float(m.group(4))}
assert all(v['d15'] == 100.0 for v in rob.values()), 'texto da legenda supõe 100% com δ=15'
dec = json.load(open(os.path.join(RR, 'decisor.json'), encoding='utf-8'))['0.90']


def f(x):
    return '--' if x is None else f'{x:.1f}'


linhas = []
for k, nome in METODOS:
    r = [rob.get((k, n), {}) for n in ('MEDIUM', 'HIGH')]
    dt, dh = dec['TODOS'][k], dec['HIGH'][k]
    cel = [f(r[0].get('d0')), f(r[1].get('d0')), f(r[0].get('cor')), f(r[1].get('cor')),
           f(dt['escolha_cumpre_pct_exec']), f(dt['lcb_cumpre_pct_exec']),
           f(dh['escolha_cumpre_pct_exec']), f(dh['lcb_cumpre_pct_exec'])]
    linhas.append(nome + ' & ' + ' & '.join(cel) + r' \\')

tex = r"""\begin{table}[htbp]
\centering
\caption{Robustness of punctuality ($\alpha=0.90$; percentage of runs). Left: the front contains a solution that complies out of sample when the lateness tolerance is stricter than planned ($\delta=0$) or when travel times share a common day factor ($\rho=0.5$, $\delta=5$\,min); 18 instances with $n=50$, 10 seeds; with $\delta=15$\,min every method complies in every run. Right: the plan a decision-maker would choose, the cheapest certified one, exists and complies out of sample, with certification against the point estimate or against a one-sided 95\% lower confidence bound (LCB); 54 instances, 10 seeds. --: not evaluated.}
\label{tab:robust}
\small
\begin{tabular}{@{}l rr rr rr rr@{}}
\toprule
 & \multicolumn{2}{c}{$\delta=0$} & \multicolumn{2}{c}{$\rho=0.5$} & \multicolumn{2}{c}{Chosen plan, all} & \multicolumn{2}{c}{Chosen plan, HIGH} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Method & MEDIUM & HIGH & MEDIUM & HIGH & Point & LCB & Point & LCB \\
\midrule
""" + '\n'.join(linhas) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(OUT, 'w', encoding='utf-8').write(tex)
print(tex)
