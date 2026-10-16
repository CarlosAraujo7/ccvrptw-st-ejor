#!/usr/bin/env python3
"""
Análise do refinamento por simulação (11/10/2026, diário Entrada 88).

Visão do decisor: em cada execução, o plano que seria implantado (a solução mais
barata certificada de cada método; para os refinados, o plano final depois da porteira),
avaliado nos 1.000 cenários comuns. Métodos:
  SIM+C, BUF0.95+C, CAL (mais barata certificada entre os degraus 0,70/0,80/0,90/0,95),
  BUF0.95+R e BUF0.95+R-LCB (refinamento a partir do BUF0.95+C, porteira pontual / LCB),
  SIM+C+R (o mesmo refinamento a partir do SIM+C, para separar o efeito do refinamento
  do efeito do ponto de partida).
Por instância (média das sementes): Δ% de custo e Δ p.p. de perda de qualidade contra
o SIM+C, com Hodges–Lehmann, IC 95% e Wilcoxon com Holm; e % de execuções que cumprem α.
Uso: python pesquisa/analisar_refinamento.py --dir pasta_com_refinamento_*.json
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equivalencia import hodges_lehmann   # noqa: E402

ALPHA = 0.90


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - k) * p[i])); adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--tabela', help='grava a tabela LaTeX do artigo neste caminho (Entrada 93)')
    a = ap.parse_args()
    B = {(x['instance'], x['seed']): x for x in json.load(open(os.path.join(a.dir, 'refinamento_C-BUF95C.json'), encoding='utf-8'))}
    S = {}; CR = {}
    fs = os.path.join(a.dir, 'refinamento_C-SIMC.json')
    if os.path.exists(fs):
        S = {(x['instance'], x['seed']): x for x in json.load(open(fs, encoding='utf-8'))}
    fc = os.path.join(a.dir, 'refinamento_CAL.json')
    if os.path.exists(fc):
        CR = {(x['instance'], x['seed']): x for x in json.load(open(fc, encoding='utf-8'))}
    planos = defaultdict(dict)   # método -> (inst, seed) -> (f1, p, q, f3)
    for k, x in B.items():
        if 'C-SIMC_f1' not in x:
            continue
        g = lambda pre: (x[f'{pre}_f1'], x[f'{pre}_p_oos'], x[f'{pre}_q_oos'], x[f'{pre}_f3'])   # noqa: E731
        planos['SIM+C'][k] = g('C-SIMC'); planos['BUF0.95+C'][k] = g('base')
        planos['BUF0.95+R'][k] = g('fin'); planos['BUF0.95+R-LCB'][k] = g('finl')
        rung = [g('base')] + [g(r) for r in ('C-BUF70C', 'C-BUF80C', 'C-BUF90C') if f'{r}_f1' in x]
        planos['CAL'][k] = min(rung, key=lambda t: t[0])
        if k in S:
            planos['SIM+C+R'][k] = (S[k]['fin_f1'], S[k]['fin_p_oos'], S[k]['fin_q_oos'], S[k]['fin_f3'])
        if k in CR:
            planos['CAL+R'][k] = (CR[k]['fin_f1'], CR[k]['fin_p_oos'], CR[k]['fin_q_oos'], CR[k]['fin_f3'])
            planos['CAL+R-LCB'][k] = (CR[k]['finl_f1'], CR[k]['finl_p_oos'], CR[k]['finl_q_oos'], CR[k]['finl_f3'])
    # Orçamento igual ao do CAL (4 buscas): melhor de 4 execuções do SIM+C (sementes s..s+3,
    # mod 10), escolhida pelo custo da escolha certificada ANTES do refinamento, como no CAL;
    # SIM4+R usa o plano refinado da semente escolhida.
    if S:
        for (inst, sd) in list(planos['SIM+C']):
            grupo = [(inst, (sd + j) % 10) for j in range(4)]
            if not all(g in S for g in grupo):
                continue
            g = min(grupo, key=lambda g: S[g]['base_f1'])
            planos['SIM4'][(inst, sd)] = (S[g]['base_f1'], S[g]['base_p_oos'], S[g]['base_q_oos'], S[g]['base_f3'])
            planos['SIM4+R'][(inst, sd)] = (S[g]['fin_f1'], S[g]['fin_p_oos'], S[g]['fin_q_oos'], S[g]['fin_f3'])
    ks = sorted(set.intersection(*[set(v) for v in planos.values()]))
    out = [f'REFINAMENTO — visão do decisor ({len(ks)} execuções em comum; avaliação em 1.000 cenários comuns; α = {ALPHA})', '']
    out.append(f"{'método':16s} {'cumpre':>7s} {'custo médio':>12s} {'perda média':>12s} {'CO2 médio':>10s}")
    for m, d in planos.items():
        v = np.array([d[k] for k in ks])
        out.append(f'{m:16s} {100*np.mean(v[:, 1] >= ALPHA):6.1f}% {v[:, 0].mean():12.1f} {100*v[:, 2].mean():11.3f}% {v[:, 3].mean():10.1f}')
    for nivel in ('MEDIUM', 'HIGH'):
        kk = [k for k in ks if k[0].endswith(nivel)]
        out.append(f'  {nivel}: ' + '  '.join(f"{m} {100*np.mean([planos[m][k][1] >= ALPHA for k in kk]):.1f}%" for m in planos))
    out.append('\nPareado por instância (média das sementes) contra SIM+C: HL [IC 95%], Wilcoxon com Holm')
    comp = defaultdict(dict)   # método -> medida -> (HL, p_Holm), para a tabela do artigo
    for idx, rot, esc in ((0, 'custo (%)', 'rel'), (2, 'perda de qualidade (p.p.)', 'abs'), (3, 'CO2 (%)', 'rel')):
        res = []
        for m in planos:
            if m == 'SIM+C':
                continue
            pi = defaultdict(list); ps = defaultdict(list)
            for k in ks:
                pi[k[0]].append(planos[m][k][idx]); ps[k[0]].append(planos['SIM+C'][k][idx])
            I = sorted(pi)
            x = np.array([np.mean(pi[i]) for i in I]); y = np.array([np.mean(ps[i]) for i in I])
            d = 100 * (x / y - 1) if esc == 'rel' else 100 * (x - y)
            hl, lo, hi = hodges_lehmann(d, 0.95)
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            res.append((m, len(I), hl, lo, hi, p))
        adj = holm([r[-1] for r in res])
        out.append(f'  {rot}:')
        for (m, n, hl, lo, hi, p), pa in zip(res, adj):
            out.append(f'    {m:16s} n={n:2d}  HL {hl:+7.3f}  [{lo:+7.3f}, {hi:+7.3f}]  p_Holm={pa:.3g}')
            comp[m][idx] = (hl, pa)
    # Orçamento igual (14/10/2026, Entrada 93): pareado por instância, CAL(+R) contra SIM4(+R), nas quatro
    # medidas; Holm dentro de cada par (quatro medidas). Na v1 esta comparação foi feita à parte (Entrada 88).
    for a_, b_ in (('CAL+R', 'SIM4+R'), ('CAL', 'SIM4')):
        if a_ not in planos or b_ not in planos:
            continue
        res = []
        for idx, rot, esc in ((0, 'custo (%)', 'rel'), (2, 'perda (p.p.)', 'abs'), (3, 'CO2 (%)', 'rel'), (1, 'P_oos (p.p.)', 'abs')):
            pi = defaultdict(list); ps = defaultdict(list)
            for k in ks:
                pi[k[0]].append(planos[a_][k][idx]); ps[k[0]].append(planos[b_][k][idx])
            I = sorted(pi)
            x = np.array([np.mean(pi[i]) for i in I]); y = np.array([np.mean(ps[i]) for i in I])
            d = 100 * (x / y - 1) if esc == 'rel' else 100 * (x - y)
            hl, lo, hi = hodges_lehmann(d, 0.95)
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            res.append((rot, len(I), hl, lo, hi, p))
        adj = holm([r[-1] for r in res])
        out.append(f'\nOrçamento igual (4 buscas), pareado por instância: {a_} contra {b_}; HL [IC 95%], Wilcoxon (p) e Holm nas 4 medidas')
        for (rot, n, hl, lo, hi, p), pa in zip(res, adj):
            out.append(f'    {rot:14s} n={n:2d}  HL {hl:+7.3f}  [{lo:+7.3f}, {hi:+7.3f}]  p={p:.3g}  p_Holm={pa:.3g}')
    txt = '\n'.join(out); print(txt)
    open(os.path.join(a.dir, 'relatorio_analise_refinamento.txt'), 'w', encoding='utf-8').write(txt + '\n')
    if a.tabela:
        tabela(planos, ks, comp, a.tabela)


ROTULO = {'SIM+C': r'\mSIM{}', 'BUF0.95+C': r'\mBUF{0.95}', 'CAL': r'\mCAL{}', 'SIM4': r'\mSIM{}, best of 4',
          'BUF0.95+R': r'\mBUF{0.95}+R', 'SIM+C+R': r'\mSIM{}+R', 'CAL+R': r'\mCAL{}+R', 'SIM4+R': r'\mSIM{}, best of 4, +R'}


def tabela(planos, ks, comp, caminho):
    """Tabela do artigo (Entrada 93): cumprimento (todas / HIGH) e diferenças pareadas contra o SIM+C
    (Hodges–Lehmann; * = p_Holm < 0,05, Holm dentro de cada medida)."""
    alto = [k for k in ks if k[0].endswith('HIGH')]

    def cel(m, idx):
        if m == 'SIM+C':
            return '--'
        hl, pa = comp[m][idx]
        txt = '0.00' if abs(hl) < 0.005 else f'{hl:+.2f}'
        return txt + ('$^*$' if pa < 0.05 else '')
    L = [r'\begin{table}[htbp]', r'\centering',
         r'\caption{Simulation-based refinement on the test set: the plan a decision-maker would choose (the cheapest certified one), '
         f'{len(ks)} runs (36 instances with medium or high uncertainty $\\times$ 10 seeds), evaluated on 1{{,}}000 common scenarios. '
         r'Complies: \% of runs whose plan meets $\alpha=0.90$ out of sample. Differences to \mSIM{}: Hodges--Lehmann estimate of the '
         r'paired difference per instance (mean over seeds); $^*$: $p_{\mathrm{Holm}}<0.05$ (Wilcoxon signed-rank test, Holm correction '
         r'within each measure). Best of 4: four runs of \mSIM{}, the budget of \mCAL{}.}',
         r'\label{tab:refine}', r'\small', r'\begin{tabular}{@{}l rr rrr@{}}', r'\toprule',
         r' & \multicolumn{2}{c}{Complies (\%)} & \multicolumn{3}{c}{Difference to \mSIM{}} \\',
         r'\cmidrule(lr){2-3}\cmidrule(l){4-6}',
         r'Method & all & high & Cost (\%) & Quality loss (p.p.) & CO$_2$ (\%) \\', r'\midrule']
    for grupo in (('SIM+C', 'BUF0.95+C', 'CAL', 'SIM4'), ('BUF0.95+R', 'SIM+C+R', 'CAL+R', 'SIM4+R')):
        for m in grupo:
            if m not in planos:
                continue
            ca = 100 * np.mean([planos[m][k][1] >= ALPHA for k in ks])
            ch = 100 * np.mean([planos[m][k][1] >= ALPHA for k in alto])
            L.append(f'{ROTULO[m]} & {ca:.1f} & {ch:.1f} & {cel(m, 0)} & {cel(m, 2)} & {cel(m, 3)} \\\\')
        L.append(r'\midrule')
    L[-1] = r'\bottomrule'
    L += [r'\end{tabular}', r'\end{table}']
    open(caminho, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print(f'tabela: {caminho}')


if __name__ == '__main__':
    main()
