#!/usr/bin/env python3
"""
Análise do teste de escala (18/10/2026, diário Entrada 116), conforme o desenho da Entrada 114.

Planos escolhidos (certificada mais barata, P_cert >= α) de cada execução, medidos nos 1.000 cenários comuns:
  motor     SIM+C, BUF0.90+C, BUF0.95+C, CAL (4 buscas), ACAL (C-ACALC-B, 1 busca)      results_c_escala
  +R        refinamento (porteira pontual, campos fin_*) a partir de CAL, ACAL e SIM+C    resultados_escala/refinamento
  PyVRP     PY-CAL (degrau certificado mais barato de 0,80/0,90/0,95) e PY-CAL+R          resultados_escala/pipeline_escala.jsonl
Comparações por instância (média das sementes 0-2 de cada método): Hodges-Lehmann com IC 95%, Wilcoxon e Holm nas
quatro medidas (custo %, perda p.p., CO2 %, P fora da amostra p.p.); grupos n = 200, n = 400 e todos.
Uso: python pesquisa/escala/analisar_escala.py
"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(BASE, 'pesquisa'))
import calibracao_preditiva_base as b      # noqa: E402
from equivalencia import hodges_lehmann    # noqa: E402
from analisar_refinamento import holm      # noqa: E402

ALPHA = b.ALPHA
RES = 'results_c_escala'
OUT = os.path.join(BASE, 'pesquisa', 'resultados_escala')
TAB = os.path.join(BASE, 'pesquisa', 'latex_ejor_c', 'tables', 'tab_escala.tex')
MEDIDAS = (('custo (%)', 0, 'rel'), ('perda (p.p.)', 2, 'abs'), ('CO2 (%)', 3, 'rel'), ('P_oos (p.p.)', 1, 'abs'))
GRUPOS = (('n = 200', lambda i: '_200_' in i), ('n = 400', lambda i: '_400_' in i), ('todas', lambda i: True))


def tup(s):
    return None if s is None else (s['f1'], s['p_on'], s['q_oos'], s['f3'])


LCB = ALPHA + 1.645 * np.sqrt(ALPHA * (1 - ALPHA) / 500)     # regra mais estrita da Seção 6.5 (análise secundária)


def carregar_lcb():
    """Escolha com a regra do limite inferior de confiança: certificada mais barata com P_cert >= LCB."""
    R = defaultdict(dict)
    for a in ['C-SIMC', 'C-ACALC-B'] + list(b.DEG):
        for f in glob.glob(os.path.join(BASE, RES, a, '*_oosC.json')):
            d = json.load(open(f, encoding='utf-8')); m = d['meta']
            cert = [s for s in d['solutions'] if s['feasible'] and (s.get('p_cert') or 0) >= LCB]
            R[(m['instance'], m['seed'])][a] = min(cert, key=lambda s: s['f1']) if cert else None
    return R


def carregar_planos():
    """{método: {(inst, seed): (f1, p_oos, q_oos, f3) ou None}}"""
    R = b.carregar(RES)
    for f in glob.glob(os.path.join(BASE, RES, 'C-ACALC-B', '*_oosC.json')):
        d = json.load(open(f, encoding='utf-8')); m = d['meta']
        cert = [s for s in d['solutions'] if s['feasible'] and (s.get('p_cert') or 0) >= ALPHA]
        R[(m['instance'], m['seed'])]['C-ACALC-B'] = min(cert, key=lambda s: s['f1']) if cert else None
    P = defaultdict(dict)
    for k, d in R.items():
        P['SIM+C'][k] = tup(d.get('C-SIMC'))
        P['BUF0.90+C'][k] = tup(d.get('C-BUF90C'))
        P['BUF0.95+C'][k] = tup(d.get('C-BUF95C'))
        P['CAL'][k] = tup(b.melhor([d.get(a) for a in b.DEG]))
        P['ACAL'][k] = tup(d.get('C-ACALC-B'))
        P['SIM3'][k] = tup(b.regra_sim(3)(R, k[0], k[1], [0, 1, 2])[0])          # melhor de 3 (orçamento de 3 buscas)
    for k, d in carregar_lcb().items():
        P['SIM+C LCB'][k] = tup(d.get('C-SIMC'))
        P['CAL LCB'][k] = tup(b.melhor([d.get(a) for a in b.DEG]))
        P['ACAL LCB'][k] = tup(d.get('C-ACALC-B'))
    for arq, rot in (('refinamento_CAL.json', 'CAL+R'), ('refinamento_C-ACALC-B.json', 'ACAL+R'),
                     ('refinamento_C-SIMC.json', 'SIM+C+R')):
        for x in json.load(open(os.path.join(OUT, 'refinamento', arq), encoding='utf-8')):
            P[rot][(x['instance'], x['seed'])] = (x['fin_f1'], x['fin_p_oos'], x['fin_q_oos'], x['fin_f3'])
    por = defaultdict(dict)
    for L in open(os.path.join(OUT, 'pipeline_escala.jsonl'), encoding='utf-8'):
        x = json.loads(L)
        por[(x['instance'], x['seed'])][x['q']] = x
    for k, d in por.items():
        cert = [x for x in d.values() if x['viavel'] and x['base_p_cert'] >= ALPHA]
        e = min(cert, key=lambda x: x['base_f1']) if cert else None
        P['PY-CAL'][k] = None if e is None else (e['base_f1'], e['base_p_oos'], e['base_q_oos'], e['base_f3'])
        P['PY-CAL+R'][k] = None if e is None else (e['fin_f1'], e['fin_p_oos'], e['fin_q_oos'], e['fin_f3'])
        x = d.get(0.95)
        ok = x is not None and x['viavel'] and x['base_p_cert'] >= ALPHA
        P['PY-BUF0.95'][k] = (x['base_f1'], x['base_p_oos'], x['base_q_oos'], x['base_f3']) if ok else None
        P['PY-BUF0.95+R'][k] = (x['fin_f1'], x['fin_p_oos'], x['fin_q_oos'], x['fin_f3']) if ok else None
    return P


def por_inst(D, filtro, idx):
    pi = defaultdict(list)
    for (i, _), t in D.items():
        if t is not None and filtro(i):
            pi[i].append(t[idx])
    return {i: float(np.mean(v)) for i, v in pi.items()}


def comparar(P, m, ref, filtro):
    res = []
    for nm, idx, esc in MEDIDAS:
        x, y = por_inst(P[m], filtro, idx), por_inst(P[ref], filtro, idx)
        I = sorted(set(x) & set(y))
        a = np.array([x[i] for i in I]); c = np.array([y[i] for i in I])
        dd = 100 * (a / c - 1) if esc == 'rel' else 100 * (a - c)
        hl, lo, hi = hodges_lehmann(dd, 0.95)
        p = wilcoxon(dd).pvalue if np.any(np.abs(dd) > 1e-12) else 1.0
        res.append([nm, len(I), hl, lo, hi, p, int(np.sum(dd < -1e-12)), int(np.sum(dd > 1e-12))])
    for r, pa in zip(res, holm([r[5] for r in res])):
        r.append(pa)
    return res


def tempos():
    T = defaultdict(list)
    for f in glob.glob(os.path.join(BASE, RES, 'C-*', '*_metrics.json')):
        d = json.load(open(f, encoding='utf-8'))
        T[(d['algorithm'], d['size'])].append(d['elapsed_s'])
    for L in open(os.path.join(OUT, 'pipeline_escala.jsonl'), encoding='utf-8'):
        x = json.loads(L)
        T[('PyVRP (1 degrau)', x['n'])].append(x['t_solve_s'] + x['cpu_ref_s'])
    for arq, rot in (('refinamento_CAL.json', 'refino CAL'), ('refinamento_C-ACALC-B.json', 'refino ACAL')):
        for x in json.load(open(os.path.join(OUT, 'refinamento', arq), encoding='utf-8')):
            T[(rot, int(x['instance'].split('_')[2]))].append(x['cpu_s'])
    return {k: float(np.mean(v)) for k, v in T.items()}


def tabela_tex(resumo, T):
    """Tabela do artigo: diferenças contra o SIM+C por tamanho (HL; * = p_Holm < 0,05), cumprimento e tempo por plano."""
    def tempo(m, n):
        t = {'SIM+C': T[('C-SIMC', n)], 'BUF0.95+C': T[('C-BUF95C', n)],
             'CAL': sum(T[(a, n)] for a in b.DEG), 'ACAL': T[('C-ACALC-B', n)],
             'PY-BUF0.95+R': T[('PyVRP (1 degrau)', n)], 'PY-CAL+R': 3 * T[('PyVRP (1 degrau)', n)]}
        t['CAL+R'] = t['CAL'] + T[('refino CAL', n)]; t['ACAL+R'] = t['ACAL'] + T[('refino ACAL', n)]
        return t[m] / 60

    def cel(g, m, nm):
        d = resumo['comp'][f'{g}|{m}|SIM+C|{nm}']
        return ('0.00' if abs(d['hl']) < 0.005 else f'{d["hl"]:+.2f}') + ('$^*$' if d['p_holm'] < 0.05 else '')
    NOME = [('SIM+C', r'\mSIM{}'), ('BUF0.95+C', r'\mBUF{0.95}'), ('CAL', r'\mCAL{}'), ('ACAL', r'\mACAL{}'),
            ('CAL+R', r'\mCAL{}+R'), ('ACAL+R', r'\mACAL{}+R'), ('PY-BUF0.95+R', r'PyVRP, BUF$_{0.95}$+C+R'),
            ('PY-CAL+R', r'PyVRP, CAL+R')]
    L = [r'\begin{table}[htbp]', r'\centering',
         r'\caption{Scale test: 24 instances with $n=200$ and $n=400$ customers (six families, medium and high uncertainty), '
         r'three seeds. Hodges--Lehmann estimate of the paired difference per instance against \mSIM{} (mean over seeds; '
         r'12 instances per size); $^*$: $p_{\mathrm{Holm}}<0.05$ (Wilcoxon signed-rank test, Holm correction over the four '
         r'measures, including punctuality). Complies: \% of the chosen plans meeting $\alpha=0.90$ out of sample. '
         r'Time: mean computing time per chosen plan, $n=200$ / $n=400$.}',
         r'\label{tab:scale}', r'\footnotesize\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{@{}l rrrr rrrr r@{}}', r'\toprule',
         r' & \multicolumn{4}{c}{$n=200$} & \multicolumn{4}{c}{$n=400$} & \\', r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
         r'Method & Cost & Loss & CO$_2$ & Compl. & Cost & Loss & CO$_2$ & Compl. & Time \\',
         r' & (\%) & (p.p.) & (\%) & (\%) & (\%) & (p.p.) & (\%) & (\%) & (min) \\', r'\midrule']
    for m, rot in NOME:
        c = []
        for g in ('n = 200', 'n = 400'):
            c += (['--'] * 3 if m == 'SIM+C' else [cel(g, m, nm) for nm, _, _ in MEDIDAS[:3]])
            c.append(f'{resumo["cumpre"][f"{g}|{m}"]:.1f}')
        L.append(f'{rot} & ' + ' & '.join(c) + f' & {tempo(m, 200):.1f} / {tempo(m, 400):.1f} \\\\')
        if m in ('ACAL', 'ACAL+R'):
            L.append(r'\midrule')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(TAB, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print(f'tabela: {TAB}')


def main():
    P = carregar_planos()
    out = ['TESTE DE ESCALA — 24 instâncias (n = 200 e 400), sementes 0-2; escolha = certificada mais barata; '
           f'α = {ALPHA}; 1.000 cenários comuns']
    resumo = {'cumpre': {}, 'sem_plano': {}, 'comp': {}, 'tempo': {}}
    metodos = ['SIM+C', 'BUF0.90+C', 'BUF0.95+C', 'CAL', 'ACAL', 'SIM+C+R', 'CAL+R', 'ACAL+R', 'PY-CAL', 'PY-CAL+R',
               'SIM3', 'PY-BUF0.95', 'PY-BUF0.95+R', 'SIM+C LCB', 'CAL LCB', 'ACAL LCB']
    for g, filtro in GRUPOS:
        out.append(f'\n== {g} ==')
        for m in metodos:
            ks = [k for k in P[m] if filtro(k[0])]
            v = np.array([P[m][k] for k in ks if P[m][k] is not None])
            sem = sum(1 for k in ks if P[m][k] is None)
            cum = 100 * np.mean(v[:, 1] >= ALPHA)
            resumo['cumpre'][f'{g}|{m}'] = cum; resumo['sem_plano'][f'{g}|{m}'] = sem
            fal = v[v[:, 1] < ALPHA, 1]
            out.append(f'  {m:12s} execuções {len(ks):3d} sem plano {sem:2d}  cumpre {cum:5.1f}%  P média {v[:, 1].mean():.3f} '
                       f'(mín {v[:, 1].min():.3f}; falhas: P média {fal.mean() if len(fal) else float("nan"):.3f})  '
                       f'custo {v[:, 0].mean():9.1f}  perda {100 * v[:, 2].mean():.3f}%  CO2 {v[:, 3].mean():.1f}')
        for m, ref in [(m, 'SIM+C') for m in metodos[1:]] + [('ACAL', 'CAL'), ('ACAL+R', 'CAL+R'), ('PY-CAL+R', 'CAL+R'),
                                                             ('CAL+R', 'CAL'), ('PY-CAL+R', 'PY-CAL'), ('PY-CAL+R', 'SIM3'),
                                                             ('CAL', 'SIM3'), ('ACAL LCB', 'SIM+C LCB'),
                                                             ('CAL LCB', 'SIM+C LCB')]:
            res = comparar(P, m, ref, filtro)
            out.append(f'  {m} contra {ref} (por instância; HL [IC 95%]; p; p_Holm; menores/maiores)')
            for nm, ni, hl, lo, hi, p, mn, mx, pa in res:
                out.append(f'      {nm:14s} n={ni:2d}  HL {hl:+7.3f} [{lo:+7.3f}, {hi:+7.3f}]  p={p:.3g}  p_Holm={pa:.3g}  ({mn}/{mx})')
                resumo['comp'][f'{g}|{m}|{ref}|{nm}'] = dict(n=ni, hl=hl, lo=lo, hi=hi, p=p, p_holm=pa, menores=mn, maiores=mx)
    T = tempos()
    out.append('\n== tempo médio por execução (s) ==')
    for k in sorted(T):
        out.append(f'  {k[0]:18s} n = {k[1]}: {T[k]:7.1f}')
        resumo['tempo'][f'{k[0]}|{k[1]}'] = T[k]
    txt = '\n'.join(out); print(txt)
    open(os.path.join(OUT, 'relatorio_escala.txt'), 'w', encoding='utf-8').write(txt + '\n')
    json.dump(resumo, open(os.path.join(OUT, 'resumo_escala.json'), 'w', encoding='utf-8'), indent=1)
    tabela_tex(resumo, T)


if __name__ == '__main__':
    main()
