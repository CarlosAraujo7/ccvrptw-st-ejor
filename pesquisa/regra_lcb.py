#!/usr/bin/env python3
"""
Regra pontual × regra LCB na escolha do plano e na porteira do refinamento (18/10/2026, diário Entrada 117).

Regra pontual: plano = certificada mais barata (P_cert >= α); porteira do refinamento P_cert >= α.
Regra LCB:     plano = mais barata com P_cert >= LCB = α + 1,645·√(α(1−α)/500); se nenhuma, a mais pontual entre as
               certificadas em α (fallback); porteira P_cert >= LCB (refinamento_<partida>_lcb.json, campos finl_*).
Conjuntos: benchmark (MEDIUM/HIGH), SP, SP com horário, Atenas, escala. Métodos: SIM+C, BUF0.95+C, CAL, ACAL, CAL+R,
ACAL+R, SIM+C+R e PyVRP, CAL+R (benchmark, SP, escala). Medidas nos 1.000 cenários comuns.
Critério (fixado antes): ver diário Entrada 117. Uso: python pesquisa/regra_lcb.py
"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(BASE, 'pesquisa'))
from equivalencia import hodges_lehmann    # noqa: E402
from analisar_refinamento import holm      # noqa: E402

ALPHA = 0.90
LCB = ALPHA + 1.645 * np.sqrt(ALPHA * (1 - ALPHA) / 500)
OUT = os.path.join(BASE, 'pesquisa', 'resultados_regra_lcb')
DEG = ['C-BUF70C', 'C-BUF80C', 'C-BUF90C', 'C-BUF95C']
MEDIDAS = (('custo (%)', 0, 'rel'), ('perda (p.p.)', 2, 'abs'), ('CO2 (%)', 3, 'rel'), ('P_oos (p.p.)', 1, 'abs'))
PIPE = os.path.join(BASE, 'pesquisa', 'resultados_pyvrp_pipeline', 'pipeline.jsonl')
CONJ = [
    ('Benchmark', 'results_c', 'results_c_acal_bench', 'pesquisa/resultados_refinamento',
     'pesquisa/resultados_acal/refinamento_bench', (PIPE, lambda i: '_SP' not in i), lambda i: i.endswith(('_MEDIUM', '_HIGH'))),
    ('São Paulo', 'results_c_sp', 'results_c_acal_sp', 'pesquisa/resultados_caso_sp/refinamento',
     'pesquisa/resultados_acal/refinamento_sp', (PIPE, lambda i: '_SP' in i), lambda i: True),
    ('São Paulo, time-dependent', 'results_c_sp_td', 'results_c_acal_sp_td', 'pesquisa/resultados_caso_sp_td/refinamento',
     'pesquisa/resultados_acal/refinamento_sp_td', None, lambda i: True),
    ('Athens', 'results_c_atenas', 'results_c_acal_atenas', 'pesquisa/resultados_caso_atenas/refinamento',
     'pesquisa/resultados_acal/refinamento_atenas', None, lambda i: True),
    ('Scale', 'results_c_escala', 'results_c_escala', 'pesquisa/resultados_escala/refinamento',
     'pesquisa/resultados_escala/refinamento',
     (os.path.join(BASE, 'pesquisa', 'resultados_escala', 'pipeline_escala.jsonl'), lambda i: True), lambda i: True),
]


def escolher(sols, regra):
    """(plano, atingiu a regra) — plano = dict do oosC ou None."""
    cert = [s for s in sols if s['feasible'] and (s.get('p_cert') or 0) >= ALPHA]
    if not cert:
        return None, False
    if regra == 'pontual':
        return min(cert, key=lambda s: s['f1']), True
    forte = [s for s in cert if s['p_cert'] >= LCB]
    if forte:
        return min(forte, key=lambda s: s['f1']), True
    return max(cert, key=lambda s: (s['p_cert'], -s['f1'])), False


def frentes(res, alg, filtro):
    F = {}
    for f in glob.glob(os.path.join(BASE, res, alg, '*_oosC.json')):
        d = json.load(open(f, encoding='utf-8')); m = d['meta']
        if filtro(m['instance']):
            F[(m['instance'], m['seed'])] = d['solutions']
    return F


def tup(s):
    return (s['f1'], s['p_on'], s['q_oos'], s['f3'])


def carregar(conj):
    nome, orig, acal, rdir, adir, pipe, filtro = conj
    P = {r: defaultdict(dict) for r in ('pontual', 'lcb')}       # regra -> método -> (inst, seed) -> tupla
    FB = {r: defaultdict(int) for r in ('pontual', 'lcb')}        # regra -> método -> nº de fallbacks
    fr = {a: frentes(orig, a, filtro) for a in ['C-SIMC'] + DEG}
    fr['C-ACALC-B'] = frentes(acal, 'C-ACALC-B', filtro)
    for regra in P:
        for m, algs in (('SIM+C', ['C-SIMC']), ('BUF0.95+C', ['C-BUF95C']), ('CAL', DEG), ('ACAL', ['C-ACALC-B'])):
            ks = set().union(*[set(fr[a]) for a in algs])
            for k in ks:
                s, ok = escolher([x for a in algs for x in fr[a].get(k, [])], regra)
                if s is not None:
                    P[regra][m][k] = tup(s)
                    FB[regra][m] += (not ok)
    for m, d, arq in (('CAL+R', rdir, 'refinamento_CAL'), ('SIM+C+R', rdir, 'refinamento_C-SIMC'),
                      ('ACAL+R', adir, 'refinamento_C-ACALC-B')):
        for regra, suf, g in (('pontual', '', 'fin'), ('lcb', '_lcb', 'finl')):
            p = os.path.join(BASE, d, arq + suf + '.json')
            if not os.path.exists(p):
                continue
            for x in json.load(open(p, encoding='utf-8')):
                if filtro(x['instance']):
                    P[regra][m][(x['instance'], x['seed'])] = (x[f'{g}_f1'], x[f'{g}_p_oos'], x[f'{g}_q_oos'], x[f'{g}_f3'])
                    if regra == 'lcb':
                        FB[regra][m] += (not x.get('lcb_ok', True))
    if pipe:
        por = defaultdict(dict)
        for L in open(pipe[0], encoding='utf-8'):
            x = json.loads(L)
            if x['viavel'] and pipe[1](x['instance']):
                por[(x['instance'], x['seed'])][x['q']] = x
        for k, d in por.items():
            cert = [x for x in d.values() if x['base_p_cert'] >= ALPHA]
            if not cert:
                continue
            e = min(cert, key=lambda x: x['base_f1'])
            P['pontual']['PyVRP, CAL+R'][k] = (e['fin_f1'], e['fin_p_oos'], e['fin_q_oos'], e['fin_f3'])
            forte = [x for x in cert if x['base_p_cert'] >= LCB]
            e = min(forte, key=lambda x: x['base_f1']) if forte else max(cert, key=lambda x: (x['base_p_cert'], -x['base_f1']))
            FB['lcb']['PyVRP, CAL+R'] += (not forte)
            g = 'ref' if e['ref_p_cert'] >= LCB else 'base'
            P['lcb']['PyVRP, CAL+R'][k] = (e[f'{g}_f1'], e[f'{g}_p_oos'], e[f'{g}_q_oos'], e[f'{g}_f3'])
    return P, FB


def por_inst(D, idx):
    pi = defaultdict(list)
    for (i, _), t in D.items():
        pi[i].append(t[idx])
    return {i: float(np.mean(v)) for i, v in pi.items()}


def comparar(A, B):
    res = []
    for nm, idx, esc in MEDIDAS:
        x, y = por_inst(A, idx), por_inst(B, idx)
        I = sorted(set(x) & set(y))
        a = np.array([x[i] for i in I]); c = np.array([y[i] for i in I])
        dd = 100 * (a / c - 1) if esc == 'rel' else 100 * (a - c)
        hl, lo, hi = hodges_lehmann(dd, 0.95)
        p = wilcoxon(dd).pvalue if np.any(np.abs(dd) > 1e-12) else 1.0
        res.append(dict(medida=nm, n=len(I), hl=hl, lo=lo, hi=hi, p=p))
    for r, pa in zip(res, holm([r['p'] for r in res])):
        r['p_holm'] = pa
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    out = [f'REGRA PONTUAL × REGRA LCB (LCB = {LCB:.4f}); escolha do decisor e porteira do refinamento; 1.000 cenários comuns']
    R = {}
    crit = []
    for conj in CONJ:
        nome = conj[0]
        P, FB = carregar(conj)
        R[nome] = {'metodos': {}, 'comp': {}}
        out.append(f'\n=== {nome} ===')
        out.append(f'  {"método":14s} {"execs":>5s} | cumpre pontual → LCB | fallback LCB | Δcusto LCB−pontual (mediana, média) | Δperda (p.p., média)')
        for m in P['pontual']:
            A, B = P['pontual'][m], P['lcb'].get(m, {})
            ks = sorted(set(A) & set(B))
            if not ks:
                continue
            ca = 100 * np.mean([A[k][1] >= ALPHA for k in A]); cb = 100 * np.mean([B[k][1] >= ALPHA for k in B])
            dc = np.array([100 * (B[k][0] / A[k][0] - 1) for k in ks]); dq = np.array([100 * (B[k][2] - A[k][2]) for k in ks])
            R[nome]['metodos'][m] = dict(n=len(ks), cumpre_pontual=ca, cumpre_lcb=cb, fallback=FB['lcb'][m],
                                         dcusto_mediana=float(np.median(dc)), dcusto_media=float(dc.mean()),
                                         dperda_media=float(dq.mean()), pmin_lcb=float(min(B[k][1] for k in B)))
            out.append(f'  {m:14s} {len(ks):5d} | {ca:5.1f}% → {cb:5.1f}% | {FB["lcb"][m]:3d} | {np.median(dc):+.3f}%, {dc.mean():+.3f}% '
                       f'| {dq.mean():+.3f}')
        for regra in ('pontual', 'lcb'):
            ref = P[regra]['SIM+C']
            for m in ('BUF0.95+C', 'CAL', 'ACAL', 'CAL+R', 'ACAL+R', 'PyVRP, CAL+R'):
                if m not in P[regra] or not P[regra][m]:
                    continue
                res = comparar(P[regra][m], ref)
                R[nome]['comp'][f'{regra}|{m}'] = res
                out.append(f'  [{regra}] {m} − SIM+C: ' + '; '.join(
                    f'{r["medida"]} {r["hl"]:+.2f} [{r["lo"]:+.2f}, {r["hi"]:+.2f}] p_H={r["p_holm"]:.2g}' for r in res))
        # critério da Entrada 117
        M = R[nome]['metodos']
        a_ok = all(M[m]['cumpre_lcb'] >= 95 for m in ('CAL+R', 'ACAL+R') if m in M)
        b_ok = all(M[m]['dcusto_mediana'] <= 0.1 and M[m]['dcusto_media'] <= 1.0 for m in ('SIM+C', 'CAL+R', 'ACAL+R') if m in M)
        c_ok = True
        for m in ('CAL+R', 'ACAL+R'):
            for r0, r1 in zip(R[nome]['comp'].get(f'pontual|{m}', []), R[nome]['comp'].get(f'lcb|{m}', [])):
                if r0['medida'] in ('custo (%)', 'perda (p.p.)') and r1['p_holm'] < 0.05 and np.sign(r1['hl']) != np.sign(r0['hl']) \
                        and abs(r0['hl']) > 1e-9:
                    c_ok = False
        crit.append((nome, a_ok, b_ok, c_ok))
        out.append(f'  CRITÉRIO: (a) cumprimento CAL+R/ACAL+R >= 95%: {a_ok}; (b) custo mediana <= 0,1% e média <= 1%: {b_ok}; '
                   f'(c) sem inversão significativa: {c_ok}')
    out.append('\nDECISÃO: ' + ('LCB vira a regra padrão' if all(all(c[1:]) for c in crit) else
                                'critério NÃO atendido em: ' + ', '.join(c[0] for c in crit if not all(c[1:]))))
    txt = '\n'.join(out); print(txt)
    open(os.path.join(OUT, 'relatorio_regra_lcb.txt'), 'w', encoding='utf-8').write(txt + '\n')
    json.dump(R, open(os.path.join(OUT, 'resumo_regra_lcb.json'), 'w', encoding='utf-8'), indent=1)
    tabela(R)


def tabela(R):
    """Tabela do suplemento: cumprimento (pontual / LCB), custo médio da regra LCB e ACAL+R − SIM+C sob o LCB."""
    def cel(c, m, regra='lcb', medida='custo (%)'):
        for r in R[c]['comp'].get(f'{regra}|{m}', []):
            if r['medida'] == medida:
                return (f'{r["hl"]:+.2f}' if abs(r['hl']) >= 0.005 else '0.00') + ('$^*$' if r['p_holm'] < 0.05 else '')
        return '--'
    L = [r'\begin{table}[htbp]', r'\centering',
         r'\caption{Certification against the lower confidence bound (LCB rule): the chosen plan is the cheapest with '
         r'$\hat P_{\mathrm{on}}\ge\alpha+1.645\sqrt{\alpha(1-\alpha)/500}=0.922$ (if none, the most punctual certified plan), '
         r'and the refinement gate uses the same bound. Complies: \% of chosen plans meeting $\alpha=0.90$ out of sample under the '
         r'point rule / the LCB rule. Cost: mean change in the cost of the chosen plan when moving from the point to the LCB rule '
         r'(the median change is zero in every cell). Last two columns: Hodges--Lehmann estimate of the paired difference per '
         r'instance between \mACAL{}+R and \mSIM{}, both under the LCB rule; $^*$: $p_{\mathrm{Holm}}<0.05$. Benchmark: medium and '
         r'high uncertainty.}',
         r'\label{tab:lcb}', r'\footnotesize\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{@{}l cccc rrr rr@{}}', r'\toprule',
         r' & \multicolumn{4}{c}{Complies (\%), point / LCB} & \multicolumn{3}{c}{Cost (\%)} & '
         r'\multicolumn{2}{c}{\mACAL{}+R $-$ \mSIM{}} \\',
         r'\cmidrule(lr){2-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}',
         r'Set & \mSIM{} & \mCAL{}+R & \mACAL{}+R & PyVRP & \mSIM{} & \mCAL{}+R & \mACAL{}+R & Cost (\%) & Loss (p.p.) \\',
         r'\midrule']
    ROT = {'Benchmark': 'Benchmark', 'São Paulo': r'S\~ao Paulo', 'São Paulo, time-dependent': r'S\~ao Paulo, time-dep.',
           'Athens': 'Athens', 'Scale': '200--400 customers'}
    for c, rot in ROT.items():
        M = R[c]['metodos']
        cp = [f'{M[m]["cumpre_pontual"]:.1f} / {M[m]["cumpre_lcb"]:.1f}' if m in M else '--'
              for m in ('SIM+C', 'CAL+R', 'ACAL+R', 'PyVRP, CAL+R')]
        cu = [f'{M[m]["dcusto_media"]:+.2f}' for m in ('SIM+C', 'CAL+R', 'ACAL+R')]
        L.append(f'{rot} & ' + ' & '.join(cp + cu + [cel(c, 'ACAL+R'), cel(c, 'ACAL+R', medida='perda (p.p.)')]) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    caminho = os.path.join(BASE, 'pesquisa', 'latex_ejor_c', 'tables', 'tab_regra_lcb.tex')
    open(caminho, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('tabela:', caminho)


if __name__ == '__main__':
    main()
