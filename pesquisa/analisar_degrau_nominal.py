#!/usr/bin/env python3
"""
Análise POST HOC do degrau nominal na escada do CAL (17/10/2026, diário Entrada 107).

Ideia surgida depois de ver o caso de Atenas (Entrada 106), por isso relatada como post hoc: incluir o nível
nominal, isto é, o plano determinístico certificado (DET+C, multiplicador 1), como degrau 0 da escada de
buffers. CAL0 = a escolha certificada mais barata entre DET+C, BUF0.70+C, BUF0.80+C, BUF0.90+C e BUF0.95+C
da mesma instância e semente; CAL = a mesma regra sem o DET+C. Nenhuma execução nova: usa os
*_oosC.json (1.000 cenários comuns) dos quatro conjuntos.
Medidas do plano escolhido: cumpre α fora da amostra (% das execuções), custo, perda de qualidade esperada e
CO2; pareado por instância (média das sementes) contra SIM+C e contra CAL, Hodges-Lehmann com IC 95% e
Wilcoxon com Holm nas medidas de cada conjunto.
Uso: python pesquisa/analisar_degrau_nominal.py   Saída: pesquisa/resultados_degrau_nominal/
"""
import glob, json, os
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT = os.path.join(BASE, 'pesquisa', 'resultados_degrau_nominal')
ALPHA = 0.90
ESCADA = ['C-BUF70C', 'C-BUF80C', 'C-BUF90C', 'C-BUF95C']
CONJ = [('benchmark', 'results_c'), ('SP', 'results_c_sp'), ('SP com horário', 'results_c_sp_td'),
        ('Atenas', 'results_c_atenas')]


def hl(d):
    """Estimador de Hodges-Lehmann (mediana das médias de Walsh) e IC 95% aproximado por bootstrap."""
    d = np.asarray(d, dtype=float)
    w = [(d[i] + d[j]) / 2 for i in range(len(d)) for j in range(i, len(d))]
    rng = np.random.default_rng(7)
    bs = []
    for _ in range(2000):
        x = rng.choice(d, len(d))
        bs.append(np.median([(x[i] + x[j]) / 2 for i in range(len(x)) for j in range(i, len(x))]))
    return float(np.median(w)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * ps[i])); adj[i] = run
    return adj


def main():
    os.makedirs(OUT, exist_ok=True)
    rel = open(os.path.join(OUT, 'relatorio_degrau_nominal.txt'), 'w', encoding='utf-8')

    def log(s=''):
        print(s, flush=True); rel.write(s + '\n')
    log('ANÁLISE POST HOC — degrau nominal (DET+C) na escada do CAL; α = 0,90; 1.000 cenários comuns')
    resumo = {}
    for nome, res in CONJ:
        esc = defaultdict(dict)
        for a in ['C-DETC', 'C-SIMC'] + ESCADA:
            for f in glob.glob(os.path.join(BASE, res, a, '*_oosC.json')):
                d = json.load(open(f, encoding='utf-8')); m = d['meta']
                cert = [s for s in d['solutions'] if s['feasible'] and (s.get('p_cert') or 0) >= ALPHA]
                if cert:
                    esc[(m['instance'], m['seed'])][a] = min(cert, key=lambda s: s['f1'])
        linhas = []
        for (inst, sd), d in esc.items():
            lad = [(d[a], a) for a in ESCADA if a in d]
            if 'C-SIMC' not in d or not lad:
                continue
            cal, _ = min(lad, key=lambda x: x[0]['f1'])
            cal0, deg0 = min(lad + ([(d['C-DETC'], 'C-DETC')] if 'C-DETC' in d else []), key=lambda x: x[0]['f1'])
            linhas.append(dict(inst=inst, seed=sd, sim=d['C-SIMC'], cal=cal, cal0=cal0, deg0=deg0))
        if not linhas:
            log(f'\n{nome}: sem dados'); continue
        n = len(linhas)
        cum = {k: 100 * np.mean([x[k]['p_on'] >= ALPHA for x in linhas]) for k in ('sim', 'cal', 'cal0')}
        nom = 100 * np.mean([x['deg0'] == 'C-DETC' for x in linhas])
        log(f'\n=== {nome} ({res}; {n} execuções, {len({x["inst"] for x in linhas})} instâncias) ===')
        log(f'  escolha cumpre α: SIM+C {cum["sim"]:.1f}%   CAL {cum["cal"]:.1f}%   CAL0 {cum["cal0"]:.1f}%   '
            f'(CAL0 escolhe o degrau nominal em {nom:.0f}% das execuções)')
        por = defaultdict(list)
        for x in linhas:
            por[x['inst']].append(x)
        testes = []
        for ref in ('sim', 'cal'):
            for k, rot in (('f1', 'custo %'), ('q_oos', 'perda p.p.'), ('f3', 'CO2 %'), ('p_on', 'P_on p.p.')):
                dif = []
                for inst, L in por.items():
                    va = np.mean([x['cal0'][k] for x in L]); vb = np.mean([x[ref][k] for x in L])
                    dif.append(100 * (va - vb) if k in ('q_oos', 'p_on') else 100 * (va / vb - 1))
                p = wilcoxon(dif).pvalue if any(abs(v) > 1e-12 for v in dif) else 1.0
                testes.append((ref, rot, dif, p))
        adj = holm(np.array([t[3] for t in testes]))
        res_g = {}
        for (ref, rot, dif, p), pa in zip(testes, adj):
            e, lo, hi = hl(dif)
            log(f'  CAL0 − {"SIM+C" if ref == "sim" else "CAL":5s} {rot:11s} n={len(dif):2d}  HL {e:+7.3f} [{lo:+7.3f}, {hi:+7.3f}]  '
                f'p={p:.3g}  p_Holm={pa:.3g}')
            res_g[f'{ref}_{rot}'] = dict(hl=e, lo=lo, hi=hi, p=float(p), p_holm=float(pa), n=len(dif))
        resumo[nome] = dict(n=n, cumpre=cum, nominal=nom, testes=res_g)
    json.dump(resumo, open(os.path.join(OUT, 'resumo_degrau_nominal.json'), 'w', encoding='utf-8'), indent=1,
              ensure_ascii=False)
    rel.close()
    tabela(resumo)


def tabela(resumo):
    """Tabela do suplemento (tables/tab_degrau_nominal.tex)."""
    rot = {'benchmark': 'Benchmark', 'SP': r'S\~ao Paulo', 'SP com horário': r'S\~ao Paulo, time-dependent',
           'Atenas': 'Athens'}

    def c(t):
        return f"{t['hl'] + 0.0:+.2f} ({t['p_holm']:.2g})"
    L = [r'\begin{table}[htbp]', r'\centering',
         r'\caption{Post hoc analysis of a nominal rung: \mCAL{}$_0$ adds \mDETC{} (the nominal level) as rung zero of '
         r'the ladder of \mCAL{} and returns the cheapest certified choice among the five rungs (existing runs; '
         r'1{,}000 common scenarios; $\alpha=0.90$). Nominal: \% of runs in which \mCAL{}$_0$ picks the nominal rung. '
         r'Comply: \% of runs whose chosen plan meets $\alpha$. $\Delta$cost: Hodges--Lehmann estimate of the paired '
         r'difference per instance (\%), with $p_{\mathrm{Holm}}$ over the eight tests of each set.}',
         r'\label{tab:nominal}', r'\small', r'\begin{tabular}{@{}l r rrr rr@{}}', r'\toprule',
         r' & & \multicolumn{3}{c}{Comply (\%)} & \multicolumn{2}{c}{$\Delta$cost of \mCAL{}$_0$ (\%)} \\',
         r'\cmidrule(lr){3-5}\cmidrule(l){6-7}',
         r'Set & Nominal (\%) & \mSIM{} & \mCAL{} & \mCAL{}$_0$ & vs \mSIM{} & vs \mCAL{} \\', r'\midrule']
    for k in ('benchmark', 'SP', 'SP com horário', 'Atenas'):
        if k not in resumo:
            continue
        r = resumo[k]; cu = r['cumpre']; t = r['testes']
        L.append(f"{rot[k]} & {r['nominal']:.0f} & {cu['sim']:.1f} & {cu['cal']:.1f} & {cu['cal0']:.1f} & "
                 f"{c(t['sim_custo %'])} & {c(t['cal_custo %'])} " + r'\\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    arq = os.path.join(BASE, 'pesquisa', 'latex_ejor_c', 'tables', 'tab_degrau_nominal.tex')
    open(arq, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('tabela:', arq)


if __name__ == '__main__':
    main()
