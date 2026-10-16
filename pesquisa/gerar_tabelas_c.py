#!/usr/bin/env python3
"""
Gera as tabelas LaTeX do manuscrito do Caminho C (05/10/2026, diário Entrada 80)
diretamente dos resultados — nenhum número é digitado à mão.

Fontes:
  results_c/*/  *_oosC.json, *_metrics.json   (F4 fases 1-2, reavaliação K=1000)
  results_c_b2/*/ *_oosC.json                  (F4 fase 3)
  pesquisa/resultados_c/por_execucao.csv       (analisar_c.py, inclui CAL/CAL-BEST)
  pesquisa/resultados_parte_b/parte_b.csv      (parte_b_caracterizacao.py)
  pesquisa/log_test_split_ls.txt               (validação contra o ótimo exato)
Saída: pesquisa/latex_ejor_c/tables/*.tex e pesquisa/latex_ejor_c/tables/numeros.json
(números citados no texto, para conferência).
"""
import csv, glob, json, os, re
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT = os.path.join(BASE, 'pesquisa', 'latex_ejor_c', 'tables')
os.makedirs(OUT, exist_ok=True)
NUM = {}
NOME = {'C-DET': r'\mDET', 'C-DETC': r'\mDETC', 'C-BUF90': r'\mBUFnc{0.90}', 'C-BUF70C': r'\mBUF{0.70}',
        'C-BUF80C': r'\mBUF{0.80}', 'C-BUF90C': r'\mBUF{0.90}', 'C-BUF95C': r'\mBUF{0.95}',
        'C-CAL': r'\mCALmin', 'C-CAL-BEST': r'\mCAL', 'C-SIMC': r'\mSIM', 'C-MOEADC': r'\mMOEAD',
        'C-SIMC95': r'\mSIM', 'C-BUF90C95': r'\mBUF{0.90}', 'C-BUF95C95': r'\mBUF{0.95}',
        'C-BUF99C95': r'\mBUF{0.99}', 'C-CAL95': r'\mCALmin', 'C-CAL95-BEST': r'\mCAL'}
NIV = ['LOW', 'MEDIUM', 'HIGH']; GR = ['C', 'R', 'RC']


def grupo(inst):
    f = inst.split('_')[1]
    return 'RC' if f.startswith('RC') else f[0]


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - k) * p[i])); adj[i] = run
    return adj


def rb(x, y):
    from scipy.stats import rankdata
    d = np.asarray(x) - np.asarray(y); d = d[d != 0]
    if not len(d):
        return 0.0
    r = rankdata(np.abs(d)); return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def fp(p):
    if p < 1e-3:
        e = int(np.floor(np.log10(p))); m = p / 10 ** e
        return rf'${m:.1f}\times10^{{{e}}}$'
    return f'{p:.3f}'


def escrever(nome, txt):
    open(os.path.join(OUT, nome), 'w', encoding='utf-8').write(txt)


R = list(csv.DictReader(open(os.path.join(BASE, 'pesquisa', 'resultados_c', 'por_execucao.csv'), encoding='utf-8')))
for r in R:
    r['cm'] = float(r['custo_min_cumpre']) if r['custo_min_cumpre'] not in ('', 'nan') else np.nan
    r['hv'] = float(r['hv_cumpre']); r['fr'] = float(r['frac_cumpre']); r['g'] = grupo(r['instance'])


def por_inst(alg, campo):
    d = defaultdict(list)
    for r in R:
        if r['algorithm'] == alg and np.isfinite(r[campo]):
            d[r['instance']].append(r[campo])
    return {k: float(np.mean(v)) for k, v in d.items()}


# ── Tabela: conformidade (α = 0,90) ─────────────────────────────────────────
algs90 = ['C-DET', 'C-DETC', 'C-BUF90', 'C-BUF70C', 'C-BUF80C', 'C-BUF90C', 'C-BUF95C', 'C-CAL-BEST', 'C-SIMC', 'C-MOEADC']
lin = []
for a in algs90:
    cel = []
    for u in ('MEDIUM', 'HIGH'):
        for g in GR:
            rr = [r for r in R if r['algorithm'] == a and r['uncertainty'] == u and r['g'] == g]
            ok = 100 * np.mean([np.isfinite(r['cm']) for r in rr]); fr = 100 * np.mean([r['fr'] for r in rr])
            cel.append(f'{ok:.0f} ({fr:.0f})')
    rr = [r for r in R if r['algorithm'] == a and r['uncertainty'] == 'LOW']
    cel.insert(0, f"{100*np.mean([np.isfinite(r['cm']) for r in rr]):.0f} ({100*np.mean([r['fr'] for r in rr]):.0f})")
    lin.append(NOME[a] + ' & ' + ' & '.join(cel) + r' \\')
    NUM[f'conf_{a}'] = cel
escrever('tab_conformidade.tex', r"""\begin{table}[htbp]
\centering
\caption{Out-of-sample compliance with the service level $\alpha=0.90$: percentage of runs whose final front contains at least one solution with $\hat P(\text{on time})\ge\alpha$ over 1{,}000 common scenarios; in parentheses, the percentage of front solutions that comply. 54 instances $\times$ 10 seeds; LOW pooled over families.}
\label{tab:compliance}
\small
\begin{tabular}{@{}l r rrr rrr@{}}
\toprule
 & LOW & \multicolumn{3}{c}{MEDIUM} & \multicolumn{3}{c}{HIGH} \\
\cmidrule(lr){3-5}\cmidrule(l){6-8}
Method & all & C & R & RC & C & R & RC \\
\midrule
""" + '\n'.join(lin) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ── Tabela: custo para cumprir α, pareado contra SIM+C ───────────────────────
blocos = []
for alpha, ref, algs in ((0.90, 'C-SIMC', ['C-DETC', 'C-BUF90', 'C-BUF70C', 'C-BUF80C', 'C-BUF90C', 'C-BUF95C', 'C-CAL', 'C-CAL-BEST', 'C-MOEADC']),
                         (0.95, 'C-SIMC95', ['C-BUF90C95', 'C-BUF95C95', 'C-BUF99C95', 'C-CAL95-BEST'])):
    base = por_inst(ref, 'cm'); res = []
    for a in algs:
        x = por_inst(a, 'cm'); ks = sorted(k for k in x if k in base)
        d = np.array([100 * (x[k] / base[k] - 1) for k in ks])
        p = wilcoxon([x[k] for k in ks], [base[k] for k in ks]).pvalue if np.any(d != 0) else 1.0
        res.append([a, len(ks), np.median(d), np.mean(d), np.max(d), int(np.sum(d > 5)), p])
    adj = holm([r[-1] for r in res])
    linhas = []
    for (a, n, med, mea, mx, n5, p), pa in zip(res, adj):
        nome = NOME[a] + (r'$^\dagger$' if a in ('C-CAL', 'C-CAL95') else '')
        linhas.append(f"{nome} & {n} & {med:+.2f} & {mea:+.2f} & {mx:+.1f} & {n5} & {fp(pa)} \\\\")
        NUM[f'custo_{alpha}_{a}'] = dict(n=n, med=round(med, 3), media=round(mea, 3), max=round(mx, 2), n_acima5=n5, p_holm=pa)
    blocos.append((alpha, ref, linhas))
txt = r"""\begin{table}[htbp]
\centering
\caption{Cost of the cheapest out-of-sample-compliant solution relative to \mSIM{} (\%; per instance, mean over the seeds that comply; paired over instances where both methods comply). Wilcoxon signed-rank test with Holm correction within each block. $^\dagger$\mCALmin{}: smallest compliant rung of the buffer ladder; \mCAL{}: cheapest rung certified compliant.}
\label{tab:cost}
\small
\begin{tabular}{@{}l r r r r r r@{}}
\toprule
Method & $n$ & median & mean & max & \#$>$5\% & $p_{\mathrm{Holm}}$ \\
\midrule
"""
for alpha, ref, linhas in blocos:
    txt += rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{$\alpha={alpha:.2f}$, reference \mSIM{{}}}}}} \\" + '\n' + '\n'.join(linhas) + '\n'
    if alpha == 0.90:
        txt += r'\midrule' + '\n'
txt += r"""\bottomrule
\end{tabular}
\end{table}
"""
escrever('tab_custo.tex', txt)

# ── Tabela: HV das soluções que cumprem (pareado contra SIM+C) ──────────────
base = por_inst('C-SIMC', 'hv'); res = []
for a in ['C-DET', 'C-DETC', 'C-BUF90', 'C-BUF90C', 'C-BUF95C', 'C-CAL-BEST', 'C-MOEADC']:
    x = por_inst(a, 'hv'); ks = sorted(k for k in x if k in base)
    xv = [x[k] for k in ks]; bv = [base[k] for k in ks]
    res.append([a, len(ks), np.median([100 * (u / v - 1) if v > 0 else 0 for u, v in zip(xv, bv)]), rb(xv, bv), wilcoxon(xv, bv).pvalue])
adj = holm([r[-1] for r in res])
linhas = [f"{NOME[a]} & {n} & {med:+.1f} & {r:+.2f} & {fp(pa)} \\\\" for (a, n, med, r, p), pa in zip(res, adj)]
for (a, n, med, r, p), pa in zip(res, adj):
    NUM[f'hv_{a}'] = dict(n=n, med=round(med, 2), r=round(r, 3), p_holm=pa)
escrever('tab_hv.tex', r"""\begin{table}[htbp]
\centering
\caption{Hypervolume of the out-of-sample-compliant solutions ($f_1,f_2,f_3$; common reference point per instance) relative to \mSIM{} ($\alpha=0.90$): median relative difference (\%), rank-biserial correlation $r$ (negative favors \mSIM{}) and Wilcoxon $p$ with Holm correction; $n=54$ instances.}
\label{tab:hv}
\small
\begin{tabular}{@{}l r r r r@{}}
\toprule
Method & $n$ & median $\Delta$HV (\%) & $r$ & $p_{\mathrm{Holm}}$ \\
\midrule
""" + '\n'.join(linhas) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ── Preço da confiabilidade e B2 (a partir dos _oosC.json) ──────────────────
def melhores(pattern, alpha=None, sementes=None):
    out = defaultdict(list)
    for f in glob.glob(pattern):
        d = json.load(open(f, encoding='utf-8')); m = d['meta']
        if sementes is not None and m['seed'] not in sementes:
            continue
        S = [s for s in d['solutions'] if s['feasible'] and (alpha is None or s['p_on'] >= alpha)]
        allS = [s for s in d['solutions'] if s['feasible']]
        if S:
            s = min(S, key=lambda x: x['f1'])
            out[m['instance']].append((s['f1'], s['q_oos'], s['f3'], s['p_on'], np.mean([z['p_on'] for z in allS])))
    return {k: np.mean(v, axis=0) for k, v in out.items()}


det = melhores(os.path.join(BASE, 'results_c', 'C-DET', '*_oosC.json'))
linhas = []
for rot, alg, alpha in (('\\mSIM', 'C-SIMC', 0.90), ('\\mBUF{0.95}', 'C-BUF95C', 0.90), ('\\mSIM', 'C-SIMC95', 0.95)):
    m = melhores(os.path.join(BASE, 'results_c', alg, '*_oosC.json'), alpha)
    for u in NIV:
        ks = [k for k in m if k.endswith(u) and k in det]
        dc = np.median([100 * (m[k][0] / det[k][0] - 1) for k in ks]); dq = np.median([100 * (m[k][1] - det[k][1]) for k in ks])
        de = np.median([100 * (m[k][2] / det[k][2] - 1) for k in ks])
        p0 = np.mean([det[k][3] for k in ks]); p1 = np.mean([m[k][3] for k in ks])
        linhas.append(f"{alpha:.2f} & {rot}{{}} & {u.capitalize()} & {p0:.2f} & {p1:.2f} & {dc:+.2f} & {dq:+.2f} & {de:+.2f} \\\\")
        NUM[f'preco_{alg}_{u}'] = dict(p_det=round(p0, 3), p=round(p1, 3), custo=round(dc, 3), qual_pp=round(dq, 3), co2=round(de, 3))
escrever('tab_preco.tex', r"""\begin{table}[htbp]
\centering
\caption{Price of reliability: cheapest out-of-sample-compliant solution versus the cheapest solution of \mDET{} (which ignores punctuality), medians over the 18 instances of each uncertainty level (quality loss evaluated out of sample). $\hat P$: out-of-sample probability that no customer is served more than $\delta=5$\,min late.}
\label{tab:price}
\small
\begin{tabular}{@{}c l l r r r r r@{}}
\toprule
$\alpha$ & Method & Level & $\hat P$ \mDET{} & $\hat P$ method & $\Delta f_1$ (\%) & $\Delta$ quality loss (p.p.) & $\Delta f_3$ (\%) \\
\midrule
""" + '\n'.join(linhas) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

linhas = []
# Entrada 87 (caminho 1): a base Z = 1 do B2 é o compartimento único com carga MISTA
# (results_c_b2/<alg>-MIX/), com a mesma cinética por produto das zonas; e a conformidade
# legal vem de conformidade_legal_b2.py (fração do valor acima do limite da categoria).
_cl = os.path.join(BASE, 'pesquisa', 'resultados_c_b2', 'conformidade_legal.json')
LEGAL = json.load(open(_cl, encoding='utf-8')) if os.path.exists(_cl) else {}
for alg in ('C-DET', 'C-BUF90C', 'C-SIMC'):
    z1 = defaultdict(dict)
    for f in glob.glob(os.path.join(BASE, 'results_c_b2', alg + '-MIX', '*_oosC.json')):
        d = json.load(open(f, encoding='utf-8')); m = d['meta']
        if m['seed'] <= 4:
            z1[(m['instance'], m['seed'])] = d
    for z in (2, 3):
        dq, dc, de, ok1, okz = [], [], [], [], []
        for f in glob.glob(os.path.join(BASE, 'results_c_b2', f'{alg}_Z{z}', '*_oosC.json')):
            d = json.load(open(f, encoding='utf-8')); m = d['meta']; k = (m['instance'], m['seed'])
            if k not in z1:
                continue
            A = [s for s in z1[k]['solutions'] if s['feasible'] and s['p_on'] >= 0.9]
            B = [s for s in d['solutions'] if s['feasible'] and s['p_on'] >= 0.9]
            ok1.append(bool(A)); okz.append(bool(B))
            if A and B:
                dq.append(100 * (min(s['q_oos'] for s in B) - min(s['q_oos'] for s in A)))
                dc.append(100 * (min(s['f1'] for s in B) / min(s['f1'] for s in A) - 1))
                de.append(100 * (min(s['f3'] for s in B) / min(s['f3'] for s in A) - 1))
        l1 = LEGAL.get(alg + '-MIX', {}).get('media', np.nan); lz = LEGAL.get(f'{alg}+B2z{z}', {}).get('media', np.nan)
        linhas.append(f"{NOME[alg]} & {z} & {100*np.mean(ok1):.0f} & {100*np.mean(okz):.0f} & {np.median(dq):+.2f} & {np.median(dc):+.2f} & {np.median(de):+.2f} & {l1:.1f} & {lz:.1f} \\\\")
        NUM[f'b2_{alg}_Z{z}'] = dict(ok1=np.mean(ok1), okz=np.mean(okz), dq=np.median(dq), dc=np.median(dc), de=np.median(de), n=len(dq),
                                     legal_z1=l1, legal_z=lz)
escrever('tab_b2.tex', r"""\begin{table}[htbp]
\centering
\caption{Multi-compartment vehicles ($Z\in\{2,3\}$ thermal zones) versus a single compartment carrying the same mixed load at the dominant set-point ($Z=1$), same instances and seeds (54 instances $\times$ 5 seeds); every product follows its own category's kinetics in both cases. Compliance: \% of runs with an out-of-sample-compliant solution ($\alpha=0.90$). Differences (medians over runs where both comply) refer to the best value of each objective among compliant solutions; quality loss evaluated out of sample. Legal: mean share of delivered value whose product temperature exceeded the legal limit of its own category \citep[Art.~30]{cvs:2026} ($\tau_p=10$\,h, nominal scenario).}
\label{tab:b2}
\small
\begin{tabular}{@{}l c r r r r r r r@{}}
\toprule
 & & \multicolumn{2}{c}{Compliance (\%)} & & & & \multicolumn{2}{c}{Legal (\% value)} \\
\cmidrule(lr){3-4}\cmidrule(l){8-9}
Method & $Z$ & $Z{=}1$ & $Z$ & $\Delta$ quality loss (p.p.) & $\Delta f_1$ (\%) & $\Delta f_3$ (\%) & $Z{=}1$ & $Z$ \\
\midrule
""" + '\n'.join(linhas) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ── Parte B ─────────────────────────────────────────────────────────────────
PB = list(csv.DictReader(open(os.path.join(BASE, 'pesquisa', 'resultados_parte_b', 'parte_b.csv'), encoding='utf-8')))
ROT = {'base': 'Base case', 'inicio=19h': 'Operations from 19:00',
       'porta=0.4s/kg': 'Unloading rate 0.4\\,s/kg', 'porta=1.2s/kg': 'Unloading rate 1.2\\,s/kg',
       'cv_porta=0.05': 'Door-time CV 0.05', 'cv_porta=0.6': 'Door-time CV 0.60',
       'tau_p=0.25h': 'Product time constant 0.25\\,h', 'tau_p=1.0h': 'Product time constant 1\\,h',
       'tau_p=4.0h': 'Product time constant 4\\,h', 'sigmaT×0.0': 'Thermostat noise $\\times0$',
       'sigmaT×2.0': 'Thermostat noise $\\times2$', 'cv_t=0.05': 'Travel-time CV 0.05', 'cv_t=0.3': 'Travel-time CV 0.30',
       'sigmaT~amb': 'Thermostat noise $\\propto$ ambient (decision-dependent)'}
conds = []
for r in PB:
    if r['condicao'] not in conds:
        conds.append(r['condicao'])
linhas = []
clim = [c for c in conds if c.startswith('clima=') and '+19h' not in c]
clim19 = [c for c in conds if c.startswith('clima=') and '+19h' in c]


def agg(cs):
    rr = [r for r in PB if r['condicao'] in cs]
    f = lambda k: np.array([float(r[k]) for r in rr])  # noqa: E731
    return (100 * f('q_esp').mean(), np.nanmin(f('rho_q')), np.nanmedian(f('rho_q')), 100 * np.nanmax(f('arrep_q_abs')),
            100 * f('quebra_esp').mean(), 100 * np.nanmax(f('arrep_quebra_abs')), f('p_pontual').mean())


grupos_pb = [('base', ['base'])] + [('6 other capitals', clim)] + [('inicio=19h', ['inicio=19h']),
             ('3 capitals, from 19:00', clim19)] + [(c, [c]) for c in conds if c.startswith(('porta', 'cv_porta', 'tau_p', 'sigma', 'cv_t'))]
for rot, cs in grupos_pb:
    q, rmin, rmed, aq, qb, aqb, p = agg(cs)
    nome = ROT.get(rot, rot)
    linhas.append(f"{nome} & {q:.2f} & {rmin:.4f} & {aq:.4f} & {qb:.2f} & {p:.3f} \\\\")
    NUM[f'parteb_{rot}'] = dict(q=q, rho_med=rmed, rho_min=rmin, arrep_q=aq, quebra=qb, arrep_quebra=aqb, p=p)
escrever('tab_parteb.tex', r"""\begin{table}[htbp]
\centering
\caption{Where uncertainty matters: re-evaluation of the final fronts of \mDET{}, \mBUF{0.90} and \mSIM{} (18 instances with $n=50$; up to 150 distinct solutions each) under alternative conditions, one factor at a time around the base case (S\~ao Paulo, operations from 06:00). $E[\text{loss}]$: mean expected quality loss; $\rho_{\min}$: smallest Spearman correlation, over instances, between nominal and expected quality loss; Regret: largest loss, over instances, of choosing by the nominal instead of the expected value within cost budgets (independent scenario samples); Breach: expected share of delivered value whose product temperature exceeded the legal limit; $\hat P$: mean probability that no customer is served more than 5\,min late.}
\label{tab:partB}
\small
\begin{tabular}{@{}p{0.34\textwidth} r r r r r@{}}
\toprule
Condition & $E[\text{loss}]$ (\%) & $\rho_{\min}$ & Regret (p.p.) & Breach (\%) & $\hat P$ \\
\midrule
""" + '\n'.join(linhas) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ── Validação contra o ótimo exato ──────────────────────────────────────────
log = open(os.path.join(BASE, 'pesquisa', 'log_test_split_ls.txt'), encoding='utf-8').read()
linhas = []
for m in re.finditer(r"n=\s*(\d+) ótimo ([\d.]+) \(([^)]*)\) \| firstfit: melhor ([\d.]+) \(([+\-\d.]+)%\), média ([+\-\d.]+)% \| split: melhor ([\d.]+) \(([+\-\d.]+)%\), média ([+\-\d.]+)% \| split\+BL: melhor ([\d.]+) \(([+\-\d.]+)%\), média ([+\-\d.]+)%", log):
    n, ot, st = m.group(1), m.group(2), 'proven' if 'provado' in m.group(3) and 'não' not in m.group(3) else 'best known'
    linhas.append(f"{n} & {float(ot):,.2f} ({st}) & {m.group(5)} / {m.group(6)} & {m.group(8)} / {m.group(9)} & {m.group(11)} / {m.group(12)} \\\\")
    NUM[f'gap_n{n}'] = m.groups()
escrever('tab_exato.tex', r"""\begin{table}[htbp]
\centering
\caption{Cost gap (\%) of the cheapest solution found by the deterministic variant with respect to the exact MILP optimum, best / mean over 5 seeds. First-fit: original parallel decoder; Split: optimal split of the giant tour; Split+LS: Split with granular local search (the configuration used in all experiments).}
\label{tab:exact}
\small
\begin{tabular}{@{}r r r r r@{}}
\toprule
$n$ & Optimum (BRL) & First-fit & Split & Split+LS \\
\midrule
""" + '\n'.join(linhas) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ── Tempo por execução ──────────────────────────────────────────────────────
t = defaultdict(list)
for f in glob.glob(os.path.join(BASE, 'results_c', 'C-*', '*_metrics.json')):
    m = json.load(open(f, encoding='utf-8')); t[(m['algorithm'], m['size'])].append(m['elapsed_s'])
linhas = []
for a in ['C-DET', 'C-DETC', 'C-BUF90C', 'C-BUF95C', 'C-SIMC', 'C-MOEADC']:
    linhas.append(NOME[a] + ' & ' + ' & '.join(f"{np.mean(t[(a, n)]):.0f}" for n in (25, 50, 100)) + r' \\')
    NUM[f'tempo_{a}'] = [float(np.mean(t[(a, n)])) for n in (25, 50, 100)]
escrever('tab_tempo.tex', r"""\begin{table}[htbp]
\centering
\caption{Mean wall-clock time per run (s; 200 generations, 10 runs in parallel on a 12-core desktop). \mCAL{} requires one run per rung of the ladder (four).}
\label{tab:time}
\small
\begin{tabular}{@{}l r r r@{}}
\toprule
Method & $n=25$ & $n=50$ & $n=100$ \\
\midrule
""" + '\n'.join(linhas) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

json.dump(NUM, open(os.path.join(OUT, 'numeros.json'), 'w', encoding='utf-8'), indent=1, default=float, ensure_ascii=False)
print('tabelas:', sorted(os.listdir(OUT)))

# ── Apêndice: dados climáticos (pesquisa/clima/resumo_clima.txt) ────────────
MESES = {1: 'Jan', 2: 'Feb', 3: 'Mar', 9: 'Sep', 11: 'Nov'}
lin = []
for l in open(os.path.join(BASE, 'pesquisa', 'clima', 'resumo_clima.txt'), encoding='utf-8'):
    m = re.match(r"(.+?\))\s+mês\s+(\d+)\s+dias\s+(\d+)\s+Tmax\s+([\d.]+)\s+amplitude\s+([\d.]+)\s+sd\(média diária\)\s+([\d.]+)\s+pico às\s+(\d+)h\s+RMSE vs\. ASHRAE T6\s+([\d.]+)", l)
    if m:
        cid = m.group(1).replace('São', r'S\~ao').replace('Galeão', r'Gale\~ao').replace('Congonhas', 'Congonhas')
        lin.append(f"{cid} & {MESES.get(int(m.group(2)), m.group(2))} & {m.group(3)} & {m.group(4)} & {m.group(5)} & {m.group(6)} & {m.group(7)} & {m.group(8)} \\\\")
# tabela inteira num arquivo: \input dentro de tabular seguido de \bottomrule quebra o LaTeX
escrever('tab_clima.tex', r"""\begin{table}[htbp]
\centering
\caption{Ambient temperature data of the seven capitals (hottest month, 2011--2020).}
\label{tab:climate}
\small
\begin{tabular}{@{}l r r r r r r r@{}}
\toprule
City (station) & Month & Days & $T_{\max}$ (\textdegree C) & Range (\textdegree C) & $\sigma_{\mathrm{day}}$ (\textdegree C) & Peak (h) & RMSE \\
\midrule
""" + '\n'.join(lin) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
print('clima:', len(lin), 'linhas')
