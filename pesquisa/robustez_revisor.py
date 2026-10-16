#!/usr/bin/env python3
"""
Análises de robustez antecipando críticas de revisor (08/10/2026, diário Entrada 81).
Nada é reotimizado: usa as frentes finais já salvas (results_c/, *_solutions.json e
*_oosC.json).

1. ORÇAMENTO IGUAL. CAL usa 4 buscas. Para comparar com o mesmo orçamento, compõe
   "melhor de 4" para SIM+C e BUF0.95+C: para cada (instância, semente s), entre as
   execuções das sementes {s, s+1, s+2, s+3} (mod 10) escolhe a de menor custo entre
   soluções APROVADAS NA CERTIFICAÇÃO INTERNA (p_cert >= α) — mesma regra do CAL — e
   avalia fora da amostra (custo mínimo entre soluções com P >= α em 1.000 cenários).
2. TEMPOS DE VIAGEM CORRELACIONADOS. Reavalia a pontualidade das frentes (n=50, 18
   instâncias, 10 sementes) com multiplicadores lognormais de média 1 e o mesmo CV
   marginal, mas com fração ρ da variância logarítmica COMUM a todos os trechos do
   cenário (fator do dia): log ζ = μ + σ√ρ·Z_s + σ√(1−ρ)·E_{s,c}. ρ = 0,5.
3. TOLERÂNCIA DE ATRASO δ ∈ {0, 5, 15} min (independente), mesmas frentes.
4. DECOMPOSIÇÃO: fração da perda de qualidade nominal (f2) devida às excursões de
   porta e resfriamento = 1 − f2(sem porta) / f2(com porta), nas soluções finais.
Saída: pesquisa/resultados_robustez/relatorio_robustez.txt (+ CSV).
"""
import glob, json, os, sys
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from scipy.stats import wilcoxon

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(BASE, 'src'))
import nsga2_simheuristic as ns   # noqa: E402
import thermal_route as tr        # noqa: E402
import ccvrp_c as cc              # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equivalencia import hodges_lehmann   # noqa: E402  (IC de Hodges–Lehmann, Entrada 84)

RC = os.path.join(BASE, 'results_c')
OUT = os.path.join(BASE, 'pesquisa', 'resultados_robustez')
K = 1000; ALPHA = 0.90; RHO = 0.5
METODOS = ['C-DETC', 'C-BUF90C', 'C-BUF95C', 'C-SIMC', 'C-MOEADC']


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - k) * p[i])); adj[i] = run
    return adj


# ── 1. orçamento igual ──────────────────────────────────────────────────────
def carregar_oos(alg):
    d = {}
    for f in glob.glob(os.path.join(RC, alg, '*_oosC.json')):
        r = json.load(open(f, encoding='utf-8')); m = r['meta']
        S = [s for s in r['solutions'] if s['feasible']]
        cert = min((s['f1'] for s in S if (s.get('p_cert') or 0) >= ALPHA), default=np.inf)
        oos = min((s['f1'] for s in S if s['p_on'] >= ALPHA), default=np.nan)
        d[(m['instance'], m['seed'])] = (cert, oos)
    return d


def melhor_de_4(d):
    out = {}
    for (inst, s) in d:
        grupo = [(inst, (s + j) % 10) for j in range(4)]
        if not all(g in d for g in grupo):
            continue
        escolhido = min(grupo, key=lambda g: d[g][0])
        out[(inst, s)] = d[escolhido][1]
    return out


def cal_best():
    esc = ['C-BUF70C', 'C-BUF80C', 'C-BUF90C', 'C-BUF95C']
    ds = {a: carregar_oos(a) for a in esc}
    out = {}
    for k in ds['C-BUF95C']:
        if all(k in ds[a] for a in esc):
            a = min(esc, key=lambda a: ds[a][k][0]); out[k] = ds[a][k][1]
    return out


def por_inst(d):
    acc = defaultdict(list)
    for (inst, s), v in d.items():
        if np.isfinite(v):
            acc[inst].append(v)
    return {k: np.mean(v) for k, v in acc.items()}


# ── 2-3. reavaliação de pontualidade (correlacionado e δ) ───────────────────
_cache = {}


def _cenarios(nome):
    if nome not in _cache:
        inst = ns.load_instance(os.path.join(BASE, 'instances', 'generated', nome + '.json'))
        _, ttime = ns.build_matrices(inst)
        n = len(inst['customers']); cv = inst['thermal_profile']['sigma_t_pct']
        s2 = np.log(1 + cv * cv); mu = -s2 / 2
        rng = np.random.default_rng(777 + sum(map(ord, nome)))
        ind = np.exp(mu + np.sqrt(s2) * rng.standard_normal((K, n + 1)))
        z = rng.standard_normal((K, 1)); e = rng.standard_normal((K, n + 1))
        cor = np.exp(mu + np.sqrt(s2) * (np.sqrt(RHO) * z + np.sqrt(1 - RHO) * e))
        _cache[nome] = (inst, ttime, ind, cor)
    return _cache[nome]


def _job(path):
    m = json.load(open(path.replace('_solutions.json', '_metrics.json'), encoding='utf-8'))
    inst, ttime, ind, cor = _cenarios(m['instance'])
    res = {'instance': m['instance'], 'seed': m['seed'], 'algorithm': m['algorithm'], 'sol': []}
    for s in json.load(open(path, encoding='utf-8')):
        if not s['feasible']:
            continue
        p = {}
        for nome, tt, deltas in (('ind', ind, (0.0, 5.0, 15.0)), ('cor', cor, (5.0,))):
            for dl in deltas:
                on = np.ones(K, dtype=bool)
                for r, d0 in zip(s['routes'], s['partida_min']):
                    on &= cc.pontual_rota(r, inst, ttime, tt, d0, dl)
                p[f'{nome}_{int(dl)}'] = float(on.mean())
        res['sol'].append({'f1': s['f1_cost_BRL'], **p})
    return res


# ── 4. decomposição da perda nominal ────────────────────────────────────────
def _decomp(path):
    m = json.load(open(path.replace('_solutions.json', '_metrics.json'), encoding='utf-8'))
    inst = ns.load_instance(os.path.join(BASE, 'instances', 'generated', m['instance'] + '.json'))
    _, ttime = ns.build_matrices(inst); cust = inst['customers']
    nom = tr.cenario_nominal(inst); sem = dict(nom); sem['door'] = nom['door'] * 0.0
    out = []
    for s in json.load(open(path, encoding='utf-8')):
        if not s['feasible']:
            continue
        a = b = tot = 0.0
        for r, d0 in zip(s['routes'], s['partida_min']):
            v = sum(cust[c - 1].get('load_value_BRL', 1.0) for c in r)
            a += tr.route_quality(r, inst, ttime, nom, per_customer_kinetics=cc.CINETICA_POR_CLIENTE, t0_min=d0)[0] * v
            b += tr.route_quality(r, inst, ttime, sem, per_customer_kinetics=cc.CINETICA_POR_CLIENTE, t0_min=d0)[0] * v
            tot += v
        fa, fb = 1 - a / tot, 1 - b / tot
        out.append((fa - fb) / fa if fa > 0 else 0.0)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    rel = open(os.path.join(OUT, 'relatorio_robustez.txt'), 'w', encoding='utf-8')

    def log(t=''):
        print(t, flush=True); rel.write(t + '\n')

    # 1
    log('1. ORÇAMENTO IGUAL (4 buscas cada; escolha pela certificação interna; avaliação fora da amostra, α=0,90)')
    cal = por_inst(cal_best())
    linhas1 = []
    for nome, d in (('SIM+C, melhor de 4', melhor_de_4(carregar_oos('C-SIMC'))),
                    ('BUF0.95+C, melhor de 4', melhor_de_4(carregar_oos('C-BUF95C'))),
                    ('SIM+C, 1 busca', {k: v[1] for k, v in carregar_oos('C-SIMC').items()})):
        x = por_inst(d); ks = sorted(k for k in x if k in cal)
        dd = np.array([100 * (x[k] / cal[k] - 1) for k in ks])
        p = wilcoxon([x[k] for k in ks], [cal[k] for k in ks]).pvalue
        hl, lo, hi = hodges_lehmann(dd, 0.95)
        linhas1.append((nome, len(ks), dd, p, hl, lo, hi))
    # Holm nas três comparações do bloco (14/10/2026, Entrada 93: o texto citava p_Holm, o relatório só tinha p)
    for (nome, n, dd, p, hl, lo, hi), pa in zip(linhas1, holm([l[3] for l in linhas1])):
        log(f'   {nome:24s} vs CAL: n={n}  Δ% mediana {np.median(dd):+.2f}  média {np.mean(dd):+.2f}  p={p:.2e}  p_Holm={pa:.2e}'
            f'  HL {hl:+.2f} IC95 [{lo:+.2f}, {hi:+.2f}]')

    # 2-3
    arqs = sorted(f for a in METODOS for f in glob.glob(os.path.join(RC, a, '*_050_*_solutions.json')))
    arqs.sort(key=lambda p: os.path.basename(p).split('__')[0])
    with Pool(5) as pool:
        res = pool.map(_job, arqs, chunksize=4)
    log('\n2-3. PONTUALIDADE DAS MESMAS FRENTES (n=50, 18 instâncias × 10 sementes): % de execuções com solução que cumpre α')
    log('     e custo mínimo médio que cumpre — independente com δ = 0 / 5 / 15 min, e correlacionado (ρ=0,5, δ=5)')
    cenarios = ['ind_0', 'ind_5', 'ind_15', 'cor_5']
    for u in ('MEDIUM', 'HIGH'):
        log(f'   {u}')
        for a in METODOS:
            rr = [r for r in res if r['algorithm'] == a and r['instance'].endswith(u)]
            cel = []
            for c in cenarios:
                ok = [min((s['f1'] for s in r['sol'] if s[c] >= ALPHA), default=np.nan) for r in rr]
                cel.append(f"{c}: {100*np.mean([np.isfinite(v) for v in ok]):5.1f}% {np.nanmean(ok):8.1f}")
            log(f'      {a:10s} ' + ' | '.join(cel))
    # comparação pareada SIM+C vs BUF0.95+C sob correlação (custo mínimo que cumpre, por instância)
    for c in cenarios:
        pi = {}
        for a in ('C-SIMC', 'C-BUF95C', 'C-DETC'):
            acc = defaultdict(list)
            for r in res:
                if r['algorithm'] == a:
                    v = min((s['f1'] for s in r['sol'] if s[c] >= ALPHA), default=np.nan)
                    if np.isfinite(v):
                        acc[r['instance']].append(v)
            pi[a] = {k: np.mean(v) for k, v in acc.items()}
        for a in ('C-BUF95C', 'C-DETC'):
            ks = [k for k in pi[a] if k in pi['C-SIMC']]
            d = [100 * (pi[a][k] / pi['C-SIMC'][k] - 1) for k in ks]
            log(f'   {c}: {a} vs SIM+C  n={len(ks)}  Δ% mediana {np.median(d):+.2f}  média {np.mean(d):+.2f}')

    # 4
    arqs4 = sorted(glob.glob(os.path.join(RC, 'C-SIMC', '*_solutions.json')) + glob.glob(os.path.join(RC, 'C-DET', '*_solutions.json')))
    with Pool(5) as pool:
        partes = pool.map(_decomp, arqs4, chunksize=8)
    v = np.concatenate([np.array(p) for p in partes if p])
    log(f'\n4. DECOMPOSIÇÃO: fração da perda nominal devida a porta + resfriamento, sobre {len(v)} soluções finais (SIM+C e DET)')
    log(f'   média {100*v.mean():.2f}%  mediana {100*np.median(v):.2f}%  p95 {100*np.percentile(v,95):.2f}%  máx {100*v.max():.2f}%')
    rel.close()


if __name__ == '__main__':
    main()
