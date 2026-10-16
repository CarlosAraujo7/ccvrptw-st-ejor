#!/usr/bin/env python3
"""
Análise fora da amostra do Caminho C (03/10/2026, diário Entrada 74;
pesquisa/plano_caminho_C.md §3).

Etapa 1 (paralela, com cache *_oosC.json ao lado de cada *_solutions.json):
reavalia cada solução final em K cenários COMUNS por instância (tempo de
viagem lognormal + temperatura: clima, porta, termostato), com a saída do
depósito que a própria solução escolheu (partida_min):
  p_on   P(nenhum cliente com atraso > δ)        — pontualidade fora da amostra
  q_oos  f2 esperado (perda média de qualidade)   — confere que a qualidade
                                                    nominal basta (Parte B)
Etapa 2 (relatório pesquisa/resultados_c/relatorio_c.txt + CSV):
  por execução: fração da frente que cumpre α fora da amostra; custo mínimo
  entre as que cumprem (preço da confiabilidade); HV (f1, f2 nominal, f3) só
  das que cumprem, régua comum por instância; P(pontual) média.
  por instância: média das sementes; Wilcoxon pareado (n = instâncias), Holm,
  rank-biserial; contagem de instâncias em que cada algoritmo consegue
  cumprir α.
Uso: python pesquisa\\analisar_c.py --dirs results_c [results_c_b2] --workers 10
"""
import argparse, csv, glob, json, os, sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from scipy.stats import rankdata, wilcoxon

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(BASE, 'src'))
import nsga2_simheuristic as ns   # noqa: E402
import thermal_route as tr        # noqa: E402
import ccvrp_c as cc              # noqa: E402

INST_DIR = os.path.join(BASE, 'instances', 'generated')
K = 1000
SEMENTE_OOS = 20261003
_cache = {}


def _ctx(name, n_zones):
    k = (name, n_zones)
    if k not in _cache:
        inst = ns.load_instance(os.path.join(INST_DIR, name + '.json'))
        _, ttime = ns.build_matrices(inst)
        # mesma semente por instância para TODOS os algoritmos = números aleatórios comuns
        sc = tr.sortear_cenarios(inst, K, np.random.default_rng(SEMENTE_OOS + sum(map(ord, name))),
                                 n_zones=n_zones, tempo_viagem=True)
        _cache[k] = (inst, ttime, sc, ns._cat_index_map(inst))
    return _cache[k]


def _job(path):
    out = path.replace('_solutions.json', '_oosC.json')
    if os.path.exists(out):
        return 0
    m = json.load(open(path.replace('_solutions.json', '_metrics.json'), encoding='utf-8'))
    mz = bool(m.get('use_multizone')); nz = m['n_zones'] if mz else 1
    inst, ttime, sc, ci = _ctx(m['instance'], nz)
    cust = inst['customers']
    alpha = (m.get('c_params') or {}).get('alpha', cc.ALPHA)
    # Entrada 87: a cinética da avaliação é a da própria execução (gravada no _metrics.json);
    # execuções sem o campo (v1) usaram a categoria dominante
    cin = bool(m.get('cinetica_por_cliente') or False)
    rows = []
    for s in json.load(open(path, encoding='utf-8')):
        on = np.ones(K, dtype=bool); acc = np.zeros(K); tot = 0.0
        zones = ns.assign_zones(s['routes'], s['zone_block'], inst, ci) if mz else None
        for i, (r, d0) in enumerate(zip(s['routes'], s['partida_min'])):
            on &= cc.pontual_rota(r, inst, ttime, sc['tt'], d0)
            if mz:
                zi = zones[i]
                zoc = [zi['zone_of_category'][cust[c - 1]['product_category']] for c in r]
                q = tr.route_quality(r, inst, ttime, sc, zoc, zi['zone_temp'], per_customer_kinetics=True, t0_min=d0)
            else:
                q = tr.route_quality(r, inst, ttime, sc, per_customer_kinetics=cin, t0_min=d0)
            v = sum(cust[c - 1].get('load_value_BRL', 1.0) for c in r)
            acc += q * v; tot += v
        rows.append({'sol_id': s['sol_id'], 'feasible': s['feasible'], 'f1': s['f1_cost_BRL'],
                     'f2_nom': s['f2_quality_loss_insample'], 'f3': s['f3_co2_kg'],
                     'p_on': float(on.mean()), 'q_oos': float(1 - (acc / tot).mean()),
                     'p_on_insample': s.get('p_on_insample'), 'p_cert': s.get('p_cert')})
    json.dump({'meta': {k: m.get(k) for k in ('instance', 'size', 'uncertainty', 'algorithm', 'seed',
                                              'n_zones', 'use_multizone')},
               'alpha': alpha, 'K': K, 'solutions': rows}, open(out, 'w', encoding='utf-8'))
    return 1


def _rb(x, y):
    d = np.asarray(x) - np.asarray(y); d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d)); return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def _holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - k) * p[i])); adj[i] = run
    return adj


def _hv(P, ref):
    class _I:
        def __init__(self, o): self.o = tuple(o)
        def obj(self): return self.o
    if len(P) == 0:
        return 0.0
    P = np.unique(np.round(P, 9), axis=0)
    if len(P) == 1:
        P = np.vstack([P, P])
    return ns.compute_hypervolume([_I(p) for p in P], ref_point=tuple(ref))


def analisar(dirs, out_dir, log, alpha_relatorio=None):
    runs = []
    for d in dirs:
        for p in glob.glob(os.path.join(d, '**', '*_oosC.json'), recursive=True):
            runs.append(json.load(open(p, encoding='utf-8')))
    log(f"Execuções reavaliadas: {len(runs)} (K={K} cenários comuns por instância, δ={cc.DELTA_MIN} min)\n")
    por_inst = defaultdict(list)
    for r in runs:
        por_inst[r['meta']['instance']].append(r)
    linhas = []; runs_ord = []
    for inst, rr in sorted(por_inst.items()):
        pts = [np.array([[s['f1'], s['f2_nom'], s['f3']]]) for r in rr for s in r['solutions'] if s['feasible']]
        ref = np.vstack(pts).max(0) * 1.1 + 1e-9 if pts else None
        for r in rr:
            a = alpha_relatorio or r['alpha']
            ok = [s for s in r['solutions'] if s['feasible'] and s['p_on'] >= a]
            fe = [s for s in r['solutions'] if s['feasible']]
            P = np.array([[s['f1'], s['f2_nom'], s['f3']] for s in ok]) if ok else np.zeros((0, 3))
            runs_ord.append(r)
            linhas.append({'instance': inst, 'uncertainty': str(r['meta']['uncertainty']).upper(), 'passos': 1,
                           'algorithm': r['meta']['algorithm'], 'seed': r['meta']['seed'], 'alpha': a,
                           'frac_cumpre': len(ok) / max(1, len(fe)),
                           'custo_min_cumpre': min((s['f1'] for s in ok), default=np.nan),
                           'hv_cumpre': _hv(P, ref) if ref is not None else 0.0,
                           'p_on_medio': float(np.mean([s['p_on'] for s in fe])) if fe else np.nan,
                           'p_on_max': max((s['p_on'] for s in fe), default=np.nan),
                           'gap_in_out': float(np.nanmean([s['p_on_insample'] - s['p_on'] for s in fe
                                                           if s.get('p_on_insample') is not None])) if fe else np.nan,
                           'gap_cert_out': float(np.nanmean([s['p_cert'] - s['p_on'] for s in fe
                                                             if s.get('p_cert') is not None and s['p_cert'] == s['p_cert']]))
                                           if any(s.get('p_cert') is not None and s['p_cert'] == s['p_cert'] for s in fe) else np.nan,
                           'dq_oos_nom': float(np.mean([s['q_oos'] - s['f2_nom'] for s in fe])) if fe else np.nan})
    # Procedimento de folga calibrada (CAL, tese refinada — Entrada 75): para cada
    # (instância, semente), a MENOR folga da escada cuja certificação interna da
    # execução (p_cert, 500 cenários próprios — não os K da análise) aprova
    # alguma solução. Avaliado fora da amostra como qualquer outra variante;
    # 'passos' = quantas execuções da escada foram necessárias (custo computacional).
    escadas = {0.90: ['C-BUF70C', 'C-BUF80C', 'C-BUF90C', 'C-BUF95C'],
               0.95: ['C-BUF90C95', 'C-BUF95C95', 'C-BUF99C95']}
    idx = {(r['meta']['instance'], r['meta']['seed'], r['meta']['algorithm']): (L, r)
           for L, r in zip(linhas, runs_ord)}
    for alpha_e, escada in escadas.items():
        nome = 'C-CAL' if alpha_e == 0.90 else 'C-CAL95'
        for (inst, seed) in sorted({(k[0], k[1]) for k in idx if k[2] in escada}):
            for passo, alg in enumerate(escada, 1):
                if (inst, seed, alg) not in idx:
                    break
                L, r = idx[(inst, seed, alg)]
                if any((s.get('p_cert') or 0) >= alpha_e for s in r['solutions']) or passo == len(escada):
                    linhas.append({**L, 'algorithm': nome, 'passos': passo})
                    break
        # CAL-best (Entrada 76): roda a escada INTEIRA e escolhe o degrau de menor
        # custo entre as soluções aprovadas na certificação INTERNA (p_cert ≥ α) —
        # decisão sem olhar a avaliação fora da amostra. Custo: len(escada) execuções.
        nome_b = nome + '-BEST'
        for (inst, seed) in sorted({(k[0], k[1]) for k in idx if k[2] in escada}):
            if not all((inst, seed, alg) in idx for alg in escada):
                continue
            def custo_cert(alg):
                L, r = idx[(inst, seed, alg)]
                return min((s['f1'] for s in r['solutions'] if (s.get('p_cert') or 0) >= alpha_e), default=np.inf)
            melhor = min(escada, key=custo_cert)
            linhas.append({**idx[(inst, seed, melhor)][0], 'algorithm': nome_b, 'passos': len(escada)})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'por_execucao.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0])); w.writeheader(); w.writerows(linhas)
    algs = sorted({L['algorithm'] for L in linhas})
    for nome in ('C-CAL', 'C-CAL95'):
        ps = [L['passos'] for L in linhas if L['algorithm'] == nome]
        if ps:
            log(f"{nome}: execuções da escada usadas por instância×semente — média {np.mean(ps):.2f}, "
                f"distribuição {dict(sorted({p: ps.count(p) for p in ps}.items()))}")
    tab = defaultdict(dict)
    acc = defaultdict(lambda: defaultdict(list))
    for L in linhas:
        for k in ('frac_cumpre', 'custo_min_cumpre', 'hv_cumpre', 'p_on_medio', 'p_on_max', 'gap_in_out', 'dq_oos_nom'):
            acc[k][(L['instance'], L['algorithm'])].append(L[k])
    for k, dd in acc.items():
        tab[k] = {ka: float(np.nanmean(v)) if not np.all(np.isnan(v)) else np.nan for ka, v in dd.items()}
    insts = sorted({L['instance'] for L in linhas})
    log("A. Médias sobre instâncias (cada instância = média das sementes)")
    log(f"   {'algoritmo':18s} {'frac.cumpre':>11s} {'P(pont.) méd':>12s} {'P máx':>6s} {'inst. c/ alguma':>15s} "
        f"{'custo mín cumpre':>16s} {'HV cumpre':>11s} {'in−out':>7s} {'q_oos−q_nom':>11s}")
    for a in algs:
        f = [tab['frac_cumpre'].get((i, a), np.nan) for i in insts]
        cm = [tab['custo_min_cumpre'].get((i, a), np.nan) for i in insts]
        log(f"   {a:18s} {np.nanmean(f):11.1%} {np.nanmean([tab['p_on_medio'].get((i, a), np.nan) for i in insts]):12.3f} "
            f"{np.nanmean([tab['p_on_max'].get((i, a), np.nan) for i in insts]):6.3f} "
            f"{sum(1 for x in cm if np.isfinite(x)):>9d}/{len(insts):<5d} "
            f"{np.nanmean(cm):16.1f} {np.nanmean([tab['hv_cumpre'].get((i, a), np.nan) for i in insts]):11.4g} "
            f"{np.nanmean([tab['gap_in_out'].get((i, a), np.nan) for i in insts]):7.3f} "
            f"{100 * np.nanmean([tab['dq_oos_nom'].get((i, a), np.nan) for i in insts]):10.4f}pp")
    for k, nome in (('hv_cumpre', 'HV das soluções que cumprem α fora da amostra'),
                    ('frac_cumpre', 'fração da frente que cumpre α fora da amostra')):
        log(f"\nB. Wilcoxon pareado — {nome} (n = instâncias, Holm)")
        pares = [(a, b) for i, a in enumerate(algs) for b in algs[i + 1:]]
        res = []
        for a, b in pares:
            x = [tab[k].get((i, a), np.nan) for i in insts]; y = [tab[k].get((i, b), np.nan) for i in insts]
            xy = [(u, v) for u, v in zip(x, y) if np.isfinite(u) and np.isfinite(v)]
            if len(xy) < 5 or all(u == v for u, v in xy):
                res.append((a, b, len(xy), np.nan, 0.0)); continue
            u, v = zip(*xy); res.append((a, b, len(xy), wilcoxon(u, v).pvalue, _rb(u, v)))
        ps = [r[3] if np.isfinite(r[3]) else 1.0 for r in res]; adj = _holm(ps)
        for (a, b, n, p, rb), pa in zip(res, adj):
            log(f"   {a + ' vs ' + b:40s} n={n:3d} p={p:9.2e} p_Holm={pa:9.2e} r={rb:+.3f}")
    log("\nD. Por nível de incerteza (médias sobre execuções)")
    for u in sorted({L['uncertainty'] for L in linhas}):
        log(f"   {u}")
        for a in algs:
            rr = [L for L in linhas if L['uncertainty'] == u and L['algorithm'] == a]
            if not rr:
                continue
            cm = [L['custo_min_cumpre'] for L in rr if np.isfinite(L['custo_min_cumpre'])]
            g = [L['gap_in_out'] for L in rr if np.isfinite(L['gap_in_out'])]
            log(f"      {a:18s} frente que cumpre {np.mean([L['frac_cumpre'] for L in rr]):6.1%}  "
                f"P(pont.) média {np.nanmean([L['p_on_medio'] for L in rr]):.3f}  "
                f"execuções com alguma que cumpre {len(cm):3d}/{len(rr):<3d}  "
                f"custo mín (quando cumpre) {np.mean(cm) if cm else np.nan:8.1f}  "
                f"gap in−out {np.mean(g) if g else np.nan:6.3f}  "
                f"gap cert−out {np.nanmean([L['gap_cert_out'] for L in rr]) if any(np.isfinite(L['gap_cert_out']) for L in rr) else np.nan:6.3f}")
    log("\nC. Custo mínimo para cumprir α, pareado onde os dois cumprem (positivo = 1º mais caro)")
    for a in algs:
        for b in algs:
            if a >= b:
                continue
            d = [tab['custo_min_cumpre'][(i, a)] - tab['custo_min_cumpre'][(i, b)] for i in insts
                 if np.isfinite(tab['custo_min_cumpre'].get((i, a), np.nan)) and np.isfinite(tab['custo_min_cumpre'].get((i, b), np.nan))]
            if d:
                log(f"   {a + ' − ' + b:40s} instâncias {len(d):3d}  diferença média {np.mean(d):+9.1f} BRL  mediana {np.median(d):+9.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', nargs='+', required=True)
    ap.add_argument('--out', default=os.path.join(BASE, 'pesquisa', 'resultados_c'))
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--so-analise', action='store_true')
    a = ap.parse_args()
    dirs = [d if os.path.isabs(d) else os.path.join(BASE, d) for d in a.dirs]
    if not a.so_analise:
        files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, 'C-*', '*_solutions.json')))
        files.sort(key=lambda p: os.path.basename(p).split('__')[0])
        print(f"Reavaliando {len(files)} execuções (K={K})...", flush=True)
        with Pool(a.workers) as pool:
            for i, _ in enumerate(pool.imap(_job, files, chunksize=4), 1):
                if i % 100 == 0 or i == len(files):
                    print(f"  {i}/{len(files)}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    rel = open(os.path.join(a.out, 'relatorio_c.txt'), 'w', encoding='utf-8')

    def log(s=''):
        print(s, flush=True); rel.write(s + '\n')
    analisar(dirs, a.out, log)
    rel.close()


if __name__ == '__main__':
    main()
