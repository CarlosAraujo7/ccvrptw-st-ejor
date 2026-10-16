#!/usr/bin/env python3
"""
Correlação DENTRO da busca (08/10/2026, diário Entrada 84).

Crítica de revisor: a verificação com tempos correlacionados (Seção 6.6) só reavalia
frentes otimizadas com tempos independentes. A simulação nunca teve a chance de usar
a correlação, justamente o caso em que ela poderia levar vantagem sobre o buffer
(o buffer por quantil marginal não "vê" a correlação; a simulação vê).

Experimento (mundo correlacionado, ρ = 0,5, fator comum do dia, mesmo CV marginal):
  C-SIMC-COR   SIM+C com amostra de busca (30) e certificação (500) CORRELACIONADAS
  C-BUF95C-COR BUF0.95+C (quantil marginal) com certificação (500) CORRELACIONADA
  — comparados entre si e com C-SIMC e C-BUF95C do experimento principal (planejados e
    certificados com tempos independentes), sementes 0–4.
Instâncias: as 12 com n = 50 e incerteza MEDIUM ou HIGH. 2 × 12 × 5 = 120 execuções.
Avaliação: 1.000 cenários correlacionados comuns por instância (os mesmos de
robustez_revisor.py), δ = 5 min, α = 0,90.

Implementação: o código da busca (src/) NÃO é alterado. Cada processo deste script
substitui tr._lognormal_media1 por uma versão correlacionada para matrizes K×(n+1),
que são exatamente as duas chamadas do motor (amostra de busca, ccvrp_c.py:176, e
certificação, ccvrp_c.py:392); vetores 1-D seguem a versão original. Depois chama o
_worker do experiment_runner, de modo que o formato de saída é idêntico.
Saída: results_c_cor/<método>/ e pesquisa/resultados_robustez/relatorio_correlacao_busca.txt
Uso: python pesquisa/correlacao_busca.py --rodar --workers 10   (≈ 0,5 h)
     python pesquisa/correlacao_busca.py --analisar
"""
import argparse, glob, json, os, sys
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from scipy.stats import wilcoxon

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(BASE, 'src'))
RHO = 0.5
OUT_RUNS = os.path.join(BASE, 'results_c_cor')
OUT_REL = os.path.join(BASE, 'pesquisa', 'resultados_robustez')
INSTS = [f'CCVRPTW-ST_{f}_050_{u}' for f in ('C1', 'C2', 'R1', 'R2', 'RC1', 'RC2') for u in ('MEDIUM', 'HIGH')]
METODOS = {'C-SIMC-COR': {'runner': 'nsga2c', 'mc_samples': 30,
                          'c_params': {'decoder': 'split', 'p_ls': 0.5, 'mc_rel': 30, 'alpha': 0.90, 'mc_cert': 500}},
           'C-BUF95C-COR': {'runner': 'nsga2c', 'mc_samples': 0,
                            'c_params': {'decoder': 'split', 'p_ls': 0.5, 'buffer_q': 0.95, 'mc_cert': 500}}}
SEEDS = range(5)
K = 1000; ALPHA = 0.90


def lognormal_correlacionada(rng, cv, shape):
    """Média 1 e CV marginal cv; para K×(n+1), fração ρ da variância log comum à linha."""
    s2 = np.log(1.0 + cv * cv); mu = -s2 / 2
    if len(shape) != 2:
        return rng.lognormal(mu, np.sqrt(s2), size=shape)
    z = rng.standard_normal((shape[0], 1)); e = rng.standard_normal(shape)
    return np.exp(mu + np.sqrt(s2) * (np.sqrt(RHO) * z + np.sqrt(1 - RHO) * e))


def _worker_cor(task):
    import thermal_route as tr
    tr._lognormal_media1 = lognormal_correlacionada       # só neste processo
    import experiment_runner as er
    return er._worker(task)


def rodar(workers):
    tarefas = []
    for alg, cfg in METODOS.items():
        out_dir = os.path.join(OUT_RUNS, alg)
        for nome in INSTS:
            for s in SEEDS:
                key = f'{nome}__{alg}__seed{s}'
                if os.path.exists(os.path.join(out_dir, key + '_metrics.json')):
                    continue
                tarefas.append({'inst_path': os.path.join(BASE, 'instances', 'generated', nome + '.json'),
                                'algo': alg, 'algo_label': alg, 'seed': s, 'pop_size': 100, 'n_gen': 200,
                                'mc_samples': cfg['mc_samples'], 'runner': cfg['runner'], 'out_dir': out_dir,
                                'result_key': key, 'use_multizone': False, 'n_zones': 1, 'dedup': False,
                                'c_params': cfg['c_params']})
    os.makedirs(OUT_RUNS, exist_ok=True)
    open(os.path.join(OUT_RUNS, 'LEIA-ME.txt'), 'w', encoding='utf-8').write(
        f'Execuções com tempos de viagem CORRELACIONADOS (ρ = {RHO}) na amostra de busca e na certificação.\n'
        'Gerado por pesquisa/correlacao_busca.py (diário Entrada 84). Motor e parâmetros idênticos a C-SIMC / C-BUF95C.\n')
    print(f'{len(tarefas)} execuções a rodar', flush=True)
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker_cor, tarefas), 1):
            print(f'  [{i}/{len(tarefas)}] {r.get("key", "")} {r.get("status", "")}', flush=True)


# ── análise ─────────────────────────────────────────────────────────────────
_cache = {}


def _cenarios(nome):
    """Os mesmos cenários correlacionados de robustez_revisor.py (semente 777 + nome)."""
    import nsga2_simheuristic as ns
    if nome not in _cache:
        inst = ns.load_instance(os.path.join(BASE, 'instances', 'generated', nome + '.json'))
        _, ttime = ns.build_matrices(inst)
        n = len(inst['customers']); cv = inst['thermal_profile']['sigma_t_pct']
        s2 = np.log(1 + cv * cv); mu = -s2 / 2
        rng = np.random.default_rng(777 + sum(map(ord, nome)))
        _ = np.exp(mu + np.sqrt(s2) * rng.standard_normal((K, n + 1)))      # independente (descartado)
        z = rng.standard_normal((K, 1)); e = rng.standard_normal((K, n + 1))
        cor = np.exp(mu + np.sqrt(s2) * (np.sqrt(RHO) * z + np.sqrt(1 - RHO) * e))
        _cache[nome] = (inst, ttime, cor)
    return _cache[nome]


def _avaliar(path):
    import ccvrp_c as cc
    m = json.load(open(path.replace('_solutions.json', '_metrics.json'), encoding='utf-8'))
    inst, ttime, cor = _cenarios(m['instance'])
    sol = []
    for s in json.load(open(path, encoding='utf-8')):
        if not s['feasible']:
            continue
        on = np.ones(K, dtype=bool)
        for r, d0 in zip(s['routes'], s['partida_min']):
            on &= cc.pontual_rota(r, inst, ttime, cor, d0)
        sol.append({'f1': s['f1_cost_BRL'], 'p': float(on.mean()), 'p_cert': s.get('p_cert')})
    return {'instance': m['instance'], 'seed': m['seed'], 'alg': m['algorithm'], 'sol': sol}


def analisar(workers):
    arqs = []
    for alg in METODOS:
        arqs += glob.glob(os.path.join(OUT_RUNS, alg, '*_solutions.json'))
    for alg in ('C-SIMC', 'C-BUF95C'):
        arqs += [f for f in glob.glob(os.path.join(BASE, 'results_c', alg, '*_050_*_solutions.json'))
                 if any(f'{i}__' in os.path.basename(f) for i in INSTS) and int(f.split('seed')[-1].split('_')[0]) in SEEDS]
    with Pool(workers) as pool:
        res = pool.map(_avaliar, sorted(arqs), chunksize=2)
    os.makedirs(OUT_REL, exist_ok=True)
    rel = open(os.path.join(OUT_REL, 'relatorio_correlacao_busca.txt'), 'w', encoding='utf-8')

    def log(t=''):
        print(t, flush=True); rel.write(t + '\n')

    log(f'CORRELAÇÃO NA BUSCA (ρ = {RHO}; 12 instâncias n=50 MEDIUM/HIGH × sementes 0–4; avaliação em {K} cenários correlacionados)')
    log('colunas: execuções | frente cumpre | escolha do decisor (mais barata certificada) cumpre | custo mínimo que cumpre (média)')
    porinst = {}
    for nivel in ('MEDIUM', 'HIGH', 'TODOS'):
        log(f'  {nivel}')
        for alg in ('C-SIMC', 'C-BUF95C', 'C-SIMC-COR', 'C-BUF95C-COR'):
            rr = [r for r in res if r['alg'] == alg and (nivel == 'TODOS' or r['instance'].endswith(nivel))]
            if not rr:
                continue
            cm = [min((s['f1'] for s in r['sol'] if s['p'] >= ALPHA), default=np.nan) for r in rr]
            esc = []
            for r in rr:
                cert = [s for s in r['sol'] if (s['p_cert'] or 0) >= ALPHA]
                pick = min(cert, key=lambda s: s['f1']) if cert else None
                esc.append(pick is not None and pick['p'] >= ALPHA)
            log(f'    {alg:13s} n={len(rr):3d} | {100*np.mean(np.isfinite(cm)):5.1f}% | {100*np.mean(esc):5.1f}% | {np.nanmean(cm):8.1f}')
            if nivel == 'TODOS':
                acc = defaultdict(list)
                for r, c in zip(rr, cm):
                    if np.isfinite(c):
                        acc[r['instance']].append(c)
                porinst[alg] = {k: np.mean(v) for k, v in acc.items()}
    log('  custo mínimo que cumpre, pareado por instância (média das sementes), Δ% contra C-SIMC-COR:')
    for alg in ('C-BUF95C-COR', 'C-SIMC', 'C-BUF95C'):
        if alg not in porinst or 'C-SIMC-COR' not in porinst:
            continue
        ks = sorted(k for k in porinst[alg] if k in porinst['C-SIMC-COR'])
        d = np.array([100 * (porinst[alg][k] / porinst['C-SIMC-COR'][k] - 1) for k in ks])
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        log(f'    {alg:13s} n={len(ks):2d}  mediana {np.median(d):+.2f}  média {np.mean(d):+.2f}  p={p:.3f}')
    rel.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rodar', action='store_true')
    ap.add_argument('--analisar', action='store_true')
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    if a.rodar:
        rodar(a.workers)
    if a.analisar:
        analisar(a.workers)
