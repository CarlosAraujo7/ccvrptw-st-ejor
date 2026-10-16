#!/usr/bin/env python3
"""
Teste rápido do motor de busca contra a referência do PyVRP (09/10/2026, diário
Entrada 85). Pergunta: quanto do gap de custo do DET em n = 50/100 (+2,8%/+5,0% na
melhor semente, contra o PyVRP) cai com mais busca local ou mais gerações?
Configurações (DET, sem pontualidade; só o custo mínimo da frente importa aqui;
A-C rodadas em 09/10 antes da eliminação de rotas, E-F depois — Entrada 85):
  A  p_ls = 0,2, 200 gerações (atual)
  B  p_ls = 1,0, 200 gerações (busca local em todo filho)
  C  p_ls = 0,2, 500 gerações
Instâncias: n = 100, uma por família (MEDIUM); semente 0. Não altera src/.
Saída: pesquisa/resultados_pyvrp/teste_motor_custo.txt
"""
import csv, json, os, sys, time
from multiprocessing import Pool
import numpy as np

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(BASE, 'src'))
CONFIGS = {'A': dict(p_ls=0.2, n_gen=200, elim=False), 'B': dict(p_ls=1.0, n_gen=200, elim=False),
           'C': dict(p_ls=0.2, n_gen=500, elim=False),
           # Entrada 85: com eliminação de rotas na busca local
           'E': dict(p_ls=0.2, n_gen=200, elim=True), 'F': dict(p_ls=0.5, n_gen=200, elim=True)}
TAMANHOS = ['100']
INSTS = [f'CCVRPTW-ST_{f}_{n}_MEDIUM' for n in TAMANHOS for f in ('C1', 'C2', 'R1', 'R2', 'RC1', 'RC2')]


def job(args):
    nome, cfg = args
    import nsga2_simheuristic as ns
    import ccvrp_c as cc
    cc.ELIMINAR_ROTAS = CONFIGS[cfg]['elim']
    inst = ns.load_instance(os.path.join(BASE, 'instances', 'generated', nome + '.json'))
    t0 = time.time()
    pareto, _ = cc.run_nsga2_c(inst, pop_size=100, n_gen=CONFIGS[cfg]['n_gen'], seed=0,
                               decoder='split', p_ls=CONFIGS[cfg]['p_ls'])
    best = min(x.f_cost for x in pareto if x.feasible)
    return nome, cfg, best, time.time() - t0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', default=list(CONFIGS))
    ap.add_argument('--tamanhos', nargs='+', default=TAMANHOS)
    ap.add_argument('--saida', default='teste_motor_custo.txt')
    a = ap.parse_args()
    insts = [f'CCVRPTW-ST_{f}_{n}_MEDIUM' for n in a.tamanhos for f in ('C1', 'C2', 'R1', 'R2', 'RC1', 'RC2')]
    ref = {}
    for r in csv.DictReader(open(os.path.join(BASE, 'pesquisa', 'resultados_pyvrp', 'pyvrp.csv'), encoding='utf-8')):
        if r['viavel'] == 'True':
            ref[r['instance']] = min(ref.get(r['instance'], np.inf), float(r['custo']))
    with Pool(9) as pool:
        res = pool.map(job, [(i, c) for i in insts for c in a.configs], chunksize=1)
    out = open(os.path.join(BASE, 'pesquisa', 'resultados_pyvrp', a.saida), 'w', encoding='utf-8')
    for n, cfg in [(n, c) for n in a.tamanhos for c in a.configs]:
        rr = [r for r in res if r[1] == cfg and f'_{n}_' in r[0]]
        gaps = [100 * (b / ref[i] - 1) for i, _, b, _ in rr]
        t = np.mean([r[3] for r in rr])
        linha = (f"n={n} {cfg} {CONFIGS[cfg]}: gap médio {np.mean(gaps):+.2f}% (máx {np.max(gaps):+.2f}%), "
                 f"tempo médio {t:.0f} s  | por instância: " + ', '.join(f'{g:+.1f}' for g in gaps))
        print(linha, flush=True); out.write(linha + '\n')
    out.close()


if __name__ == '__main__':
    main()
