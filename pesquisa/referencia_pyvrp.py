#!/usr/bin/env python3
"""
Referência externa para a qualidade da busca (08/10/2026, diário Entrada 84).

Crítica de revisor: a validação exata só vai até n = 20; em n = 50 e n = 100 não se
sabe quão longe do ótimo o motor de busca fica. Este script resolve o problema de
CUSTO (f1) de cada instância com o PyVRP 0.13.3 (algoritmo genético híbrido, estado da
arte para VRPTW; instalado com aval do pesquisador) e compara com o custo mínimo da
frente do DET (Split + busca local) nas mesmas instâncias.

Mesmo problema da nossa busca, nos mesmos termos de src/ccvrp_c.py:
  custo = Σ_rotas (fixo + R$/km × km) + penalidade × veículos além da frota;
  capacidade; chegada ≤ fim da janela (espera permitida); tempo de serviço;
  tempos nominais; SEM exigência de retorno ao depósito até o horizonte (o teste
  _rota_viavel da nossa busca também não exige).
A penalidade por veículo extra entra como um segundo tipo de veículo, ilimitado, com
custo fixo = fixo + penalidade. Distâncias e tempos viram inteiros (escala 10^4, ou
seja, 0,1 m e 0,006 s); o custo final de cada solução do PyVRP é RECALCULADO em BRL e a
viabilidade é RECONFERIDA com as funções da nossa busca (_custo_rota, _rota_viavel).
Validação do mapeamento: as 4 instâncias da validação exata (n = 5 a 20) devem
reproduzir os ótimos do MILP (485,84; 879,64; 1.464,02 incumbente; 1.281,15).

Saída: pesquisa/resultados_pyvrp/pyvrp.csv (+ relatório com o gap do DET, se houver
results_c/C-DET). Uso: python pesquisa/referencia_pyvrp.py [--workers 4] [--so-gap]
"""
import argparse, csv, glob, json, math, os, sys
from multiprocessing import Pool
import numpy as np

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(BASE, 'src'))
import nsga2_simheuristic as ns   # noqa: E402
import ccvrp_c as cc              # noqa: E402

OUT = os.path.join(BASE, 'pesquisa', 'resultados_pyvrp')
ESC = 10_000                      # escala de distância (km) e tempo (min) para inteiros
TEMPO = {5: 5, 10: 5, 15: 10, 20: 10, 25: 30, 50: 60, 100: 120}   # s por resolução
SEMENTES = (1, 2, 3)
EXATOS = {5: 485.84, 10: 879.64, 15: 1464.02, 20: 1281.15}


def _caso_sp():
    sys.path.insert(0, os.path.join(BASE, 'pesquisa', 'caso_sp'))
    import rodar_caso_sp
    rodar_caso_sp.aplicar_matrizes()


def resolver(args):
    path, seed = args[0], args[1]
    if len(args) > 2 and args[2]:
        _caso_sp()
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime
    inst = ns.load_instance(path)
    ctx = cc.Contexto(inst, 0, decoder='split', p_ls=0.0)
    cs = inst['customers']; n = len(cs); fl = inst['fleet']
    fix, var, K = fl['fixed_cost_BRL'], fl['variable_cost_BRL_km'], fl['num_vehicles']
    pen = ns.EXTRA_VEHICLE_COST          # 400 BRL, a mesma constante da busca
    # unidade de custo = var / ESC BRL (1 por unidade de distância de 10^-4 km)
    u = var / ESC
    m = Model()
    BIG = 10 ** 12
    dep = m.add_depot(x=0, y=0, tw_early=0, tw_late=BIG)
    m.add_vehicle_type(num_available=K, capacity=int(round(fl['capacity_kg'] * 100)),
                       fixed_cost=int(round(fix / u)), unit_distance_cost=1, tw_early=0, tw_late=BIG)
    m.add_vehicle_type(num_available=n, capacity=int(round(fl['capacity_kg'] * 100)),
                       fixed_cost=int(round((fix + pen) / u)), unit_distance_cost=1, tw_early=0, tw_late=BIG)
    cl = [m.add_client(x=0, y=0, delivery=int(round(c['demand_kg'] * 100)),
                       service_duration=int(round(c['service_time_min'] * ESC)),
                       tw_early=int(round(c['ready_time_min'] * ESC)),
                       tw_late=int(math.floor(c['due_date_min'] * ESC))) for c in cs]
    locs = [dep] + cl
    for i, a in enumerate(locs):
        for j, b in enumerate(locs):
            if i != j:
                m.add_edge(a, b, distance=int(round(ctx.dist[i][j] * ESC)),
                           duration=int(math.ceil(ctx.ttime[i][j] * ESC)))
    res = m.solve(stop=MaxRuntime(TEMPO[n]), seed=seed, display=False)
    best = res.best
    rotas = [[int(v) for v in r.visits()] for r in best.routes()]
    viavel = res.is_feasible() and all(cc._rota_viavel(r, ctx) for r in rotas) \
        and sorted(c for r in rotas for c in r) == list(range(1, n + 1))
    custo = sum(cc._custo_rota(r, ctx) for r in rotas) + pen * max(0, len(rotas) - K)
    return dict(instance=inst['name'], n=n, seed=seed, tempo_s=TEMPO[n], custo=round(custo, 2),
                rotas=len(rotas), viavel=viavel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--so-gap', action='store_true', help='só recalcula o gap a partir de pyvrp.csv')
    ap.add_argument('--det-dir', default=os.path.join(BASE, 'results_c', 'C-DET'))
    ap.add_argument('--caso-sp', action='store_true', help='instâncias do caso São Paulo (matrizes viárias)')
    a = ap.parse_args()
    global OUT
    if a.caso_sp:
        OUT = os.path.join(BASE, 'pesquisa', 'resultados_caso_sp', 'pyvrp')
        if a.det_dir == os.path.join(BASE, 'results_c', 'C-DET'):
            a.det_dir = os.path.join(BASE, 'results_c_sp', 'C-DET')
    os.makedirs(OUT, exist_ok=True)
    arq = os.path.join(OUT, 'pyvrp.csv')
    if not a.so_gap:
        if a.caso_sp:
            insts = sorted(glob.glob(os.path.join(BASE, 'instances', 'caso_sp', 'CCVRPTW-ST_SP*.json')))
        else:
            insts = sorted(glob.glob(os.path.join(BASE, 'instances', 'validacao_exata', 'CCVRPTW-ST_valexata_C1_*_LOW.json'))) \
                + sorted(glob.glob(os.path.join(BASE, 'instances', 'generated', 'CCVRPTW-ST_*.json')))
        tarefas = [(p, s, a.caso_sp) for p in insts for s in SEMENTES]
        with Pool(a.workers) as pool:
            linhas = []
            for i, r in enumerate(pool.imap_unordered(resolver, tarefas), 1):
                linhas.append(r)
                if i % 20 == 0 or i == len(tarefas):
                    print(f'  {i}/{len(tarefas)}', flush=True)
        with open(arq, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(linhas[0])); w.writeheader(); w.writerows(sorted(linhas, key=lambda r: (r['instance'], r['seed'])))
    L = list(csv.DictReader(open(arq, encoding='utf-8')))
    melhor = {}
    for r in L:
        if r['viavel'] == 'True':
            melhor[r['instance']] = min(melhor.get(r['instance'], math.inf), float(r['custo']))
    rel = open(os.path.join(OUT, 'relatorio_pyvrp.txt'), 'w', encoding='utf-8')

    def log(t=''):
        print(t, flush=True); rel.write(t + '\n')

    log(f'PyVRP 0.13.3, {len(SEMENTES)} sementes por instância; tempo por resolução {TEMPO}; soluções inviáveis pela nossa regra: '
        f"{sum(r['viavel'] != 'True' for r in L)} de {len(L)}")
    log('Validação do mapeamento (instâncias da validação exata):' if not a.caso_sp else '(caso São Paulo: sem validação exata)')
    for nn, ot in ({} if a.caso_sp else EXATOS).items():
        k = f'CCVRPTW-ST_C1_{nn:03d}_LOW'          # nome interno das instâncias da validação exata
        log(f'   n={nn:2d}  MILP {ot:9.2f}   PyVRP {melhor.get(k, float("nan")):9.2f}')
    if not os.path.isdir(a.det_dir):
        log(f'\n(sem {a.det_dir}: gap do DET não calculado)'); rel.close(); return
    det = {}
    for f in glob.glob(os.path.join(a.det_dir, '*_solutions.json')):
        mt = json.load(open(f.replace('_solutions.json', '_metrics.json'), encoding='utf-8'))
        S = [s['f1_cost_BRL'] for s in json.load(open(f, encoding='utf-8')) if s['feasible']]
        if S:
            det.setdefault(mt['instance'], []).append(min(S))
    log(f'\nGap do custo mínimo da frente do DET (Split + BL) contra o melhor do PyVRP, por tamanho ({a.det_dir}):')
    for nn in (25, 50, 100):
        ks = [k for k in det if k in melhor and f'_{nn:03d}_' in k]
        if not ks:
            continue
        g_best = [100 * (min(det[k]) / melhor[k] - 1) for k in ks]
        g_mean = [100 * (np.mean(det[k]) / melhor[k] - 1) for k in ks]
        log(f'   n={nn:3d} ({len(ks)} instâncias): melhor semente média {np.mean(g_best):+.2f}% (máx {np.max(g_best):+.2f}%), '
            f'média das sementes {np.mean(g_mean):+.2f}% (máx {np.max(g_mean):+.2f}%)')
    rel.close()


if __name__ == '__main__':
    main()
