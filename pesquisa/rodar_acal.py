#!/usr/bin/env python3
"""
Calibração dentro de uma busca (ACAL) — execução e avaliação fora da amostra (17/10/2026, diário Entrada 109).

  --fase dev    variantes C-ACALC-A e C-ACALC-B, sementes 10-12 (não usadas no teste), em 12 instâncias do benchmark
                (n = 50, MEDIUM/HIGH, seis famílias), 4 do SP (SP1/SP2, n = 50) e 3 de Atenas (ATH1, ATH3, ATH5);
  --fase teste  a variante escolhida no desenvolvimento, sementes 0-9, em todas as instâncias de teste: benchmark v2
                (54), SP (16), SP com horário (16) e Atenas (9); 950 execuções.
Resultados em pastas próprias (results_c_acal_<conjunto>), para não misturar com as execuções do artigo; a avaliação
fora da amostra usa os mesmos 1.000 cenários comuns de analisar_c (mesma semente por instância).
Uso: python pesquisa/rodar_acal.py --fase dev --workers 10   |   --fase teste --variante A --workers 10
"""
import argparse, glob, json, os, sys, time
from multiprocessing import Pool

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(BASE, 'src')); sys.path.insert(0, os.path.join(BASE, 'pesquisa'))
sys.path.insert(0, os.path.join(BASE, 'pesquisa', 'caso_sp'))

CONJ = {'bench': ('instances/generated', 'results_c_acal_bench'), 'SP': ('instances/caso_sp', 'results_c_acal_sp'),
        'SPT': ('instances/caso_sp_td', 'results_c_acal_sp_td'), 'ATH': ('instances/caso_atenas', 'results_c_acal_atenas')}
DEV = {'bench': [f'CCVRPTW-ST_{f}_050_{u}' for f in ('C1', 'C2', 'R1', 'R2', 'RC1', 'RC2') for u in ('MEDIUM', 'HIGH')],
       'SP': [f'CCVRPTW-ST_SP{a}_050_{u}' for a in (1, 2) for u in ('MEDIUM', 'HIGH')],
       'ATH': ['CCVRPTW-ST_ATH1_078_DATA', 'CCVRPTW-ST_ATH3_067_DATA', 'CCVRPTW-ST_ATH5_075_DATA']}


def _init():
    import rodar_caso_sp
    rodar_caso_sp.aplicar_matrizes()


def _rodar(t):
    _init()
    import experiment_runner as er
    return er._worker(t)


def _oos(args):
    path, idir = args
    _init()
    import analisar_c as ac
    ac.INST_DIR = os.path.join(BASE, idir)
    return ac._job(path)


def tarefas(fase, variante):
    import experiment_runner as er
    T = []
    for c, (idir, res) in CONJ.items():
        if fase == 'dev':
            if c not in DEV:
                continue
            nomes, sementes, algs = DEV[c], (10, 11, 12), ('C-ACALC-A', 'C-ACALC-B')
            res = res + '_dev'
        else:
            base_res = {'bench': 'results_c', 'SP': 'results_c_sp', 'SPT': 'results_c_sp_td', 'ATH': 'results_c_atenas'}[c]
            nomes = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(BASE, base_res, 'C-SIMC', '*_oosC.json'))})
            sementes, algs = range(10), (f'C-ACALC-{variante}',)
        for nm in nomes:
            for alg in algs:
                cfg = er.ALGO_CONFIG[alg]; out_dir = os.path.join(BASE, res, alg)
                for s in sementes:
                    key = f'{nm}__{alg}__seed{s}'
                    if os.path.exists(os.path.join(out_dir, key + '_metrics.json')):
                        continue
                    T.append({'inst_path': os.path.join(BASE, idir, nm + '.json'), 'algo': alg, 'algo_label': alg, 'seed': s,
                              'pop_size': 100, 'n_gen': 200, 'mc_samples': cfg['mc_samples'], 'runner': cfg['runner'],
                              'out_dir': out_dir, 'result_key': key, 'use_multizone': False, 'n_zones': 1,
                              'dedup': cfg.get('dedup', False), 'c_params': cfg.get('c_params')})
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fase', choices=['dev', 'teste'], required=True)
    ap.add_argument('--variante', default='A')
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    T = tarefas(a.fase, a.variante)
    print(f'{len(T)} execuções a rodar', flush=True)
    t0 = time.time()
    with Pool(a.workers, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_rodar, T), 1):
            print(f"  [{i}/{len(T)}] {r.get('key', '')} {r.get('status', '')} {r.get('elapsed', '')}", flush=True)
    print(f'execuções concluídas em {time.time() - t0:.0f} s; avaliação fora da amostra...', flush=True)
    jobs = []
    for c, (idir, res) in CONJ.items():
        pasta = res + ('_dev' if a.fase == 'dev' else '')
        jobs += [(f, idir) for f in glob.glob(os.path.join(BASE, pasta, 'C-ACALC-*', '*_solutions.json'))]
    with Pool(a.workers, initializer=_init) as pool:
        n = sum(pool.map(_oos, jobs, chunksize=2))
    print(f'{n} avaliações novas; FIM ACAL {a.fase}', flush=True)


if __name__ == '__main__':
    main()
