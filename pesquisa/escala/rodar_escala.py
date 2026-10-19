#!/usr/bin/env python3
"""
Teste de escala (17/10/2026, diário Entrada 114; análise de fraquezas A3), desenho fixado antes de rodar.

Instâncias: as 24 de instances/escala (C1, C2, R1, R2, RC1, RC2; n = 200 e 400; MEDIUM e HIGH).
  --motor   métodos do motor próprio, sementes 0-2, mesma configuração da execução principal (população 100, 200
            gerações): C-SIMC, a escada C-BUF70C/80C/90C/95C (que compõe o CAL) e C-ACALC-B; 432 execuções em
            results_c_escala; avaliação fora da amostra nos 1.000 cenários comuns (analisar_c, INST_DIR = escala).
  --pyvrp   pipeline com o PyVRP (pipeline_pyvrp_buffer.tarefa): degraus 0,80/0,90/0,95, sementes 0-2, tempo por
            resolução 240 s (n = 200) e 480 s (n = 400); certificação, refinamento e avaliação iguais às do artigo;
            216 resoluções em pesquisa/resultados_escala/pipeline_escala.jsonl.
Uso: python pesquisa/escala/rodar_escala.py --motor --pyvrp --workers 10
"""
import argparse, glob, json, os, random, sys, time
from multiprocessing import Pool

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(BASE, 'src')); sys.path.insert(0, os.path.join(BASE, 'pesquisa'))
sys.path.insert(0, os.path.join(BASE, 'pesquisa', 'caso_sp'))
IDIR = os.path.join(BASE, 'instances', 'escala')
RES = os.path.join(BASE, 'results_c_escala')
OUT = os.path.join(BASE, 'pesquisa', 'resultados_escala')
METODOS = ['C-SIMC', 'C-BUF70C', 'C-BUF80C', 'C-BUF90C', 'C-BUF95C', 'C-ACALC-B']
SEMENTES = (0, 1, 2)


def _init():
    import rodar_caso_sp
    rodar_caso_sp.aplicar_matrizes()
    import analisar_c as ac
    ac.INST_DIR = IDIR
    import referencia_pyvrp as rp
    rp.TEMPO.update({200: 240, 400: 480})


def _rodar(t):
    import experiment_runner as er
    return er._worker(t)


def _oos(path):
    import analisar_c as ac
    ac.INST_DIR = IDIR
    return ac._job(path)


def _pyvrp(args):
    import pipeline_pyvrp_buffer as pp
    return pp.tarefa(args)


def motor(workers):
    import experiment_runner as er
    T = []
    for f in sorted(glob.glob(os.path.join(IDIR, 'CCVRPTW-ST_*.json'))):
        nm = os.path.basename(f)[:-5]
        for alg in METODOS:
            cfg = er.ALGO_CONFIG[alg]; out_dir = os.path.join(RES, alg)
            for s in SEMENTES:
                key = f'{nm}__{alg}__seed{s}'
                if os.path.exists(os.path.join(out_dir, key + '_metrics.json')):
                    continue
                T.append({'inst_path': f, 'algo': alg, 'algo_label': alg, 'seed': s, 'pop_size': 100, 'n_gen': 200,
                          'mc_samples': cfg['mc_samples'], 'runner': cfg['runner'], 'out_dir': out_dir, 'result_key': key,
                          'use_multizone': False, 'n_zones': 1, 'dedup': cfg.get('dedup', False), 'c_params': cfg.get('c_params')})
    T.sort(key=lambda t: -int(os.path.basename(t['inst_path']).split('_')[2]))      # as maiores primeiro
    print(f'motor: {len(T)} execuções a rodar', flush=True)
    t0 = time.time()
    with Pool(workers, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_rodar, T), 1):
            print(f"  [{i}/{len(T)}] {r.get('key', '')} {r.get('status', '')} {r.get('elapsed', '')}", flush=True)
    print(f'motor concluído em {time.time() - t0:.0f} s; avaliação fora da amostra...', flush=True)
    with Pool(workers, initializer=_init) as pool:
        n = sum(pool.map(_oos, sorted(glob.glob(os.path.join(RES, 'C-*', '*_solutions.json'))), chunksize=2))
    print(f'{n} avaliações novas; FIM MOTOR ESCALA', flush=True)


def pyvrp(workers):
    os.makedirs(OUT, exist_ok=True)
    arq = os.path.join(OUT, 'pipeline_escala.jsonl')
    feitas = set()
    if os.path.exists(arq):
        for L in open(arq, encoding='utf-8'):
            x = json.loads(L); feitas.add((x['instance'], x['q'], x['seed']))
    T = [(p, q, s, False) for p in sorted(glob.glob(os.path.join(IDIR, 'CCVRPTW-ST_*.json')))
         for q in (0.80, 0.90, 0.95) for s in SEMENTES]
    T = [t for t in T if (os.path.basename(t[0])[:-5], t[1], t[2]) not in feitas]
    random.Random(2026).shuffle(T)
    print(f'PyVRP: {len(T)} resoluções a fazer', flush=True)
    with Pool(workers, initializer=_init) as pool, open(arq, 'a', encoding='utf-8') as fh:
        for i, r in enumerate(pool.imap_unordered(_pyvrp, T), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            print(f"  [{i}/{len(T)}] {r['instance']} q={r['q']} s{r['seed']} viável={r['viavel']} t={r['t_solve_s']}s "
                  f"P_cert {r['base_p_cert']:.3f} -> {r['ref_p_cert']:.3f}", flush=True)
    print('FIM PYVRP ESCALA', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--motor', action='store_true'); ap.add_argument('--pyvrp', action='store_true')
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--sementes', nargs='+', type=int, default=list(SEMENTES),
                    help='sementes (padrão 0-2; B3, Entrada 118: 3-9)')
    a = ap.parse_args()
    SEMENTES = tuple(a.sementes)
    if a.motor:
        motor(a.workers)
    if a.pyvrp:
        pyvrp(a.workers)
