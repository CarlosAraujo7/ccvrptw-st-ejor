#!/usr/bin/env python3
"""
Refinamento sob a regra LCB nos cinco conjuntos (18/10/2026, diário Entrada 117).
Parte da escolha pelo LCB (refinamento_simulacao.py --regra lcb) para CAL, ACAL e SIM+C; grava
refinamento_<partida>_lcb.json ao lado dos resultados da regra pontual (que não são tocados).
Uso: python pesquisa/rodar_regra_lcb.py --workers 6
"""
import argparse, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
S10 = [str(s) for s in range(10)]
ATH = ['063', '066', '067', '069', '074', '075', '077', '078', '084']
# (rótulo, env, res_dir, partidas, caso_sp, níveis, tamanhos, sementes, saída)
CONJ = [
    ('bench', {}, 'results_c', ['CAL', 'C-SIMC'], False, ['MEDIUM', 'HIGH'], ['025', '050', '100'], S10,
     'pesquisa/resultados_refinamento'),
    ('bench_acal', {}, 'results_c_acal_bench', ['C-ACALC-B'], False, ['MEDIUM', 'HIGH'], ['025', '050', '100'], S10,
     'pesquisa/resultados_acal/refinamento_bench'),
    ('sp', {}, 'results_c_sp', ['CAL', 'C-SIMC'], True, ['MEDIUM', 'HIGH'], ['050', '100'], S10,
     'pesquisa/resultados_caso_sp/refinamento'),
    ('sp_acal', {}, 'results_c_acal_sp', ['C-ACALC-B'], True, ['MEDIUM', 'HIGH'], ['050', '100'], S10,
     'pesquisa/resultados_acal/refinamento_sp'),
    ('spt', {'CCVRP_SP_VARIANTE': 'td'}, 'results_c_sp_td', ['CAL', 'C-SIMC'], True, ['MEDIUM', 'HIGH'], ['050', '100'], S10,
     'pesquisa/resultados_caso_sp_td/refinamento'),
    ('spt_acal', {'CCVRP_SP_VARIANTE': 'td'}, 'results_c_acal_sp_td', ['C-ACALC-B'], True, ['MEDIUM', 'HIGH'], ['050', '100'],
     S10, 'pesquisa/resultados_acal/refinamento_sp_td'),
    ('ath', {'CCVRP_SP_VARIANTE': 'atenas'}, 'results_c_atenas', ['CAL', 'C-SIMC'], True, ['DATA'], ATH, S10,
     'pesquisa/resultados_caso_atenas/refinamento'),
    ('ath_acal', {'CCVRP_SP_VARIANTE': 'atenas'}, 'results_c_acal_atenas', ['C-ACALC-B'], True, ['DATA'], ATH, S10,
     'pesquisa/resultados_acal/refinamento_atenas'),
    ('escala', {'CCVRP_SP_INST_DIR': os.path.join(BASE, 'instances', 'escala')}, 'results_c_escala',
     ['CAL', 'C-ACALC-B', 'C-SIMC'], True, ['MEDIUM', 'HIGH'], ['200', '400'], ['0', '1', '2'],
     'pesquisa/resultados_escala/refinamento'),
]


def job(t):
    rot, env, res, partida, sp, niv, tam, sem, saida = t
    log = os.path.join(BASE, 'pesquisa', 'resultados_regra_lcb', f'log_{rot}_{partida}.txt')
    cmd = [sys.executable, os.path.join(BASE, 'pesquisa', 'refinamento_simulacao.py'), '--regra', 'lcb', '--res-dir', res,
           '--niveis', *niv, '--tamanhos', *tam, '--sementes', *sem, '--partida', partida, '--saida', os.path.join(BASE, saida)]
    if sp:
        cmd.append('--caso-sp')
    e = dict(os.environ, PYTHONIOENCODING='utf-8', **env)
    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as fh:
        r = subprocess.run(cmd, cwd=BASE, env=e, stdout=fh, stderr=subprocess.STDOUT)
    return f'{rot} {partida}: código {r.returncode}, {time.time() - t0:.0f} s'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    os.makedirs(os.path.join(BASE, 'pesquisa', 'resultados_regra_lcb'), exist_ok=True)
    T = [(c[0], c[1], c[2], p, *c[4:]) for c in CONJ for p in c[3]]
    print(f'{len(T)} tarefas; início {time.strftime("%H:%M:%S")}', flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for m in ex.map(job, T):
            print(' ', m, flush=True)
    print(f'FIM REGRA LCB {time.strftime("%H:%M:%S")}', flush=True)


if __name__ == '__main__':
    main()
