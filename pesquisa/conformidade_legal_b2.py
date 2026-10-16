#!/usr/bin/env python3
"""
Conformidade legal no B2 (10/10/2026, diário Entrada 87).

Com carga MISTA num compartimento único (set-point da categoria dominante, 2 °C), os
congelados viajam acima do limite legal de −12 °C (Portaria CVS 3/2026, Arts. 30 e
109). Este script mede, para cada solução das frentes finais, a fração do valor
entregue cuja temperatura máxima do produto (modelo concentrado de 1ª ordem, τp =
10 h, como no caso-base da Parte B) ultrapassou o limite legal da PRÓPRIA categoria:
  Z = 1 — base de compartimento único com carga mista (results_c_b2/<alg>-MIX/)
  Z = 2, 3 — zonas (results_c_b2/<alg>_Z{2,3}/)
Cenário nominal (perfil horário médio, tempos nominais, porta nominal).
Saída: pesquisa/resultados_c_b2/conformidade_legal.txt e .json
Uso: python pesquisa/conformidade_legal_b2.py [--b2-dir results_c_b2] [--sementes 5]
"""
import argparse, glob, json, os, sys
from collections import defaultdict
from multiprocessing import Pool
import numpy as np

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(BASE, 'src')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
LIMITE = {'meat': 7.0, 'frozen': -12.0, 'dairy': 10.0, 'produce': 10.0}   # CVS 3/2026, Art. 30 (limite superior)
TAU_P = 10.0


def _job(path):
    import nsga2_simheuristic as ns
    import thermal_route as tr
    m = json.load(open(path.replace('_solutions.json', '_metrics.json'), encoding='utf-8'))
    inst = ns.load_instance(os.path.join(BASE, 'instances', 'generated', m['instance'] + '.json'))
    _, ttime = ns.build_matrices(inst); cust = inst['customers']
    mz = bool(m.get('use_multizone')); nz = m['n_zones'] if mz else 1
    nom = tr.cenario_nominal(inst, n_zones=nz)
    ci = ns._cat_index_map(inst) if mz else None
    fr = []
    for s in json.load(open(path, encoding='utf-8')):
        if not s['feasible']:
            continue
        zones = ns.assign_zones(s['routes'], s['zone_block'], inst, ci) if mz else None
        viol = tot = 0.0
        for i, (r, d0) in enumerate(zip(s['routes'], s['partida_min'])):
            if mz:
                zi = zones[i]
                zoc = [zi['zone_of_category'][cust[c - 1]['product_category']] for c in r]
                d = tr.route_quality(r, inst, ttime, nom, zoc, zi['zone_temp'], per_customer_kinetics=True,
                                     tau_p_h=TAU_P, detalhe=True, t0_min=d0)
            else:
                d = tr.route_quality(r, inst, ttime, nom, per_customer_kinetics=True, tau_p_h=TAU_P,
                                     detalhe=True, t0_min=d0)
            lim = np.array([LIMITE[cust[c - 1]['product_category']] for c in r])
            v = d['values']
            viol += float(((d['Tp_max'][0] > lim) * v).sum()); tot += float(v.sum())
        fr.append(viol / tot)
    return {'alg': m['algorithm'], 'instance': m['instance'], 'seed': m['seed'], 'frac': fr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b2-dir', default=os.path.join(BASE, 'results_c_b2'))
    ap.add_argument('--sementes', type=int, default=5)
    ap.add_argument('--out', default=os.path.join(BASE, 'pesquisa', 'resultados_c_b2'))
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    arqs = [f for f in glob.glob(os.path.join(a.b2_dir, '*', '*_solutions.json'))
            if int(f.split('seed')[-1].split('_')[0]) < a.sementes]
    with Pool(a.workers) as pool:
        res = pool.map(_job, sorted(arqs), chunksize=4)
    por = defaultdict(list)
    for r in res:
        if r['frac']:
            por[r['alg']].append(100 * np.mean(r['frac']))
    os.makedirs(a.out, exist_ok=True)
    linhas = ['CONFORMIDADE LEGAL (CVS 3/2026, limite da própria categoria; τp = 10 h; cenário nominal)',
              'fração média do valor entregue acima do limite, por execução (média sobre as soluções da frente)']
    resumo = {}
    for alg in sorted(por):
        v = np.array(por[alg]); resumo[alg] = dict(n=len(v), media=float(v.mean()), min=float(v.min()), max=float(v.max()))
        linhas.append(f'  {alg:22s} n={len(v):3d}  média {v.mean():6.2f}%  (mín {v.min():6.2f}%, máx {v.max():6.2f}%)')
    txt = '\n'.join(linhas); print(txt)
    open(os.path.join(a.out, 'conformidade_legal.txt'), 'w', encoding='utf-8').write(txt + '\n')
    json.dump(resumo, open(os.path.join(a.out, 'conformidade_legal.json'), 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()
