#!/usr/bin/env python3
"""
Tamanho da amostra de busca do simheurístico (08/10/2026, diário Entrada 81).
Compara C-SIMC100 (100 cenários de tempo de viagem na busca; results_c_extra/) com C-SIMC
(30 cenários; results_c/) nas MESMAS 12 instâncias (n=50, MEDIUM e HIGH) e sementes 0–4.
Reaproveita a avaliação fora da amostra de analisar_c.py (mesmos 1.000 cenários comuns por
instância; cria *_oosC.json para C-SIMC100). Por execução:
  otimismo  = média, sobre a frente, de P_on na amostra de busca − P_on fora da amostra
  cumpre    = a frente tem solução com P_on >= α fora da amostra
  custo     = custo mínimo entre as que cumprem fora da amostra (conteúdo da frente)
  escolha   = a mais barata aprovada na certificação cumpre fora da amostra (visão do decisor)
  tempo     = elapsed_s
Média das sementes por instância; Wilcoxon pareado (n = 12 instâncias).
Saída: pesquisa/resultados_robustez/relatorio_simc100.txt
"""
import glob, json, os, sys
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from scipy.stats import wilcoxon

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(BASE, 'pesquisa'))
import analisar_c as ac  # noqa: E402

ALPHA = 0.90
OUT = os.path.join(BASE, 'pesquisa', 'resultados_robustez')
DIRS = {'C-SIMC100': os.path.join(BASE, 'results_c_extra', 'C-SIMC100'),
        'C-SIMC': os.path.join(BASE, 'results_c', 'C-SIMC')}


def por_execucao(alg, insts):
    d = {}
    for f in glob.glob(os.path.join(DIRS[alg], '*_oosC.json')):
        r = json.load(open(f, encoding='utf-8')); m = r['meta']
        if m['instance'] not in insts or m['seed'] > 4:
            continue
        met = json.load(open(f.replace('_oosC.json', '_metrics.json'), encoding='utf-8'))
        S = [s for s in r['solutions'] if s['feasible']]
        otim = [s['p_on_insample'] - s['p_on'] for s in S if s.get('p_on_insample') is not None]
        cert = [s for s in S if (s.get('p_cert') or 0) >= ALPHA]
        pick = min(cert, key=lambda s: s['f1']) if cert else None
        d[(m['instance'], m['seed'])] = {
            'otim': np.mean(otim) if otim else np.nan,
            'cumpre': any(s['p_on'] >= ALPHA for s in S),
            'custo': min((s['f1'] for s in S if s['p_on'] >= ALPHA), default=np.nan),
            'escolha': (pick is not None and pick['p_on'] >= ALPHA),
            'tempo': met['elapsed_s']}
    return d


def main():
    files = sorted(glob.glob(os.path.join(DIRS['C-SIMC100'], '*_solutions.json')))
    with Pool(6) as pool:
        pool.map(ac._job, files, chunksize=2)
    insts = {json.load(open(f.replace('_solutions.json', '_metrics.json'), encoding='utf-8'))['instance'] for f in files}
    os.makedirs(OUT, exist_ok=True)
    rel = open(os.path.join(OUT, 'relatorio_simc100.txt'), 'w', encoding='utf-8')

    def log(t=''):
        print(t, flush=True); rel.write(t + '\n')

    D = {a: por_execucao(a, insts) for a in DIRS}
    comuns = sorted(set(D['C-SIMC100']) & set(D['C-SIMC']))
    log(f'SIM+C com 100 vs 30 cenários na busca: {len(insts)} instâncias, {len(comuns)} execuções pareadas (sementes 0–4), α={ALPHA}')
    for nivel in ('MEDIUM', 'HIGH', 'TODOS'):
        ks = [k for k in comuns if nivel == 'TODOS' or k[0].endswith(nivel)]
        log(f'  {nivel} ({len(ks)} execuções)')
        for a in ('C-SIMC', 'C-SIMC100'):
            v = [D[a][k] for k in ks]
            log(f"    {a:10s} otimismo {np.nanmean([x['otim'] for x in v]):+.4f} | frente cumpre {100*np.mean([x['cumpre'] for x in v]):5.1f}% | "
                f"escolha cumpre {100*np.mean([x['escolha'] for x in v]):5.1f}% | tempo médio {np.mean([x['tempo'] for x in v]):6.1f} s")
    # custo por instância (média das sementes em que ambos cumprem)
    pi = defaultdict(lambda: defaultdict(list))
    for k in comuns:
        if np.isfinite(D['C-SIMC'][k]['custo']) and np.isfinite(D['C-SIMC100'][k]['custo']):
            for a in DIRS:
                pi[k[0]][a].append(D[a][k]['custo'])
    xs = sorted(pi)
    a100 = np.array([np.mean(pi[i]['C-SIMC100']) for i in xs]); a30 = np.array([np.mean(pi[i]['C-SIMC']) for i in xs])
    dd = 100 * (a100 / a30 - 1)
    p = wilcoxon(a100, a30).pvalue if np.any(dd != 0) else 1.0
    log(f'  custo mínimo que cumpre, 100 vs 30 cenários: n={len(xs)} instâncias  Δ% mediana {np.median(dd):+.2f}  média {np.mean(dd):+.2f}  p={p:.3f}')
    oti = defaultdict(lambda: defaultdict(list))
    for k in comuns:
        for a in DIRS:
            oti[k[0]][a].append(D[a][k]['otim'])
    o100 = np.array([np.nanmean(oti[i]['C-SIMC100']) for i in xs]); o30 = np.array([np.nanmean(oti[i]['C-SIMC']) for i in xs])
    log(f'  otimismo por instância: 30 cen. média {o30.mean():+.4f}, 100 cen. média {o100.mean():+.4f}, Wilcoxon p={wilcoxon(o100, o30).pvalue:.3f}')
    rel.close()


if __name__ == '__main__':
    main()
