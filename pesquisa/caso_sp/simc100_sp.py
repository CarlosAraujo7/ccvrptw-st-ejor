#!/usr/bin/env python3
"""
SIM+C com 100 cenários de busca no caso São Paulo (16/10/2026, análise de fraquezas A3 do ciclo de 16/10;
aprovado pelo pesquisador: "Pode rodar o teste com 100 cenários").

Crítica que o teste responde: "no caso SP a simheurística perde para os buffers só porque busca com 30
cenários". Roda C-SIMC100 (mesma configuração da execução principal, 100 cenários na busca) nas 16
instâncias SP com 10 sementes (160 execuções), numa pasta própria (results_c_sp_simc100), para não mexer
nas famílias de testes do relatório principal; avalia nos MESMOS 1.000 cenários comuns e compara, por
instância (média das sementes), o plano escolhido (mais barato certificado) e o custo mínimo que cumpre com
os de SIM+C (30 cenários), BUF0.95+C, CAL e CAL+R.
Uso: python pesquisa/caso_sp/simc100_sp.py --rodar --workers 10   depois   --analisar
Saída: pesquisa/resultados_caso_sp/relatorio_simc100_sp.txt
"""
import argparse, glob, json, os, sys
from collections import defaultdict
from multiprocessing import Pool
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(AQUI, '..', '..'))
sys.path.insert(0, os.path.join(BASE, 'src')); sys.path.insert(0, os.path.join(BASE, 'pesquisa')); sys.path.insert(0, AQUI)
import rodar_caso_sp as rc   # noqa: E402

RES = os.path.join(BASE, 'results_c_sp_simc100')
ALPHA = 0.90
ESCADA = ['C-BUF70C', 'C-BUF80C', 'C-BUF90C', 'C-BUF95C']


def rodar(workers):
    import experiment_runner as er
    cfg = er.ALGO_CONFIG['C-SIMC100']; out = os.path.join(RES, 'C-SIMC100')
    tarefas = []
    for f in sorted(glob.glob(os.path.join(rc.INST_DIR, 'CCVRPTW-ST_SP*.json'))):
        nome = os.path.basename(f)[:-5]
        for s in range(10):
            key = f'{nome}__C-SIMC100__seed{s}'
            if not os.path.exists(os.path.join(out, key + '_metrics.json')):
                tarefas.append({'inst_path': f, 'algo': 'C-SIMC100', 'algo_label': 'C-SIMC100', 'seed': s, 'pop_size': 100,
                                'n_gen': 200, 'mc_samples': cfg['mc_samples'], 'runner': cfg['runner'], 'out_dir': out,
                                'result_key': key, 'use_multizone': False, 'n_zones': 1, 'dedup': cfg.get('dedup', False),
                                'c_params': cfg.get('c_params')})
    print(f'{len(tarefas)} execuções a rodar', flush=True)
    with Pool(workers, initializer=rc.aplicar_matrizes) as pool:
        for i, r in enumerate(pool.imap_unordered(rc._worker, tarefas), 1):
            print(f"  [{i}/{len(tarefas)}] {r.get('key', '')} {r.get('status', '')} {r.get('elapsed', '')}", flush=True)
    rc.aplicar_matrizes()
    import analisar_c as ac
    arqs = sorted(glob.glob(os.path.join(RES, 'C-*', '*_solutions.json')))
    with Pool(workers, initializer=rc.aplicar_matrizes) as pool:
        pool.map(ac._job, arqs, chunksize=2)


def resumo(arq_oos):
    S = [s for s in json.load(open(arq_oos, encoding='utf-8'))['solutions'] if s['feasible']]
    ok = [s for s in S if s['p_on'] >= ALPHA]
    cert = [s for s in S if (s.get('p_cert') or 0) >= ALPHA]
    e = min(cert, key=lambda s: s['f1']) if cert else None
    return dict(custo_ok=min(s['f1'] for s in ok) if ok else np.nan,
                esc=(e['f1'], e['p_on'], e['q_oos'], e['f3']) if e else None)


def analisar():
    from scipy.stats import wilcoxon
    from equivalencia import hodges_lehmann
    from analisar_refinamento import holm
    run = defaultdict(dict)
    for pasta, met in ((os.path.join(RES, 'C-SIMC100'), 'SIM100'), (os.path.join(BASE, 'results_c_sp', 'C-SIMC'), 'SIM30'),
                       (os.path.join(BASE, 'results_c_sp', 'C-BUF95C'), 'BUF0.95+C')):
        for f in glob.glob(os.path.join(pasta, '*_oosC.json')):
            nome, _, sd = os.path.basename(f)[:-len('_oosC.json')].split('__')
            run[(nome, int(sd.replace('seed', '')))][met] = resumo(f)
    for (nome, sd), d in run.items():          # CAL: degrau com a escolha certificada mais barata
        deg = []
        for a in ESCADA:
            f = os.path.join(BASE, 'results_c_sp', a, f'{nome}__{a}__seed{sd}_oosC.json')
            if os.path.exists(f):
                r = resumo(f)
                if r['esc']:
                    deg.append(r)
        if deg:
            d['CAL'] = min(deg, key=lambda r: r['esc'][0])
    fr = os.path.join(BASE, 'pesquisa', 'resultados_caso_sp', 'refinamento', 'refinamento_CAL.json')
    for x in json.load(open(fr, encoding='utf-8')):
        k = (x['instance'], x['seed'])
        if k in run:
            run[k]['CAL+R'] = dict(custo_ok=np.nan, esc=(x['fin_f1'], x['fin_p_oos'], x['fin_q_oos'], x['fin_f3']))
    ks = sorted(k for k, d in run.items() if 'SIM100' in d)
    out = [f'SIM+C COM 100 CENÁRIOS NO CASO SP — {len(ks)} execuções; α = {ALPHA}; 1.000 cenários comuns']
    for m in ('SIM30', 'SIM100', 'BUF0.95+C', 'CAL', 'CAL+R'):
        v = [run[k][m]['esc'] for k in ks if m in run[k] and run[k][m]['esc']]
        if v:
            v = np.array(v)
            out.append(f'  {m:10s} escolha cumpre {100*np.mean(v[:,1] >= ALPHA):5.1f}%  custo {v[:,0].mean():8.1f}  '
                       f'perda {100*v[:,2].mean():.3f}%  CO2 {v[:,3].mean():.1f}  (n={len(v)})')
    out.append('\nPlano escolhido, pareado por instância (média das sementes): método contra SIM100; HL [IC 95%]; Holm nas 4 medidas')
    for m in ('SIM30', 'BUF0.95+C', 'CAL', 'CAL+R'):
        res = []
        for idx, nm, esc in ((0, 'custo (%)', 'rel'), (2, 'perda (p.p.)', 'abs'), (3, 'CO2 (%)', 'rel'), (1, 'P_oos (p.p.)', 'abs')):
            pi, ps = defaultdict(list), defaultdict(list)
            for k in ks:
                if m in run[k] and run[k][m]['esc'] and run[k]['SIM100']['esc']:
                    pi[k[0]].append(run[k][m]['esc'][idx]); ps[k[0]].append(run[k]['SIM100']['esc'][idx])
            I = sorted(pi)
            x = np.array([np.mean(pi[i]) for i in I]); y = np.array([np.mean(ps[i]) for i in I])
            dd = 100 * (x / y - 1) if esc == 'rel' else 100 * (x - y)
            hl, lo, hi = hodges_lehmann(dd, 0.95)
            p = wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
            res.append((nm, len(I), hl, lo, hi, p, int(np.sum(dd < 0)), int(np.sum(dd > 0))))
        out.append(f'  {m}:')
        for (nm, n, hl, lo, hi, p, mn, mx), pa in zip(res, holm([r[5] for r in res])):
            out.append(f'      {nm:14s} n={n:2d}  HL {hl:+7.3f} [{lo:+7.3f}, {hi:+7.3f}]  p={p:.3g}  p_Holm={pa:.3g}  ({mn} menores, {mx} maiores)')
    out.append('\nCusto da solução mais barata que cumpre, por execução (mediana), relativo ao SIM100:')
    for m in ('SIM30', 'BUF0.95+C', 'CAL'):
        d = [100 * (run[k][m]['custo_ok'] / run[k]['SIM100']['custo_ok'] - 1) for k in ks
             if m in run[k] and np.isfinite(run[k][m]['custo_ok']) and np.isfinite(run[k]['SIM100']['custo_ok'])]
        out.append(f'  {m:10s} mediana {np.median(d):+.2f}%  média {np.mean(d):+.2f}%  (n={len(d)})')
    txt = '\n'.join(out); print(txt)
    open(os.path.join(BASE, 'pesquisa', 'resultados_caso_sp', 'relatorio_simc100_sp.txt'), 'w', encoding='utf-8').write(txt + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rodar', action='store_true')
    ap.add_argument('--analisar', action='store_true')
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    if a.rodar:
        rodar(a.workers)
    if a.analisar:
        analisar()
