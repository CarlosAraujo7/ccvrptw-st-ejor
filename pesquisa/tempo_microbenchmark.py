#!/usr/bin/env python3
"""
Microbenchmark controlado de tempo por execução, para a Tabela 8 (14/10/2026, diário Entradas 90,
91 e 93). A execução v2 misturou 6 e 10 workers e uma máquina compartilhada, então o elapsed_s
das execuções não é comparável entre métodos (Lei Geral II.12: confusão por carga da máquina).

Desenho (fixado antes de rodar):
  - máquina livre (nenhuma outra execução do projeto ativa; conferido no início);
  - 18 instâncias de incerteza HIGH (6 famílias × n = 25, 50, 100), sementes 0 e 1;
  - 6 métodos da Tabela 8 (DET, DET+C, BUF0.90+C, BUF0.95+C, SIM+C, MOEA/D+C), a mesma
    configuração da execução principal (ALGO_CONFIG do experiment_runner);
  - 10 processos em paralelo, como na execução principal, com a ORDEM DAS TAREFAS SORTEADA
    (semente 2026), para que todos os métodos rodem misturados sob a mesma carga;
  - mede o tempo de relógio (elapsed_s do runner) e o tempo de CPU do processo;
  - saída numa pasta própria (results_c_tempo/), que não entra em nenhuma análise de qualidade.
216 execuções; ≈ 1 h com 10 processos.
Uso: python pesquisa/tempo_microbenchmark.py --rodar     (depois)  --tabela
"""
import argparse, glob, json, os, random, sys, time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(BASE, 'src'))
RES = os.path.join(BASE, 'results_c_tempo')
METODOS = ['C-DET', 'C-DETC', 'C-BUF90C', 'C-BUF95C', 'C-SIMC', 'C-MOEADC']
NOME = {'C-DET': r'\mDET', 'C-DETC': r'\mDETC', 'C-BUF90C': r'\mBUF{0.90}', 'C-BUF95C': r'\mBUF{0.95}',
        'C-SIMC': r'\mSIM', 'C-MOEADC': r'\mMOEAD'}
SEMENTES = (0, 1)
WORKERS = 10


def _worker(task):
    import experiment_runner as er
    t0 = time.process_time()
    r = er._worker(task)
    r['cpu_s'] = time.process_time() - t0
    json.dump({'cpu_s': r['cpu_s']}, open(os.path.join(task['out_dir'], task['result_key'] + '_cpu.json'), 'w'))
    return r


def rodar():
    import experiment_runner as er
    tarefas = []
    for f in sorted(glob.glob(os.path.join(BASE, 'instances', 'generated', 'CCVRPTW-ST_*_HIGH.json'))):
        nome = os.path.basename(f)[:-5]
        for alg in METODOS:
            cfg = er.ALGO_CONFIG[alg]; out = os.path.join(RES, alg); os.makedirs(out, exist_ok=True)
            for s in SEMENTES:
                key = f'{nome}__{alg}__seed{s}'
                if os.path.exists(os.path.join(out, key + '_cpu.json')):
                    continue
                tarefas.append({'inst_path': f, 'algo': alg, 'algo_label': alg, 'seed': s, 'pop_size': 100,
                                'n_gen': 200, 'mc_samples': cfg['mc_samples'], 'runner': cfg['runner'],
                                'out_dir': out, 'result_key': key, 'use_multizone': False, 'n_zones': 1,
                                'dedup': cfg.get('dedup', False), 'c_params': cfg.get('c_params')})
    random.Random(2026).shuffle(tarefas)
    print(f'{len(tarefas)} execuções, {WORKERS} processos, ordem sorteada', flush=True)
    with Pool(WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tarefas), 1):
            print(f"  [{i}/{len(tarefas)}] {r.get('key', '')} {r.get('status', '')} {r.get('elapsed', '')} s", flush=True)


def tabela():
    t = defaultdict(list); c = defaultdict(list)
    for f in glob.glob(os.path.join(RES, 'C-*', '*_metrics.json')):
        m = json.load(open(f, encoding='utf-8')); k = (m['algorithm'], m['size'])
        t[k].append(m['elapsed_s'])
        fc = f.replace('_metrics.json', '_cpu.json')
        if os.path.exists(fc):
            c[k].append(json.load(open(fc))['cpu_s'])
    linhas, rel = [], ['MICROBENCHMARK DE TEMPO (HIGH, 18 instâncias × 2 sementes, 10 processos, ordem sorteada)',
                       'método      n=25 relógio/CPU   n=50 relógio/CPU   n=100 relógio/CPU   (s, médias)']
    for a in METODOS:
        linhas.append(NOME[a] + ' & ' + ' & '.join(f'{np.mean(t[(a, n)]):.0f}' for n in (25, 50, 100)) + r' \\')
        rel.append(f'{a:10s} ' + '   '.join(f'{np.mean(t[(a, n)]):7.1f}/{np.mean(c[(a, n)]) if c[(a, n)] else float("nan"):7.1f}'
                                          for n in (25, 50, 100)))
    tex = (r'\begin{table}[htbp]' '\n' r'\centering' '\n'
           r'\caption{Mean wall-clock time per run (s; 200 generations), measured in a controlled benchmark: '
           r'the 18 high-uncertainty instances, 2 seeds, all methods interleaved in random order, 10 runs in parallel '
           r'on an otherwise idle 12-core desktop computer. \mCAL{} requires one run per rung of the ladder (four).}' '\n'
           r'\label{tab:time}' '\n' r'\small' '\n' r'\begin{tabular}{@{}l r r r@{}}' '\n' r'\toprule' '\n'
           r'Method & $n=25$ & $n=50$ & $n=100$ \\' '\n' r'\midrule' '\n' + '\n'.join(linhas) + '\n'
           r'\bottomrule' '\n' r'\end{tabular}' '\n' r'\end{table}' '\n')
    open(os.path.join(BASE, 'pesquisa', 'latex_ejor_c', 'tables', 'tab_tempo.tex'), 'w', encoding='utf-8').write(tex)
    os.makedirs(os.path.join(BASE, 'pesquisa', 'resultados_tempo'), exist_ok=True)
    open(os.path.join(BASE, 'pesquisa', 'resultados_tempo', 'relatorio_tempo.txt'), 'w', encoding='utf-8').write('\n'.join(rel) + '\n')
    print('\n'.join(rel)); print(tex)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rodar', action='store_true')
    ap.add_argument('--tabela', action='store_true')
    a = ap.parse_args()
    if a.rodar:
        rodar()
    if a.tabela:
        tabela()
