#!/usr/bin/env python3
"""
Teste de estresse do caso São Paulo: congestionamento de pico (14/10/2026, diário Entrada 94).

As instâncias usam o congestionamento MÉDIO de São Paulo em 2025 (tempo de fluxo livre × 1,585), mas
a maioria das entregas está programada para a manhã (08-11 h), quando o congestionamento é maior.
Valores da TomTom Traffic Index 2025, página de São Paulo, aba "City", consultada em 14/10/2026:
média 58,5%, pico da manhã 88,2%, pico da tarde 102,6%.
O teste executa o plano ESCOLHIDO de cada execução (o mais barato certificado), sem replanejar, nos
mesmos 1.000 cenários comuns da avaliação, com todos os tempos nominais multiplicados por
1,882 / 1,585 (pico da manhã o dia todo, um caso pessimista) e mede se ele ainda cumpre α.
Uso: python pesquisa/caso_sp/estresse_pico.py   (depois de rodar_caso_sp.py --analisar)
Saída: pesquisa/resultados_caso_sp/relatorio_estresse_pico.txt
"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(AQUI, '..', '..'))
sys.path.insert(0, os.path.join(BASE, 'src')); sys.path.insert(0, os.path.join(BASE, 'pesquisa')); sys.path.insert(0, AQUI)
import rodar_caso_sp   # noqa: E402
rodar_caso_sp.aplicar_matrizes()
import analisar_c as ac   # noqa: E402
import ccvrp_c as cc      # noqa: E402

ALPHA = 0.90
FATORES = {'média (58,5%)': 1.0, 'pico da manhã (88,2%)': 1.882 / 1.585}
ESCADA = ['C-BUF70C', 'C-BUF80C', 'C-BUF90C', 'C-BUF95C']
METODOS = ['C-DETC', 'C-BUF90C', 'C-BUF95C', 'CAL', 'C-SIMC']
RES = os.path.join(BASE, 'results_c_sp')


def escolhido(arq):
    S = json.load(open(arq, encoding='utf-8'))
    cert = [s for s in S if s['feasible'] and (s.get('p_cert') or 0) >= ALPHA]
    return min(cert, key=lambda s: s['f1_cost_BRL']) if cert else None


def main():
    planos = defaultdict(dict)   # (inst, seed) -> método -> solução escolhida
    for a in ['C-DETC', 'C-SIMC'] + ESCADA:
        for f in glob.glob(os.path.join(RES, a, '*_solutions.json')):
            nome, _, sd = os.path.basename(f)[:-len('_solutions.json')].split('__')
            e = escolhido(f)
            if e:
                planos[(nome, int(sd.replace('seed', '')))][a] = e
    for k, d in planos.items():
        deg = [d[a] for a in ESCADA if a in d]
        if deg:
            d['CAL'] = min(deg, key=lambda s: s['f1_cost_BRL'])
    res = defaultdict(list)       # (nível, fator, método) -> P_on
    for (nome, sd), d in sorted(planos.items()):
        inst, ttime, sc, _ = ac._ctx(nome, 1)
        nivel = nome.split('_')[-1]
        for rot, fat in FATORES.items():
            tt = [[v * fat for v in row] for row in ttime]
            for a in METODOS:
                if a not in d:
                    continue
                on = np.ones(ac.K, dtype=bool)
                for r, d0 in zip(d[a]['routes'], d[a]['partida_min']):
                    on &= cc.pontual_rota(r, inst, tt, sc['tt'], d0)
                res[(nivel, rot, a)].append(float(on.mean()))
    out = ['ESTRESSE DE PICO — caso SP v2: plano escolhido (mais barato certificado), sem replanejar; 1.000 cenários comuns',
           'colunas: % dos planos que cumprem α = 0,90 | P_on média | P_on mínima']
    for nivel in ('MEDIUM', 'HIGH'):
        out.append(f'\n{nivel}')
        for rot in FATORES:
            out.append(f'  tempos: {rot}')
            for a in METODOS:
                v = np.array(res[(nivel, rot, a)])
                if len(v):
                    out.append(f'    {a:9s} n={len(v):2d}  cumpre {100 * np.mean(v >= ALPHA):5.1f}%  P média {v.mean():.3f}  mín {v.min():.3f}')
    txt = '\n'.join(out); print(txt)
    os.makedirs(os.path.join(BASE, 'pesquisa', 'resultados_caso_sp'), exist_ok=True)
    open(os.path.join(BASE, 'pesquisa', 'resultados_caso_sp', 'relatorio_estresse_pico.txt'), 'w', encoding='utf-8').write(txt + '\n')


if __name__ == '__main__':
    main()
