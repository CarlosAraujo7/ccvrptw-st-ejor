#!/usr/bin/env python3
"""
Planejar com o tempo certo: caso São Paulo no pico da manhã (15/10/2026, análise de fraquezas A3).

As instâncias SPP1/SPP2 (instances/caso_sp_pico) têm as MESMAS lojas, janelas, serviços e demandas de
SP1/SP2, com os tempos do pico da manhã (congestionamento 88,2% em vez de 58,5%; TomTom 2025).
Para cada instância e semente, no cenário de pico (1.000 cenários comuns da instância SPP):
  - plano escolhido (mais barato certificado) planejado com o congestionamento MÉDIO (results_c_sp),
    executado sem replanejar no pico;
  - plano escolhido planejado com o congestionamento do PICO (results_c_sp_pico).
Mede: % de planos que cumprem α, P média, custo, perda de qualidade e CO2 (tempos de pico).
Uso: python pesquisa/caso_sp/comparar_pico.py   (depois de rodar e avaliar results_c_sp_pico)
Saída: pesquisa/resultados_caso_sp_pico/relatorio_comparar_pico.txt
"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np

os.environ['CCVRP_SP_VARIANTE'] = 'pico'
AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(AQUI, '..', '..'))
sys.path.insert(0, os.path.join(BASE, 'src')); sys.path.insert(0, os.path.join(BASE, 'pesquisa')); sys.path.insert(0, AQUI)
import rodar_caso_sp   # noqa: E402
rodar_caso_sp.aplicar_matrizes()       # INST_DIR da avaliação = instances/caso_sp_pico
import analisar_c as ac            # noqa: E402
import ccvrp_c as cc               # noqa: E402
import thermal_route as tr         # noqa: E402
import refinamento_simulacao as rs  # noqa: E402

ALPHA = 0.90
ESCADA = ['C-BUF70C', 'C-BUF80C', 'C-BUF90C', 'C-BUF95C']
METODOS = ['C-DETC', 'C-BUF95C', 'CAL', 'C-SIMC']


def escolhidos(res_dir):
    P = defaultdict(dict)
    for a in ['C-DETC', 'C-SIMC'] + ESCADA:
        for f in glob.glob(os.path.join(BASE, res_dir, a, '*_solutions.json')):
            nome, _, sd = os.path.basename(f)[:-len('_solutions.json')].split('__')
            S = json.load(open(f, encoding='utf-8'))
            cert = [s for s in S if s['feasible'] and (s.get('p_cert') or 0) >= ALPHA]
            if cert:
                P[(nome, int(sd.replace('seed', '')))][a] = min(cert, key=lambda s: s['f1_cost_BRL'])
    for d in P.values():
        deg = [d[a] for a in ESCADA if a in d]
        if deg:
            d['CAL'] = min(deg, key=lambda s: s['f1_cost_BRL'])
    return P


def avaliar(sol, nome_pico):
    """Executa rotas e partidas na instância de pico: P_on, perda esperada (1.000 cenários) e custo/CO2 nominais."""
    inst, ttime, sc, _ = ac._ctx(nome_pico, 1)
    ctx = cc.Contexto(inst, 0, decoder='split', p_ls=0.0)
    R, D = sol['routes'], sol['partida_min']
    p, q = rs.avaliar_oos(R, D, nome_pico, ctx)
    f1, _, f3 = rs.Plano(ctx, np.ones((1, len(inst['customers']) + 1))).objetivos(R, D)
    return dict(p=p, q=q, f1=f1, f3=f3)


def main():
    med = escolhidos('results_c_sp'); pic = escolhidos('results_c_sp_pico')
    res = defaultdict(list)          # (nível, método, origem) -> dicts
    for (nome_p, sd), d in sorted(pic.items()):
        nome_m = nome_p.replace('_SPP', '_SP')
        nivel = nome_p.split('_')[-1]
        for a in METODOS:
            if a in d and a in med.get((nome_m, sd), {}):
                res[(nivel, a, 'médio')].append(avaliar(med[(nome_m, sd)][a], nome_p))
                res[(nivel, a, 'pico')].append(avaliar(d[a], nome_p))
    out = ['PLANEJAR COM O TEMPO CERTO — plano escolhido executado no pico da manhã (instâncias SPP; 1.000 cenários comuns)',
           'colunas: % que cumpre α | P média | custo médio | perda média | CO2 médio; Δ = pico − médio (medianas pareadas)']
    for nivel in ('MEDIUM', 'HIGH'):
        out.append(f'\n{nivel}')
        for a in METODOS:
            A, B = res[(nivel, a, 'médio')], res[(nivel, a, 'pico')]
            if not A:
                continue
            for rot, L in (('planejado no médio', A), ('planejado no pico ', B)):
                out.append(f'  {a:9s} {rot}  n={len(L):2d}  cumpre {100*np.mean([x["p"] >= ALPHA for x in L]):5.1f}%  '
                           f'P {np.mean([x["p"] for x in L]):.3f}  custo {np.mean([x["f1"] for x in L]):8.1f}  '
                           f'perda {100*np.mean([x["q"] for x in L]):.3f}%  CO2 {np.mean([x["f3"] for x in L]):.1f}')
            dc = [100 * (b['f1'] / a_['f1'] - 1) for a_, b in zip(A, B)]
            dq = [100 * (b['q'] - a_['q']) for a_, b in zip(A, B)]
            de = [100 * (b['f3'] / a_['f3'] - 1) for a_, b in zip(A, B)]
            out.append(f'  {a:9s} Δ pico − médio: custo {np.median(dc):+.2f}%  perda {np.median(dq):+.3f} p.p.  CO2 {np.median(de):+.2f}%')
    txt = '\n'.join(out); print(txt)
    os.makedirs(os.path.join(BASE, 'pesquisa', 'resultados_caso_sp_pico'), exist_ok=True)
    open(os.path.join(BASE, 'pesquisa', 'resultados_caso_sp_pico', 'relatorio_comparar_pico.txt'), 'w', encoding='utf-8').write(txt + '\n')


if __name__ == '__main__':
    main()
