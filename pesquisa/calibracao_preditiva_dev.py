#!/usr/bin/env python3
"""
Calibração preditiva — exploração da regra SÓ no conjunto de desenvolvimento (v1), 17/10/2026, diário Entrada 108.
Regra "padrão + desvio": roda o nível padrão D, a não ser que o piloto mostre que outro nível L tem plano certificado
mais barato por uma margem relativa θ: C̃(L) < (1 − θ)·C̃(D). Se o padrão não certifica no piloto, usa o mais barato.
Lê as previsões gravadas por calibracao_preditiva.py (C̃ por nível) e avalia offline como a etapa 2.
Uso: python pesquisa/calibracao_preditiva_dev.py <sufixo das previsões> [conjunto]
"""
import json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibracao_preditiva_base as b

OUT = os.path.join(b.BASE, 'pesquisa', 'resultados_calibracao_preditiva')


def regra(P, padrao, theta):
    def fn(R, i, s, sem):
        pil = P[(i, s)]['piloto']
        cd = pil.get(padrao, {}).get('c', math.inf)
        cands = [(v['c'], m) for m, v in pil.items() if math.isfinite(v['c'])]
        if not cands:
            esc = 'C-BUF95C'
        elif not math.isfinite(cd):
            esc = min(cands)[1]
        else:
            melhores = [(c, m) for c, m in cands if c < (1 - theta) * cd]
            esc = min(melhores)[1] if melhores else padrao
        return R[(i, s)].get(esc), 1
    return fn


def main():
    suf = sys.argv[1] if len(sys.argv) > 1 else '_n100_semnom'
    conj = sys.argv[2] if len(sys.argv) > 2 else 'v1'
    res_dir = {'v1': 'results_c_v1_cindom', 'v2': 'results_c', 'SP': 'results_c_sp', 'SPT': 'results_c_sp_td',
               'ATH': 'results_c_atenas'}[conj]
    P = {(p['instance'], p['seed']): p for p in json.load(open(os.path.join(OUT, f'previsoes_{conj}{suf}.json'), encoding='utf-8'))}
    R = b.carregar(res_dir)
    regras = {'SIM+C': b.regra_sim(1), 'BUF0.90+C': b.regra_degraus(['C-BUF90C']), 'BUF0.95+C': b.regra_degraus(['C-BUF95C'])}
    for padrao in ('C-BUF90C', 'C-BUF95C'):
        for theta in (0.0, 0.005, 0.01, 0.02, 0.03):
            regras[f'pad {padrao[-4:-1]} θ={100 * theta:.1f}%'] = regra(P, padrao, theta)
    b.avaliar(f'{conj} — exploração da regra (previsões {suf})', R, regras)


if __name__ == '__main__':
    main()
