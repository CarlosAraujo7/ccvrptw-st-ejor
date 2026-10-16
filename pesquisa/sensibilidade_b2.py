#!/usr/bin/env python3
"""
Sensibilidade do B2 aos parâmetros estimados (10/10/2026, diário Entrada 87; mesmo
desenho da Entrada 44): GAMMA_ZONE_COLD (potência extra por °C de zona mais fria,
0,0135) e, juntos, ZONE_LOADING/UNLOADING_COST_KM_MULTIPLE (2,60 / 2,05), cada um em
metade e dobro, sobre a simheurística certificada (C-SIMC), Z ∈ {2, 3}, 18 instâncias
com n = 50, sementes 0–2.
Para cada configuração (e o valor-base, C-SIMC em results_c_b2/), compara com a base
de compartimento único com carga mista (C-SIMC-MIX, mesmas instâncias e sementes) as
melhores soluções que cumprem α = 0,90 fora da amostra: Δ perda de qualidade (p.p.),
Δ custo (%), Δ CO2 (%), medianas. Descritivo, sem teste de significância (Entrada 44):
a pergunta é se a DIREÇÃO do resultado se mantém (2 zonas melhoram a qualidade com custo
pequeno; 3 zonas não compensam), não se metade e dobro diferem entre si.
Entradas: *_oosC.json de results_c_b2/ e results_c_b2_sens/ (analisar_c.py).
Saída: pesquisa/resultados_c_b2_sens/relatorio_sensibilidade_b2.txt e .json
"""
import glob, json, os
import numpy as np

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
B2 = os.path.join(BASE, 'results_c_b2'); SENS = os.path.join(BASE, 'results_c_b2_sens')
OUT = os.path.join(BASE, 'pesquisa', 'resultados_c_b2_sens')
CONFIGS = [('valor-base', B2, 'C-SIMC'), ('γ × 0,5', SENS, 'C-SIMC-G05'), ('γ × 2', SENS, 'C-SIMC-G2'),
           ('m_L, m_U × 0,5', SENS, 'C-SIMC-KM05'), ('m_L, m_U × 2', SENS, 'C-SIMC-KM2')]
SEEDS = (0, 1, 2); ALPHA = 0.90


def melhores(arq):
    d = json.load(open(arq, encoding='utf-8'))
    S = [s for s in d['solutions'] if s['feasible'] and s['p_on'] >= ALPHA]
    if not S:
        return None
    return min(s['q_oos'] for s in S), min(s['f1'] for s in S), min(s['f3'] for s in S)


def main():
    base = {}
    for f in glob.glob(os.path.join(B2, 'C-SIMC-MIX', '*_050_*_oosC.json')):
        m = json.load(open(f, encoding='utf-8'))['meta']
        if m['seed'] in SEEDS:
            base[(m['instance'], m['seed'])] = melhores(f)
    os.makedirs(OUT, exist_ok=True)
    linhas = ['SENSIBILIDADE DO B2 (SIM+C, n=50, sementes 0–2; contra compartimento único com carga mista)',
              'configuração          Z   n   Δ qualidade (p.p.)   Δ custo (%)   Δ CO2 (%)   [medianas]']
    res = {}
    for rot, pasta, alg in CONFIGS:
        for z in (2, 3):
            dq, dc, de = [], [], []
            for f in glob.glob(os.path.join(pasta, f'{alg}_Z{z}', '*_050_*_oosC.json')):
                m = json.load(open(f, encoding='utf-8'))['meta']; k = (m['instance'], m['seed'])
                if m['seed'] not in SEEDS or not base.get(k):
                    continue
                b = melhores(f)
                if not b:
                    continue
                q1, c1, e1 = base[k]
                dq.append(100 * (b[0] - q1)); dc.append(100 * (b[1] / c1 - 1)); de.append(100 * (b[2] / e1 - 1))
            if not dq:
                continue
            res[f'{alg}_Z{z}'] = dict(config=rot, z=z, n=len(dq), dq=float(np.median(dq)), dc=float(np.median(dc)),
                                      de=float(np.median(de)))
            linhas.append(f'  {rot:18s}  {z}  {len(dq):3d}   {np.median(dq):+8.2f}            {np.median(dc):+7.2f}      {np.median(de):+7.2f}')
    txt = '\n'.join(linhas); print(txt)
    open(os.path.join(OUT, 'relatorio_sensibilidade_b2.txt'), 'w', encoding='utf-8').write(txt + '\n')
    json.dump(res, open(os.path.join(OUT, 'sensibilidade_b2.json'), 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()
