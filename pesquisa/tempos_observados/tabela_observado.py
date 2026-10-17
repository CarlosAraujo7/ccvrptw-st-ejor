#!/usr/bin/env python3
"""
Tabela dos tempos de viagem observados para o artigo (17/10/2026, diário Entrada 113): % de planos escolhidos que
cumprem α no benchmark (incerteza média e alta) e no caso SP, com a lognormal do artigo (M0), com a forma e o fator do
dia observados (M2), com M2 depois de recertificar as frentes guardadas com cenários observados, e com o CV observado (M3).
Lê resultados/resumo_reavaliacao_observada.json e resultados/resumo_recertificacao_observada.json.
Saída: latex_ejor_c/tables/tab_observado.tex
"""
import json, os

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(AQUI, '..', '..'))
R = json.load(open(os.path.join(AQUI, 'resultados', 'resumo_reavaliacao_observada.json'), encoding='utf-8'))
C = json.load(open(os.path.join(AQUI, 'resultados', 'resumo_recertificacao_observada.json'), encoding='utf-8'))
V = json.load(open(os.path.join(AQUI, 'resultados', 'variabilidade_tlc.json'), encoding='utf-8'))
NOME = {'C-DETC': r'\mDETC{}', 'C-BUF90C': r'\mBUF{0.90}', 'C-BUF95C': r'\mBUF{0.95}', 'CAL': r'\mCAL{}',
        'C-SIMC': r'\mSIM{}', 'ACAL': r'\mACAL{}'}
CONJ = [('benchmark (média e alta)', 'Benchmark'), ('SP', r'S\~ao Paulo')]


def main():
    L = [r'\begin{table}[htbp]', r'\centering',
         r'\caption{Chosen plans under observed travel-time variability (taxi trips in New York City, weekdays of 2024): '
         r'percentage of runs whose chosen plan meets $\alpha=0.90$ on 1{,}000 scenarios with the lognormal model of the paper '
         r'(M0); with the instance CV but the observed shape and day factor (M2); with M2 after re-certifying the stored fronts '
         r'on 500 independent observed scenarios (M2, re-cert.); and with the observed CV of the arc duration (M3, '
         f"{100 * 0.20:.0f}\\% to 29\\%). Benchmark: 360 runs with medium or high uncertainty; S\\~ao Paulo: 160 runs; runs without a "
         r"re-certified plan (at most 3\%) are excluded from the re-certified column." + '}',
         r'\label{tab:observed}', r'\small', r'\begin{tabular}{@{}l rrrr rrrr@{}}', r'\toprule',
         r' & \multicolumn{4}{c}{Benchmark} & \multicolumn{4}{c}{S\~ao Paulo} \\',
         r'\cmidrule(lr){2-5}\cmidrule(l){6-9}',
         r'Method & M0 & M2 & \shortstack{M2,\\re-cert.} & M3 & M0 & M2 & \shortstack{M2,\\re-cert.} & M3 \\', r'\midrule']
    for a in ('C-DETC', 'C-BUF90C', 'C-BUF95C', 'CAL', 'C-SIMC', 'ACAL'):
        cel = []
        for k, _ in CONJ:
            r = R[k][a]; c = C[k][a]['M2']
            cel += [f"{r['M0']['cumpre']:.0f}", f"{r['M2']['cumpre']:.0f}", f"{c['cumpre']:.0f}", f"{r['M3']['cumpre']:.0f}"]
        L.append(NOME[a] + ' & ' + ' & '.join(cel) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    arq = os.path.join(BASE, 'pesquisa', 'latex_ejor_c', 'tables', 'tab_observado.tex')
    open(arq, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('tabela:', arq)


if __name__ == '__main__':
    main()
