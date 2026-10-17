#!/usr/bin/env python3
"""
Tabela dos tempos de viagem observados para o artigo (17/10/2026, diário Entradas 113 e 115): % de planos escolhidos que
cumprem α no benchmark (incerteza média e alta) e no caso SP, com a lognormal do artigo (M0); com a forma e o fator do dia
observados nos táxis de Nova York (M2), antes e depois de recertificar as frentes guardadas com cenários observados; com
a forma observada nos caminhões do DT-CARGO, antes e depois da recertificação; e com o CV observado nos táxis (M3).
Lê resultados/resumo_reavaliacao_observada.json, resumo_recertificacao_observada.json e resumo_caminhoes.json.
Saída: latex_ejor_c/tables/tab_observado.tex
"""
import json, os

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(AQUI, '..', '..'))
RES = os.path.join(AQUI, 'resultados')
R = json.load(open(os.path.join(RES, 'resumo_reavaliacao_observada.json'), encoding='utf-8'))
C = json.load(open(os.path.join(RES, 'resumo_recertificacao_observada.json'), encoding='utf-8'))
T = json.load(open(os.path.join(RES, 'resumo_caminhoes.json'), encoding='utf-8'))
NOME = {'C-DETC': r'\mDETC{}', 'C-BUF90C': r'\mBUF{0.90}', 'C-BUF95C': r'\mBUF{0.95}', 'CAL': r'\mCAL{}',
        'C-SIMC': r'\mSIM{}', 'ACAL': r'\mACAL{}'}
CONJ = [('benchmark (média e alta)', 'Benchmark'), ('SP', r'S\~ao Paulo')]


def main():
    L = [r'\begin{table}[htbp]', r'\centering',
         r'\caption{Chosen plans under observed travel-time variability: percentage of runs whose chosen plan meets '
         r'$\alpha=0.90$ on 1{,}000 scenarios. M0: lognormal model of the paper. Taxi: instance CV with the shape and day '
         r'factor observed in New York taxi trips (M2). Truck: instance CV with the shape observed in the DT-CARGO truck trips. '
         r'Re-cert.: the stored front re-certified on 500 independent scenarios of the same observed model (runs without a '
         r're-certified plan are excluded). M3: CV observed in the taxi trips for the arc duration (20\% to 29\%). '
         r'Benchmark: 360 runs with medium or high uncertainty; S\~ao Paulo: 160 runs.}',
         r'\label{tab:observed}', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
         r'\begin{tabular}{@{}l rrrrrr rrrrrr@{}}', r'\toprule',
         r' & \multicolumn{6}{c}{Benchmark} & \multicolumn{6}{c}{S\~ao Paulo} \\',
         r'\cmidrule(lr){2-7}\cmidrule(l){8-13}',
         r' & & \multicolumn{2}{c}{Taxi} & \multicolumn{2}{c}{Truck} & & & \multicolumn{2}{c}{Taxi} & \multicolumn{2}{c}{Truck} & \\',
         r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){9-10}\cmidrule(lr){11-12}',
         r'Method & M0 & orig. & re-cert. & orig. & re-cert. & M3 & M0 & orig. & re-cert. & orig. & re-cert. & M3 \\', r'\midrule']
    for a in ('C-DETC', 'C-BUF90C', 'C-BUF95C', 'CAL', 'C-SIMC', 'ACAL'):
        cel = []
        for k, _ in CONJ:
            r = R[k][a]; c = C[k][a]['M2']; t = T[k][a]
            cel += [f"{r['M0']['cumpre']:.0f}", f"{r['M2']['cumpre']:.0f}", f"{c['cumpre']:.0f}",
                    f"{t['original']:.0f}", f"{t['recertificada']:.0f}", f"{r['M3']['cumpre']:.0f}"]
        L.append(NOME[a] + ' & ' + ' & '.join(cel) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    arq = os.path.join(BASE, 'pesquisa', 'latex_ejor_c', 'tables', 'tab_observado.tex')
    open(arq, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('tabela:', arq)


if __name__ == '__main__':
    main()
