#!/usr/bin/env python3
"""
Calibração preditiva — etapa 1: linhas de base offline (17/10/2026, diário Entrada 108).

Todas as buscas da escada (DET+C, BUF0.70/0.80/0.90/0.95+C) e do SIM+C já foram rodadas em todos os conjuntos, então
qualquer regra que escolha QUAIS degraus rodar pode ser avaliada sem execuções novas: a busca de um degrau é a execução
existente desse degrau, e o plano escolhido é a solução certificada mais barata entre as buscas rodadas.
Orçamento = número de buscas. Medidas do plano escolhido nos 1.000 cenários comuns: cumpre α, custo, perda, CO2.
Comparações pareadas por instância (média das sementes) com Hodges-Lehmann e Wilcoxon.

Conjuntos: v1 (desenvolvimento, results_c_v1_cindom) e, para teste, v2 (results_c), SP, SP com horário, Atenas.
Uso: python pesquisa/calibracao_preditiva_base.py
"""
import glob, json, os
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
ALPHA = 0.90
DEG = {'C-BUF70C': 0.70, 'C-BUF80C': 0.80, 'C-BUF90C': 0.90, 'C-BUF95C': 0.95}
CONJ = [('v1 (desenvolvimento)', 'results_c_v1_cindom'), ('v2', 'results_c'), ('SP', 'results_c_sp'),
        ('SP com horário', 'results_c_sp_td'), ('Atenas', 'results_c_atenas')]


def carregar(res):
    """{(inst, seed): {método: escolha certificada mais barata (linha do oosC) ou None}}"""
    R = defaultdict(dict)
    for a in ['C-DETC', 'C-SIMC'] + list(DEG):
        for f in glob.glob(os.path.join(BASE, res, a, '*_oosC.json')):
            d = json.load(open(f, encoding='utf-8')); m = d['meta']
            cert = [s for s in d['solutions'] if s['feasible'] and (s.get('p_cert') or 0) >= ALPHA]
            R[(m['instance'], m['seed'])][a] = min(cert, key=lambda s: s['f1']) if cert else None
    return R


def melhor(cands):
    c = [x for x in cands if x is not None]
    return min(c, key=lambda s: s['f1']) if c else None


def hl(d):
    d = np.asarray(d, float)
    return float(np.median([(d[i] + d[j]) / 2 for i in range(len(d)) for j in range(i, len(d))]))


def avaliar(nome, R, regras, ref='SIM+C', log=print):
    insts = sorted({i for i, _ in R}); sementes = sorted({s for _, s in R})
    esc = {r: {} for r in regras}
    for (i, s), d in R.items():
        for r, fn in regras.items():
            esc[r][(i, s)] = fn(R, i, s, sementes)
    log(f'\n=== {nome}: {len(insts)} instâncias, {len(R)} execuções ===')
    log(f'  {"regra":24s} {"buscas":>6s} {"cumpre":>7s} {"sem escolha":>11s} | vs {ref}: custo HL   perda HL   CO2 HL  (p custo)')
    for r in regras:
        E = esc[r]
        bus = np.mean([e[1] for e in E.values() if e is not None])
        ok = [e[0]['p_on'] >= ALPHA for e in E.values() if e is not None and e[0] is not None]
        sem = sum(1 for e in E.values() if e is None or e[0] is None)
        dif = {'f1': [], 'q_oos': [], 'f3': []}
        for i in insts:
            par = [(E[(i, s)][0], esc[ref][(i, s)][0]) for s in sementes
                   if (i, s) in E and E[(i, s)] and E[(i, s)][0] and esc[ref].get((i, s)) and esc[ref][(i, s)][0]]
            if not par:
                continue
            for k in dif:
                a = np.mean([x[k] for x, _ in par]); b = np.mean([y[k] for _, y in par])
                dif[k].append(100 * (a - b) if k == 'q_oos' else 100 * (a / b - 1))
        p = wilcoxon(dif['f1']).pvalue if any(abs(v) > 1e-12 for v in dif['f1']) else 1.0
        log(f'  {r:24s} {bus:6.2f} {100 * np.mean(ok):6.1f}% {sem:11d} | {hl(dif["f1"]):+8.3f}% {hl(dif["q_oos"]):+8.3f}pp '
            f'{hl(dif["f3"]):+7.3f}%  ({p:.2g})')
    return esc


def regra_degraus(ds):
    return lambda R, i, s, sem: (melhor([R[(i, s)].get(a) for a in ds]), len(ds))


def regra_sim(k):
    def fn(R, i, s, sem):
        ss = [sem[(sem.index(s) + j) % len(sem)] for j in range(k)]
        return melhor([R.get((i, t), {}).get('C-SIMC') for t in ss]), k
    return fn


REGRAS = {
    'SIM+C': regra_sim(1), 'SIM2 (melhor de 2)': regra_sim(2), 'SIM4 (melhor de 4)': regra_sim(4),
    'BUF0.90+C': regra_degraus(['C-BUF90C']), 'BUF0.95+C': regra_degraus(['C-BUF95C']),
    'CAL (4 degraus)': regra_degraus(list(DEG)),
    'CAL {0.70, 0.95}': regra_degraus(['C-BUF70C', 'C-BUF95C']),
    'CAL {0.80, 0.95}': regra_degraus(['C-BUF80C', 'C-BUF95C']),
    'CAL {0.70, 0.90}': regra_degraus(['C-BUF70C', 'C-BUF90C']),
    'CAL {DET+C, 0.95}': regra_degraus(['C-DETC', 'C-BUF95C']),
    'oráculo 0.95 + 1 degrau': None,
}


def oraculo(R, i, s, sem):
    """Teto de uma regra de duas buscas com piloto em 0,95: o melhor degrau extra escolhido sabendo o resultado
    (usa a certificação, não a avaliação fora da amostra)."""
    d = R[(i, s)]
    return melhor([d.get('C-BUF95C')] + [d.get(a) for a in list(DEG) + ['C-DETC']]), 2


REGRAS['oráculo 0.95 + 1 degrau'] = oraculo


def main():
    out = os.path.join(BASE, 'pesquisa', 'resultados_calibracao_preditiva'); os.makedirs(out, exist_ok=True)
    rel = open(os.path.join(out, 'relatorio_linhas_de_base.txt'), 'w', encoding='utf-8')

    def log(t=''):
        print(t, flush=True); rel.write(t + '\n')
    log('CALIBRAÇÃO PREDITIVA — linhas de base offline (escolha = certificada mais barata entre as buscas rodadas)')
    for nome, res in CONJ:
        avaliar(nome, carregar(res), REGRAS, log=log)
    rel.close()


if __name__ == '__main__':
    main()
