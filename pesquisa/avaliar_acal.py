#!/usr/bin/env python3
"""
Avaliação da calibração dentro de uma busca (ACAL), 17/10/2026, diário Entrada 109.

  --fase dev    por instância de desenvolvimento: média (sementes 10-12) do plano escolhido (certificada mais barata)
                de cada variante contra a média (sementes 0-9) de SIM+C, BUF0.90+C e CAL nas execuções existentes
                (comparação por instância, não pareada por semente);
  --fase teste  pareado por instância e semente (sementes 0-9) contra SIM+C (1 busca), SIM2, SIM4, BUF0.90+C, BUF0.95+C e
                CAL (4 buscas), com Hodges-Lehmann e Wilcoxon com Holm, nos quatro conjuntos de teste.
Uso: python pesquisa/avaliar_acal.py --fase dev | --fase teste --variante A
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibracao_preditiva_base as b

BASE = b.BASE; ALPHA = b.ALPHA
OUT = os.path.join(BASE, 'pesquisa', 'resultados_acal'); os.makedirs(OUT, exist_ok=True)
ORIG = {'bench': 'results_c', 'SP': 'results_c_sp', 'SPT': 'results_c_sp_td', 'ATH': 'results_c_atenas'}
ACAL = {'bench': 'results_c_acal_bench', 'SP': 'results_c_acal_sp', 'SPT': 'results_c_acal_sp_td', 'ATH': 'results_c_acal_atenas'}
NOMES = {'bench': 'benchmark (v2)', 'SP': 'SP', 'SPT': 'SP com horário', 'ATH': 'Atenas'}


def escolhas(pasta, alg):
    E = {}
    for f in glob.glob(os.path.join(BASE, pasta, alg, '*_oosC.json')):
        d = json.load(open(f, encoding='utf-8')); m = d['meta']
        cert = [s for s in d['solutions'] if s['feasible'] and (s.get('p_cert') or 0) >= ALPHA]
        E[(m['instance'], m['seed'])] = min(cert, key=lambda s: s['f1']) if cert else None
    return E


def dev(log):
    for c in ('bench', 'SP', 'ATH'):
        R = b.carregar(ORIG[c])
        ref = {}
        for nome, fn in (('SIM+C', b.regra_sim(1)), ('BUF0.90+C', b.regra_degraus(['C-BUF90C'])),
                         ('CAL', b.regra_degraus(list(b.DEG)))):
            sem = sorted({s for _, s in R})
            ref[nome] = {k: fn(R, k[0], k[1], sem)[0] for k in R}
        log(f'\n=== desenvolvimento — {NOMES[c]} ===')
        for v in ('A', 'B'):
            E = escolhas(ACAL[c] + '_dev', f'C-ACALC-{v}')
            if not E:
                continue
            insts = sorted({i for i, _ in E})
            linha = [f'  C-ACALC-{v}: cumpre {100 * np.mean([e["p_on"] >= ALPHA for e in E.values() if e]):.1f}% '
                     f'(sem escolha {sum(1 for e in E.values() if e is None)})']
            for nome, Rf in ref.items():
                dif = []
                for i in insts:
                    a = [e['f1'] for (ii, _), e in E.items() if ii == i and e]
                    r = [e['f1'] for (ii, _), e in Rf.items() if ii == i and e]
                    if a and r:
                        dif.append(100 * (np.mean(a) / np.mean(r) - 1))
                linha.append(f'custo vs {nome}: mediana {np.median(dif):+.2f}%')
            log('; '.join(linha))
            qf = []
            for f in glob.glob(os.path.join(BASE, ACAL[c] + '_dev', f'C-ACALC-{v}', '*_metrics.json')):
                qh = json.load(open(f, encoding='utf-8')).get('q_hist') or []
                if qh:
                    qf.append(qh[-1][1])
            log(f'      nível final: mediana {np.median(qf):.3f}, P10 {np.percentile(qf, 10):.3f}, P90 {np.percentile(qf, 90):.3f}')


def teste(variante, log):
    alg = f'C-ACALC-{variante}'
    for c in ('bench', 'SP', 'SPT', 'ATH'):
        R = b.carregar(ORIG[c])
        E = escolhas(ACAL[c], alg)
        for k in R:
            R[k][alg] = E.get(k)
        regras = {'SIM+C': b.regra_sim(1), 'SIM2 (melhor de 2)': b.regra_sim(2), 'SIM4 (melhor de 4)': b.regra_sim(4),
                  'BUF0.90+C': b.regra_degraus(['C-BUF90C']), 'BUF0.95+C': b.regra_degraus(['C-BUF95C']),
                  'CAL (4 buscas)': b.regra_degraus(list(b.DEG)), f'ACAL (1 busca)': b.regra_degraus([alg])}
        b.avaliar(NOMES[c], R, regras, log=log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fase', choices=['dev', 'teste'], required=True)
    ap.add_argument('--variante', default='A')
    a = ap.parse_args()
    rel = open(os.path.join(OUT, f'relatorio_acal_{a.fase}.txt'), 'w', encoding='utf-8')

    def log(t=''):
        print(t, flush=True); rel.write(t + '\n')
    log(f'CALIBRAÇÃO DENTRO DE UMA BUSCA (ACAL) — fase {a.fase}; escolha = certificada mais barata; 1.000 cenários comuns')
    dev(log) if a.fase == 'dev' else teste(a.variante, log)
    rel.close()


if __name__ == '__main__':
    main()
