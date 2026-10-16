#!/usr/bin/env python3
"""
Visão do decisor (08/10/2026, diário Entrada 81). As tabelas principais medem o CONTEÚDO
da frente (a solução mais barata que cumpre P_on >= α fora da amostra). Um decisor não
conhece P_on fora da amostra: escolhe a solução mais barata APROVADA NA CERTIFICAÇÃO
(p_cert >= α). Este script mede, para os métodos certificados, (a) a fração de execuções
em que essa escolha cumpre fora da amostra, (b) a P_on média dela e (c) a diferença de
custo para a mais barata que cumpre fora da amostra. CAL: escolha entre os 4 degraus pela
certificação, como no Algoritmo 1. Nada é reotimizado (usa *_oosC.json).
Saída: pesquisa/resultados_robustez/relatorio_decisor.txt
"""
import glob, json, os
from collections import defaultdict
import numpy as np

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
RC = os.path.join(BASE, 'results_c')
OUT = os.path.join(BASE, 'pesquisa', 'resultados_robustez')
CONF = {0.90: ['C-DETC', 'C-BUF90C', 'C-BUF95C', 'CAL', 'C-SIMC', 'C-MOEADC'],
        0.95: ['C-BUF95C95', 'CAL95', 'C-SIMC95']}
MARGEM = {a: 1.645 * np.sqrt(a * (1 - a) / 500) for a in (0.90, 0.95)}
ESCADA = {'CAL': ['C-BUF70C', 'C-BUF80C', 'C-BUF90C', 'C-BUF95C'],
          'CAL95': ['C-BUF90C95', 'C-BUF95C95', 'C-BUF99C95']}


def carregar(alg, alpha):
    d = {}
    for f in glob.glob(os.path.join(RC, alg, '*_oosC.json')):
        r = json.load(open(f, encoding='utf-8')); m = r['meta']
        S = [s for s in r['solutions'] if s['feasible']]
        cert = [s for s in S if (s.get('p_cert') or 0) >= alpha]
        pick = min(cert, key=lambda s: s['f1']) if cert else None
        # variante com limite inferior de confiança unilateral de 95%: p_cert >= α + 1,645·√(α(1−α)/500)
        lcb = [s for s in cert if s['p_cert'] >= alpha + MARGEM[alpha]]
        pick_lcb = min(lcb, key=lambda s: s['f1']) if lcb else None
        d[(m['instance'], m['seed'])] = (pick, pick_lcb)
    return d


RES = {}


def main():
    os.makedirs(OUT, exist_ok=True)
    rel = open(os.path.join(OUT, 'relatorio_decisor.txt'), 'w', encoding='utf-8')

    def log(t=''):
        print(t, flush=True); rel.write(t + '\n')

    log('ESCOLHA DO DECISOR: solução mais barata aprovada na certificação (500 cenários), avaliada em 1.000 cenários novos')
    log('colunas: execuções | % com escolha | % escolha cumpre fora | P_on das escolhas que falham (média, mín) | variante LCB 95%: % com escolha, % cumpre, Δcusto% vs escolha')
    for alpha, metodos in CONF.items():
        log(f'\nα = {alpha:.2f}')
        for nivel in ('LOW', 'MEDIUM', 'HIGH', 'TODOS'):
            log(f'  {nivel}')
            for a in metodos:
                if a in ESCADA:
                    ds = [carregar(x, alpha) for x in ESCADA[a]]
                    d = {}
                    for k in ds[-1]:
                        if all(k in di for di in ds):
                            c0 = [di[k][0] for di in ds if di[k][0] is not None]
                            c1 = [di[k][1] for di in ds if di[k][1] is not None]
                            d[k] = (min(c0, key=lambda s: s['f1']) if c0 else None,
                                    min(c1, key=lambda s: s['f1']) if c1 else None)
                else:
                    d = carregar(a, alpha)
                ks = [k for k in d if nivel == 'TODOS' or k[0].endswith(nivel)]
                picks = [d[k] for k in ks if d[k][0] is not None]
                if not ks:
                    continue
                cumpre = [p['p_on'] >= alpha for p, _ in picks]
                falha = [p['p_on'] for p, _ in picks if p['p_on'] < alpha]
                lcb = [d[k][1] for k in ks if d[k][1] is not None]
                c_lcb = [p['p_on'] >= alpha for p in lcb]
                dc = [100 * (q['f1'] / p['f1'] - 1) for p, q in picks if q is not None]
                fal = f"{np.mean(falha):.3f} {np.min(falha):.3f}" if falha else '  --    -- '
                RES.setdefault(f'{alpha:.2f}', {}).setdefault(nivel, {})[a] = {
                    'n': len(ks), 'escolha_cumpre_pct_exec': 100 * sum(cumpre) / len(ks),
                    'lcb_cumpre_pct_exec': 100 * sum(c_lcb) / len(ks),
                    'lcb_dcusto_mediana': float(np.median(dc)), 'lcb_dcusto_media': float(np.mean(dc)),
                    'falha_pon_media': float(np.mean(falha)) if falha else None,
                    'falha_pon_min': float(np.min(falha)) if falha else None}
                log(f'    {a:11s} n={len(ks):3d} | {100*len(picks)/len(ks):5.1f}% | {100*np.mean(cumpre):5.1f}% | '
                    f'falhas P_on média/mín {fal} | LCB: escolha {100*len(lcb)/len(ks):5.1f}% cumpre {100*np.mean(c_lcb):5.1f}% '
                    f'Δcusto vs escolha mediana {np.median(dc):+.2f} média {np.mean(dc):+.2f}')
    rel.close()
    json.dump(RES, open(os.path.join(OUT, 'decisor.json'), 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()
