#!/usr/bin/env python3
"""
Desvio entre a pontualidade certificada (500 cenários) e a fora da amostra (1.000 cenários
comuns), por método, sobre todas as soluções finais certificadas (14/10/2026, diário Entrada 93).
O texto (Seção 6.3) cita a média do desvio e o percentil 95 do desvio absoluto; antes esses
números não tinham script próprio. Também relata o otimismo da estimativa na amostra de busca
(in-sample − fora da amostra) dos métodos que simulam durante a busca.
Uso: python pesquisa/desvio_certificacao.py [--dir results_c]
Saída: pesquisa/resultados_robustez/relatorio_desvio_certificacao.txt
"""
import argparse, glob, json, os
import numpy as np

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
METODOS = ['C-DETC', 'C-BUF70C', 'C-BUF80C', 'C-BUF90C', 'C-BUF95C', 'C-SIMC', 'C-MOEADC',
           'C-BUF90C95', 'C-BUF95C95', 'C-BUF99C95', 'C-SIMC95']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='results_c')
    a = ap.parse_args()
    out = [f'DESVIO certificação (500) − fora da amostra (1.000), todas as soluções finais certificadas ({a.dir})',
           f"{'método':12s} {'soluções':>9s} {'média':>8s} {'p95 |desvio|':>13s} {'otimismo busca (média)':>24s}"]
    medias, p95s = [], []
    for m in METODOS:
        d, ins = [], []
        for f in glob.glob(os.path.join(BASE, a.dir, m, '*_oosC.json')):
            for s in json.load(open(f, encoding='utf-8'))['solutions']:
                if s.get('p_cert') is not None:
                    d.append(s['p_cert'] - s['p_on'])
                if s.get('feasible') and s.get('p_on_insample') is not None and np.isfinite(s['p_on_insample']):
                    ins.append(s['p_on_insample'] - s['p_on'])
        if not d:
            continue
        d = np.array(d); medias.append(d.mean()); p95s.append(np.percentile(np.abs(d), 95))
        oti = f'{np.mean(ins):+.4f}' if ins else '--'
        out.append(f'{m:12s} {len(d):9d} {d.mean():+8.4f} {np.percentile(np.abs(d), 95):13.4f} {oti:>24s}')
    out.append(f'máximo entre métodos: média {max(medias, key=abs):+.4f}; p95 do desvio absoluto {max(p95s):.4f}')
    txt = '\n'.join(out); print(txt)
    os.makedirs(os.path.join(BASE, 'pesquisa', 'resultados_robustez'), exist_ok=True)
    open(os.path.join(BASE, 'pesquisa', 'resultados_robustez', 'relatorio_desvio_certificacao.txt'), 'w',
         encoding='utf-8').write(txt + '\n')


if __name__ == '__main__':
    main()
