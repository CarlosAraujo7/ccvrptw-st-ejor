#!/usr/bin/env python3
"""
Tabela e relatório do caso São Paulo v2 para o artigo (14/10/2026, diário Entrada 94).
Lê os *_oosC.json de results_c_sp/ (avaliação fora da amostra em 1.000 cenários comuns, feita por
rodar_caso_sp.py --analisar) e, se existirem, os refinamentos de pesquisa/resultados_caso_sp/refinamento/.

Por nível de incerteza (MEDIUM, HIGH), sobre as execuções (8 instâncias × 10 sementes; 4 instâncias
por nível, poucas para testes por instância, então a unidade é a execução):
  - frente com solução que cumpre α fora da amostra (% de execuções);
  - plano escolhido (mais barato certificado) cumpre α fora da amostra (% das execuções com escolha);
  - custo da solução mais barata que cumpre, relativo ao SIM+C da mesma instância e semente (mediana, %);
  - preço da pontualidade: plano escolhido de SIM+C contra a solução mais barata do DET
    (medianas: custo %, perda p.p., CO2 %), como na Tabela 7.
CAL = degrau da escada 0,70/0,80/0,90/0,95 com a escolha certificada mais barata (como no benchmark).
Uso: python pesquisa/caso_sp/tabela_caso_sp.py
Saídas: pesquisa/resultados_caso_sp/relatorio_tabela_caso_sp.txt e latex_ejor_c/tables/tab_caso_sp.tex
"""
import glob, json, os
from collections import defaultdict
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(AQUI, '..', '..'))
RES = os.path.join(BASE, 'results_c_sp')
OUT = os.path.join(BASE, 'pesquisa', 'resultados_caso_sp')
TAB = os.path.join(BASE, 'pesquisa', 'latex_ejor_c', 'tables', 'tab_caso_sp.tex')
if os.environ.get('CCVRP_SP_VARIANTE') == 'pico':      # variante do pico (15/10/2026, análise A3)
    RES = os.path.join(BASE, 'results_c_sp_pico')
    OUT = os.path.join(BASE, 'pesquisa', 'resultados_caso_sp_pico')
    TAB = os.path.join(BASE, 'pesquisa', 'latex_ejor_c', 'tables', 'tab_caso_sp_pico.tex')
ALPHA = 0.90
ESCADA = ['C-BUF70C', 'C-BUF80C', 'C-BUF90C', 'C-BUF95C']
LINHAS = ['C-DET', 'C-DETC', 'C-BUF90C', 'C-BUF95C', 'CAL', 'C-SIMC']
NOME = {'C-DET': r'\mDET{}', 'C-DETC': r'\mDETC{}', 'C-BUF90C': r'\mBUF{0.90}', 'C-BUF95C': r'\mBUF{0.95}',
        'CAL': r'\mCAL{}', 'C-SIMC': r'\mSIM{}', 'CAL+R': r'\mCAL{}+R'}


def carregar():
    R = {}
    for f in glob.glob(os.path.join(RES, 'C-*', '*_oosC.json')):
        d = json.load(open(f, encoding='utf-8')); m = d['meta']
        R[(m['algorithm'], m['instance'], m['seed'])] = d['solutions']
    return R


def resumo(sols):
    S = [s for s in sols if s['feasible']]
    ok = [s for s in S if s['p_on'] >= ALPHA]
    cert = [s for s in S if (s.get('p_cert') or 0) >= ALPHA]
    esc = min(cert, key=lambda s: s['f1']) if cert else None
    return dict(front_ok=bool(ok), custo_ok=min(s['f1'] for s in ok) if ok else np.nan,
                esc=esc, esc_ok=(esc['p_on'] >= ALPHA) if esc else None,
                mais_barata=min(S, key=lambda s: s['f1']) if S else None)


def main():
    R = carregar()
    chaves = sorted({(i, s) for (_, i, s) in R})
    run = defaultdict(dict)                      # (inst, seed) -> método -> resumo
    for (a, i, s), sols in R.items():
        run[(i, s)][a] = resumo(sols)
    for k, d in run.items():                     # CAL: degrau com a escolha certificada mais barata
        deg = [d[a] for a in ESCADA if a in d and d[a]['esc'] is not None]
        if deg:
            run[k]['CAL'] = min(deg, key=lambda r: r['esc']['f1'])
    ref = {}
    fr = os.path.join(OUT, 'refinamento', 'refinamento_CAL.json')
    if os.path.exists(fr):
        for x in json.load(open(fr, encoding='utf-8')):
            ref[(x['instance'], x['seed'])] = x
    rel = [f'CASO SÃO PAULO v2 — {len(chaves)} execuções (instância × semente); α = {ALPHA}; avaliação em 1.000 cenários comuns']
    tab = {}
    for nivel in ('MEDIUM', 'HIGH'):
        ks = [k for k in chaves if k[0].endswith(nivel)]
        rel.append(f'\n{nivel} ({len(ks)} execuções)')
        for a in LINHAS + (['CAL+R'] if ref else []):
            if a == 'CAL+R':
                xs = [ref[k] for k in ks if k in ref]
                fo = np.nan; eo = 100 * np.mean([x['fin_p_oos'] >= ALPHA for x in xs]) if xs else np.nan
                dc = [100 * (x['fin_f1'] / run[(x['instance'], x['seed'])]['C-SIMC']['esc']['f1'] - 1)
                      for x in xs if run[(x['instance'], x['seed'])].get('C-SIMC', {}).get('esc')]
                dc_txt = np.median(dc) if dc else np.nan
                tab[(nivel, a)] = (fo, eo, dc_txt, 'escolha')
                rel.append(f'  {a:10s} escolha cumpre {eo:5.1f}%  custo da escolha vs escolha do SIM+C (mediana) {dc_txt:+.2f}%')
                continue
            rr = [run[k][a] for k in ks if a in run[k]]
            fo = 100 * np.mean([r['front_ok'] for r in rr])
            esc = [r['esc_ok'] for r in rr if r['esc_ok'] is not None]
            eo = 100 * np.mean(esc) if esc else np.nan
            dc = [100 * (run[k][a]['custo_ok'] / run[k]['C-SIMC']['custo_ok'] - 1) for k in ks
                  if a in run[k] and 'C-SIMC' in run[k] and np.isfinite(run[k][a]['custo_ok']) and np.isfinite(run[k]['C-SIMC']['custo_ok'])]
            med = np.median(dc) if dc else np.nan
            tab[(nivel, a)] = (fo, eo, med, 'cumpre')
            rel.append(f'  {a:10s} frente cumpre {fo:5.1f}%  escolha cumpre {eo:5.1f}% (n={len(esc)})  '
                       f'custo mín. que cumpre vs SIM+C: mediana {med:+.2f}% média {np.mean(dc) if dc else np.nan:+.2f}% '
                       f'máx {np.max(dc) if dc else np.nan:+.2f}% (n={len(dc)})')
        # preço da pontualidade: escolha do SIM+C contra a mais barata do DET
        pc, pq, pe, pdet, psim = [], [], [], [], []
        for k in ks:
            d = run[k]
            if 'C-DET' in d and 'C-SIMC' in d and d['C-SIMC']['esc'] and d['C-DET']['mais_barata']:
                a_, b_ = d['C-SIMC']['esc'], d['C-DET']['mais_barata']
                pc.append(100 * (a_['f1'] / b_['f1'] - 1)); pq.append(100 * (a_['q_oos'] - b_['q_oos']))
                pe.append(100 * (a_['f3'] / b_['f3'] - 1)); pdet.append(b_['p_on']); psim.append(a_['p_on'])
        rel.append(f'  preço (escolha do SIM+C vs mais barata do DET, medianas): P {np.median(pdet):.3f} → {np.median(psim):.3f}; '
                   f'custo {np.median(pc):+.2f}%  perda {np.median(pq):+.3f} p.p.  CO2 {np.median(pe):+.2f}%')
        tab[(nivel, 'preco')] = (np.median(pdet), np.median(psim), np.median(pc), np.median(pq), np.median(pe))
    txt = '\n'.join(rel); print(txt)
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, 'relatorio_tabela_caso_sp.txt'), 'w', encoding='utf-8').write(txt + '\n')

    def f(x, fmt):
        return '--' if x is None or not np.isfinite(x) else fmt % x
    L = [r'\begin{table}[htbp]', r'\centering',
         r'\caption{S\~ao Paulo case' + (r', planned with the morning-peak congestion' if 'pico' in RES else '')
         + f' ({len({i for i, _ in chaves})} instances $\\times$ 10 seeds; ' + r'$\alpha=0.90$; evaluation on 1{,}000 common scenarios). '
         r'Front: \% of runs whose front contains an out-of-sample-compliant solution. Chosen: \% of runs whose chosen plan (cheapest '
         r'certified) complies. Cost: median cost of the cheapest compliant solution relative to \mSIM{} in the same run (\%; for '
         r'\mCAL{}+R, cost of the chosen plan relative to the chosen plan of \mSIM{}).}',
         r'\label{tab:sp-pico}' if 'pico' in RES else r'\label{tab:sp}', r'\small', r'\begin{tabular}{@{}l rrr rrr@{}}', r'\toprule',
         r' & \multicolumn{3}{c}{Medium uncertainty} & \multicolumn{3}{c}{High uncertainty} \\',
         r'\cmidrule(lr){2-4}\cmidrule(l){5-7}',
         r'Method & Front & Chosen & Cost & Front & Chosen & Cost \\', r'\midrule']
    for a in LINHAS + (['CAL+R'] if ref else []):
        cel = []
        for nivel in ('MEDIUM', 'HIGH'):
            fo, eo, dc, _ = tab[(nivel, a)]
            cel += [f(fo, '%.0f'), f(eo, '%.0f'), '--' if a == 'C-SIMC' else f(dc, '%+.2f')]
        L.append(NOME[a] + ' & ' + ' & '.join(cel) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(TAB, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print(f'tabela: {TAB}')


if __name__ == '__main__':
    main()
