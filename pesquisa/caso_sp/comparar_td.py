#!/usr/bin/env python3
"""
Planejar com o horário: caso São Paulo com tempos dependentes do horário (16/10/2026, diário Entrada 101).

Pergunta Q1 do desenho fixado antes de rodar: no mundo dependente do horário (instâncias SPT, 1.000
cenários comuns de cada instância), executa SEM REPLANEJAR o plano escolhido (o mais barato certificado
no próprio mundo de planejamento) de cada método e semente, planejado com
  SP   tempos estáticos, média anual de todas as horas (58,5%; results_c_sp)
  SPM  tempos estáticos, média das horas de operação (73,5%; results_c_sp_media)
  SPP  tempos estáticos, pico da manhã (88,2%; results_c_sp_pico, só amostras 1 e 2)
  SPT  tempos dependentes do horário (results_c_sp_td)
Mede: % de execuções cujo plano cumpre α, P média, custo, perda de qualidade esperada e CO2 (duração
nominal com os tempos dependentes do horário); diferenças pareadas contra o plano SPT do mesmo método,
instância e semente. Acrescenta CAL+R planejado com o horário (refinamento em resultados_caso_sp_td).
Uso: python pesquisa/caso_sp/comparar_td.py   Saída: pesquisa/resultados_caso_sp_td/relatorio_comparar_td.txt
"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

os.environ['CCVRP_SP_VARIANTE'] = 'td'
AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(AQUI, '..', '..'))
sys.path.insert(0, os.path.join(BASE, 'src')); sys.path.insert(0, os.path.join(BASE, 'pesquisa')); sys.path.insert(0, AQUI)
import rodar_caso_sp   # noqa: E402
rodar_caso_sp.aplicar_matrizes()       # INST_DIR da avaliação = instances/caso_sp_td
import analisar_c as ac            # noqa: E402
import ccvrp_c as cc               # noqa: E402
import refinamento_simulacao as rs  # noqa: E402

ALPHA = 0.90
ESCADA = ['C-BUF70C', 'C-BUF80C', 'C-BUF90C', 'C-BUF95C']
METODOS = ['C-DETC', 'C-BUF95C', 'CAL', 'C-SIMC']
FONTES = [('SP', 'results_c_sp', 'SP'), ('SPM', 'results_c_sp_media', 'SPM'), ('SPP', 'results_c_sp_pico', 'SPP'),
          ('SPT', 'results_c_sp_td', 'SPT')]
OUT = os.path.join(BASE, 'pesquisa', 'resultados_caso_sp_td')


def escolhidos(res_dir, pref):
    """{(nome SPT, semente): {método: solução escolhida}} para os planos de uma fonte."""
    P = defaultdict(dict)
    for a in ['C-DETC', 'C-SIMC'] + ESCADA:
        for f in glob.glob(os.path.join(BASE, res_dir, a, '*_solutions.json')):
            nome, _, sd = os.path.basename(f)[:-len('_solutions.json')].split('__')
            S = json.load(open(f, encoding='utf-8'))
            cert = [s for s in S if s['feasible'] and (s.get('p_cert') or 0) >= ALPHA]
            if cert:
                P[(nome.replace('_' + pref, '_SPT', 1), int(sd.replace('seed', '')))][a] = min(cert, key=lambda s: s['f1_cost_BRL'])
    for d in P.values():
        deg = [d[a] for a in ESCADA if a in d]
        if deg:
            d['CAL'] = min(deg, key=lambda s: s['f1_cost_BRL'])
    return P


_ctx = {}


def avaliar(sol, nome_td):
    """Executa rotas e partidas no mundo dependente do horário."""
    if nome_td not in _ctx:
        inst, _, _, _ = ac._ctx(nome_td, 1)
        _ctx[nome_td] = cc.Contexto(inst, 0, decoder='split', p_ls=0.0)
    ctx = _ctx[nome_td]
    R, D = sol['routes'], sol['partida_min']
    p, q = rs.avaliar_oos(R, D, nome_td, ctx)
    f1, _, f3 = rs.Plano(ctx, np.ones((1, len(ctx.inst['customers']) + 1))).objetivos(R, D)
    return dict(p=p, q=q, f1=f1, f3=f3)


def main():
    os.makedirs(OUT, exist_ok=True)
    rel = open(os.path.join(OUT, 'relatorio_comparar_td.txt'), 'w', encoding='utf-8')

    def log(s=''):
        print(s, flush=True); rel.write(s + '\n')
    linhas = []
    for fonte, res_dir, pref in FONTES:
        if not os.path.isdir(os.path.join(BASE, res_dir)):
            log(f'{fonte}: {res_dir} não existe; pulada'); continue
        for (nome_td, sd), d in sorted(escolhidos(res_dir, pref).items()):
            if not os.path.exists(os.path.join(ac.INST_DIR, nome_td + '.json')):
                continue
            for a in METODOS:
                if a in d:
                    linhas.append(dict(fonte=fonte, inst=nome_td, seed=sd, metodo=a, nivel=nome_td.split('_')[-1],
                                       **avaliar(d[a], nome_td)))
        log(f'{fonte}: {sum(1 for x in linhas if x["fonte"] == fonte)} planos avaliados')
    # CAL+R planejado com o horário (já avaliado no mundo dependente do horário pelo refinamento)
    arq = os.path.join(OUT, 'refinamento', 'refinamento_CAL.json')
    if os.path.exists(arq):
        for r in json.load(open(arq, encoding='utf-8')):
            linhas.append(dict(fonte='SPT', inst=r['instance'], seed=r['seed'], metodo='CAL+R', nivel=r['nivel'],
                               p=r['fin_p_oos'], q=r['fin_q_oos'], f1=r['fin_f1'], f3=r['fin_f3']))
    json.dump(linhas, open(os.path.join(OUT, 'comparar_td.json'), 'w', encoding='utf-8'), indent=1)
    idx = {(x['fonte'], x['inst'], x['seed'], x['metodo']): x for x in linhas}
    for niv in ('MEDIUM', 'HIGH'):
        log(f'\n=== incerteza {niv} — planos executados no mundo dependente do horário ===')
        log(f'{"fonte":5s} {"método":9s} {"n":>4s} {"cumpre":>7s} {"P média":>8s} | vs SPT do mesmo método: '
            f'{"Δcusto":>8s} {"Δperda":>9s} {"ΔCO2":>8s} (medianas; pares)')
        for fonte, _, _ in FONTES:
            for a in METODOS + (['CAL+R'] if fonte == 'SPT' else []):
                L = [x for x in linhas if x['fonte'] == fonte and x['metodo'] == a and x['nivel'] == niv]
                if not L:
                    continue
                cum = 100 * np.mean([x['p'] >= ALPHA for x in L]); pm = np.mean([x['p'] for x in L])
                par = [(x, idx[('SPT', x['inst'], x['seed'], a)]) for x in L if ('SPT', x['inst'], x['seed'], a) in idx]
                if fonte != 'SPT' and par:
                    dc = np.median([100 * (x['f1'] / y['f1'] - 1) for x, y in par])
                    dq = np.median([100 * (x['q'] - y['q']) for x, y in par])
                    de = np.median([100 * (x['f3'] / y['f3'] - 1) for x, y in par])
                    cmp = f'{dc:+7.2f}% {dq:+8.2f}pp {de:+7.2f}%  ({len(par)})'
                else:
                    cmp = ''
                log(f'{fonte:5s} {a:9s} {len(L):4d} {cum:6.1f}% {pm:8.3f} | {cmp}')
    # Q2 no mundo dependente do horário: métodos contra SIM+C (pareado por instância, média das sementes)
    log('\n=== Q2: métodos planejados com o horário contra SIM+C (16 instâncias, média das sementes; Wilcoxon) ===')
    for a in ['C-BUF95C', 'CAL', 'CAL+R']:
        pares = defaultdict(lambda: defaultdict(list))
        for x in linhas:
            if x['fonte'] == 'SPT' and x['metodo'] in (a, 'C-SIMC'):
                pares[x['inst']][x['metodo']].append(x)
        for k, rot in (('f1', 'custo %'), ('q', 'perda p.p.'), ('f3', 'CO2 %')):
            dif = []
            for inst, d in pares.items():
                if a in d and 'C-SIMC' in d:
                    va = np.mean([x[k] for x in d[a]]); vs = np.mean([x[k] for x in d['C-SIMC']])
                    dif.append(100 * (va - vs) if k == 'q' else 100 * (va / vs - 1))
            if len(dif) >= 5:
                p = wilcoxon(dif).pvalue if any(abs(v) > 1e-12 for v in dif) else 1.0
                log(f'  {a:9s} {rot:10s} mediana {np.median(dif):+7.2f}  (n={len(dif)}, p={p:.2g})')
    rel.close()
    tabela(linhas)


ROT_F = {'SP': r'Daily average (58.5\%)', 'SPM': r'Operating-hours mean (73.5\%)',
         'SPP': r'Morning peak (88.2\%)$^a$', 'SPT': 'Time-dependent'}
ROT_M = {'C-DETC': r'\mDETC{}', 'C-BUF95C': r'\mBUF{0.95}', 'CAL': r'\mCAL{}', 'C-SIMC': r'\mSIM{}', 'CAL+R': r'\mCAL{}+R'}


def tabela(linhas):
    """Tabela do suplemento (tables/tab_comparar_td.tex): planos executados no mundo dependente do horário."""
    idx = {(x['fonte'], x['inst'], x['seed'], x['metodo']): x for x in linhas}
    L = [r'\begin{table}[htbp]', r'\centering',
         r'\caption{S\~ao Paulo case: plans built with four travel-time models, executed without replanning with the '
         r'time-dependent travel times (1{,}000 common scenarios per instance; chosen plan = cheapest certified solution in '
         r'its own planning model). Comply: \% of runs whose chosen plan meets $\alpha=0.90$; $\Delta$cost: median cost '
         r'difference to the time-dependent plan of the same method, instance and seed (\%). $^a$Samples SP1 and SP2 only.}',
         r'\label{tab:sp-td-cmp}', r'\small', r'\begin{tabular}{@{}l l rr rr@{}}', r'\toprule',
         r'Planned with & Method & \multicolumn{2}{c}{Medium uncertainty} & \multicolumn{2}{c}{High uncertainty} \\',
         r'\cmidrule(lr){3-4}\cmidrule(l){5-6}', r' & & Comply & $\Delta$cost & Comply & $\Delta$cost \\', r'\midrule']
    for fonte in ('SP', 'SPM', 'SPP', 'SPT'):
        mets = METODOS + (['CAL+R'] if fonte == 'SPT' else [])
        for k, a in enumerate(mets):
            cel = []
            for niv in ('MEDIUM', 'HIGH'):
                X = [x for x in linhas if x['fonte'] == fonte and x['metodo'] == a and x['nivel'] == niv]
                cum = 100 * np.mean([x['p'] >= ALPHA for x in X]) if X else float('nan')
                par = [(x, idx[('SPT', x['inst'], x['seed'], a)]) for x in X if ('SPT', x['inst'], x['seed'], a) in idx]
                dc = (f"{round(float(np.median([100 * (x['f1'] / y['f1'] - 1) for x, y in par])), 2) + 0.0:+.2f}"
                      if fonte != 'SPT' and par else '--')
                cel += [f'{cum:.0f}', dc]
            L.append(f"{ROT_F[fonte] if k == 0 else ''} & {ROT_M[a]} & " + ' & '.join(cel) + r' \\')
        if fonte != 'SPT':
            L.append(r'\addlinespace')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    arq = os.path.join(BASE, 'pesquisa', 'latex_ejor_c', 'tables', 'tab_comparar_td.tex')
    open(arq, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('tabela:', arq)


if __name__ == '__main__':
    if '--tabela' in sys.argv:
        tabela(json.load(open(os.path.join(OUT, 'comparar_td.json'), encoding='utf-8')))
    else:
        main()
