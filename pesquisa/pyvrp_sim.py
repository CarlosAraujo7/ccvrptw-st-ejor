#!/usr/bin/env python3
"""
Simheurística sobre o PyVRP (PY-SIM): retroalimentação por simulação em volta de um resolvedor externo
(18/10/2026, diário Entrada 118, ciclo B4; desenho fixado antes de rodar).

Por execução (instância, semente), até N_RES = 3 resoluções do PyVRP (o mesmo orçamento do PY-CAL, que resolve
os 3 degraus), cada uma com o tempo do pipeline (referencia_pyvrp.TEMPO; 240/480 s na escala):
  k = 0  problema de CUSTO com os tempos NOMINAIS e as janelas originais; rotas saem no instante 0;
  depois de cada resolução, o plano é simulado em M_BUSCA = 200 cenários de busca (independentes dos de
  certificação, refinamento e avaliação); para cada cliente j, g_j = Q_0,99(A_j) − (b_j + δ), com A_j a chegada;
  se g_j > 0, o prazo DO MODELO encolhe: b_j ← b_j − g_j (cumulativo; piso max(a_j, τ_0j nominal)).
  A simulação protege só os clientes críticos (Proposição 2), em vez de inflar todos os arcos.
  Para quando nenhum cliente é crítico ou quando uma resolução é inviável.
Escolha: iteração certificada mais barata (P_cert ≥ α em 500 cenários; também a regra LCB); refinamento +R
(refinamento_simulacao, porteira α ou LCB); avaliação nos 1.000 cenários comuns. Amostras de refinamento e
certificação iguais às de pipeline_pyvrp_buffer.py (por instância e semente).
Uso: python pesquisa/pyvrp_sim.py --rodar --conjuntos bench sp escala --workers 10
     python pesquisa/pyvrp_sim.py --analisar
"""
import argparse, glob, json, math, os, random, sys, time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(BASE, 'src')); sys.path.insert(0, os.path.join(BASE, 'pesquisa'))
OUT = os.path.join(BASE, 'pesquisa', 'resultados_pyvrp_sim')
ALPHA = 0.90
LCB = ALPHA + 1.645 * math.sqrt(ALPHA * (1 - ALPHA) / 500)
M_BUSCA, QUANT, N_RES = 200, 0.99, 3
CONJ = {
    'bench': (os.path.join(BASE, 'instances', 'generated'), ('_MEDIUM.json', '_HIGH.json'), (0, 1, 2, 3, 4), False),
    'sp': (os.path.join(BASE, 'instances', 'caso_sp'), ('.json',), (0, 1, 2, 3, 4), True),
    'escala': (os.path.join(BASE, 'instances', 'escala'), ('.json',), (0, 1, 2), False),
}
TEMPO_ESCALA = {200: 240, 400: 480}


def _init():
    sys.path.insert(0, os.path.join(BASE, 'pesquisa', 'caso_sp'))
    import rodar_caso_sp
    rodar_caso_sp.aplicar_matrizes()      # cai no build_matrices original quando a instância não traz matriz viária
    import referencia_pyvrp as rp
    rp.TEMPO.update(TEMPO_ESCALA)


def chegadas(R, inst, ttime, tt):
    """Chegada a cada cliente por cenário (partidas em 0), como em ccvrp_c.pontual_rota (caso estático)."""
    cust = inst['customers']; A = {}
    for r in R:
        t = np.zeros(tt.shape[0]); prev = 0
        for c in r:
            t = t + ttime[prev][c] * tt[:, c]
            A[c] = t.copy()
            t = np.maximum(t, cust[c - 1]['ready_time_min']) + cust[c - 1]['service_time_min']
            prev = c
    return A


def tarefa(args):
    path, sd = args
    import analisar_c as ac
    ac.INST_DIR = os.path.dirname(os.path.abspath(path))
    import nsga2_simheuristic as ns
    import ccvrp_c as cc
    import tempo_dependente as td
    import thermal_route as tr
    import referencia_pyvrp as rp
    import refinamento_simulacao as rs
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime
    inst = ns.load_instance(path); nome = inst['name']
    cs = inst['customers']; n = len(cs); fl = inst['fleet']
    ctx = cc.Contexto(inst, 0, decoder='split', p_ls=0.0)
    assert not td.eh_td(ctx.ttime), 'PY-SIM só para tempos estáticos'
    fix, var, K, pen = fl['fixed_cost_BRL'], fl['variable_cost_BRL_km'], fl['num_vehicles'], ns.EXTRA_VEHICLE_COST
    u = var / rp.ESC; BIG = 10 ** 12
    due = [c['due_date_min'] for c in cs]
    piso = [max(c['ready_time_min'], ctx.ttime_plan[0][j + 1]) for j, c in enumerate(cs)]
    b = list(due)

    def resolver(b):
        m = Model()
        dep = m.add_depot(x=0, y=0, tw_early=0, tw_late=BIG)
        m.add_vehicle_type(num_available=K, capacity=int(round(fl['capacity_kg'] * 100)),
                           fixed_cost=int(round(fix / u)), unit_distance_cost=1, tw_early=0, tw_late=BIG)
        m.add_vehicle_type(num_available=n, capacity=int(round(fl['capacity_kg'] * 100)),
                           fixed_cost=int(round((fix + pen) / u)), unit_distance_cost=1, tw_early=0, tw_late=BIG)
        cl = [m.add_client(x=0, y=0, delivery=int(round(c['demand_kg'] * 100)),
                           service_duration=int(round(c['service_time_min'] * rp.ESC)),
                           tw_early=int(round(c['ready_time_min'] * rp.ESC)),
                           tw_late=max(int(round(c['ready_time_min'] * rp.ESC)), int(math.floor(bj * rp.ESC))))
              for c, bj in zip(cs, b)]
        locs = [dep] + cl
        for i, a in enumerate(locs):
            for j, c in enumerate(locs):
                if i != j:
                    m.add_edge(a, c, distance=int(round(ctx.dist[i][j] * rp.ESC)),
                               duration=int(math.ceil(ctx.ttime_plan[i][j] * rp.ESC)))
        t0 = time.time()
        res = m.solve(stop=MaxRuntime(rp.TEMPO[n]), seed=sd + 1, display=False)
        R = [[int(v) for v in r.visits()] for r in res.best.routes()]
        ok = (res.is_feasible() and all(cc._rota_viavel(r, ctx) for r in R)
              and sorted(c for r in R for c in r) == list(range(1, n + 1)))
        return R, ok, time.time() - t0

    cv = inst['thermal_profile']['sigma_t_pct']
    sem = sum(map(ord, nome))
    tt_bus = tr._lognormal_media1(np.random.default_rng(5151 + 83 * sd + sem), cv, (M_BUSCA, n + 1))
    tt_ref = tr._lognormal_media1(np.random.default_rng(31337 + 97 * sd + sem), cv, (rs.K_REF, n + 1))
    tt_cer = tr._lognormal_media1(np.random.default_rng(4242 + 89 * sd + sem), cv, (rs.K_CERT, n + 1))
    pl = rs.Plano(ctx, tt_ref); pc = rs.Plano(ctx, tt_cer)
    its = []
    for k in range(N_RES):
        R, ok, t_sol = resolver(b)
        it = dict(k=k, viavel=bool(ok), t_solve_s=round(t_sol, 1), aperto_min=float(sum(d - x for d, x in zip(due, b))))
        its.append(it)
        if not ok:
            break
        A = chegadas(R, inst, ctx.ttime, tt_bus)
        on = np.ones(M_BUSCA, dtype=bool)
        g = {}
        for c, a in A.items():
            on &= (a - due[c - 1]) <= cc.DELTA_MIN
            g[c] = float(np.quantile(a, QUANT) - (due[c - 1] + cc.DELTA_MIN))
        crit = [c for c, v in g.items() if v > 1e-9]
        D0 = [0.0] * len(R)
        f1, f2, f3 = pl.objetivos(R, D0)
        it.update(R=R, p_busca=float(on.mean()), n_crit=len(crit), p_cert=pc.p_conj(R, D0), f1=f1, f3=f3, rotas=len(R))
        if not crit:
            break
        for c in crit:
            b[c - 1] = max(piso[c - 1], b[c - 1] - g[c])
    linha = dict(instance=nome, seed=sd, n=n, n_res=len(its),
                 its=[{k: v for k, v in it.items() if k != 'R'} for it in its])
    cand = [it for it in its if it['viavel']]
    for regra, lim in (('pt', ALPHA), ('lcb', LCB)):
        cert = [it for it in cand if it['p_cert'] >= ALPHA]
        if regra == 'pt':
            e = min(cert, key=lambda it: it['f1']) if cert else None
        else:
            forte = [it for it in cert if it['p_cert'] >= LCB]
            e = (min(forte, key=lambda it: it['f1']) if forte else max(cert, key=lambda it: (it['p_cert'], -it['f1']))) \
                if cert else None
        linha[f'{regra}_ok'] = e is not None
        if e is None:
            continue
        R = e['R']; D0 = [0.0] * len(R)
        p_oos, q_oos = rs.avaliar_oos(R, D0, nome, ctx)
        t1 = time.process_time()
        R1, D1 = rs.refinar_custo(R, D0, pl, random.Random(sd))
        D2 = rs.ajustar_partidas(R1, D1, pl)
        p1c = pc.p_conj(R1, D2); g1, g2, g3 = pl.objetivos(R1, D2)
        p1, q1 = rs.avaliar_oos(R1, D2, nome, ctx)
        ac_ = p1c >= lim
        linha.update({f'{regra}_iter': e['k'], f'{regra}_base_f1': e['f1'], f'{regra}_base_p_cert': e['p_cert'],
                      f'{regra}_base_p_oos': p_oos, f'{regra}_base_q_oos': q_oos, f'{regra}_base_f3': e['f3'],
                      f'{regra}_ref_p_cert': p1c, f'{regra}_ref_aceito': bool(ac_),
                      f'{regra}_fin_f1': g1 if ac_ else e['f1'], f'{regra}_fin_p_oos': p1 if ac_ else p_oos,
                      f'{regra}_fin_q_oos': q1 if ac_ else q_oos, f'{regra}_fin_f3': g3 if ac_ else e['f3'],
                      f'{regra}_cpu_ref_s': time.process_time() - t1})
    return linha


def rodar(conjuntos, workers):
    os.makedirs(OUT, exist_ok=True)
    arq = os.path.join(OUT, 'pyvrp_sim.jsonl')
    feitas = set()
    if os.path.exists(arq):
        for L in open(arq, encoding='utf-8'):
            x = json.loads(L); feitas.add((x['instance'], x['seed']))
    T = []
    for c in conjuntos:
        d, suf, sems, _ = CONJ[c]
        for p in sorted(glob.glob(os.path.join(d, 'CCVRPTW-ST_*.json'))):
            if p.endswith(suf):
                T += [(p, s) for s in sems if (os.path.basename(p)[:-5], s) not in feitas]
    T.sort(key=lambda t: -int(os.path.basename(t[0]).split('_')[2]))        # as maiores primeiro
    print(f'PY-SIM: {len(T)} execuções a fazer ({len(feitas)} já feitas)', flush=True)
    with Pool(workers, initializer=_init) as pool, open(arq, 'a', encoding='utf-8') as fh:
        for i, r in enumerate(pool.imap_unordered(tarefa, T), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            print(f"  [{i}/{len(T)}] {r['instance']} s{r['seed']} resoluções {r['n_res']} "
                  f"P_busca {[round(it.get('p_busca', float('nan')), 3) for it in r['its']]} "
                  f"escolhida {r.get('pt_iter')} P_oos {r.get('pt_fin_p_oos', float('nan')):.3f}", flush=True)
    print('FIM PY-SIM', flush=True)


def _pycal(arq, filtro, sementes):
    """PY-CAL+R pelas duas regras, a partir dos registros do pipeline: {regra: {(inst, seed): (f1, p_oos, q_oos, f3)}}."""
    por = defaultdict(dict)
    for L in open(arq, encoding='utf-8'):
        x = json.loads(L)
        if x['viavel'] and filtro(x['instance']) and x['seed'] in sementes:
            por[(x['instance'], x['seed'])][x['q']] = x
    P = {'pt': {}, 'lcb': {}}
    for k, d in por.items():
        cert = [x for x in d.values() if x['base_p_cert'] >= ALPHA]
        if not cert:
            continue
        e = min(cert, key=lambda x: x['base_f1'])
        P['pt'][k] = (e['fin_f1'], e['fin_p_oos'], e['fin_q_oos'], e['fin_f3'])
        forte = [x for x in cert if x['base_p_cert'] >= LCB]
        e = min(forte, key=lambda x: x['base_f1']) if forte else max(cert, key=lambda x: (x['base_p_cert'], -x['base_f1']))
        g = 'ref' if e['ref_p_cert'] >= LCB else 'base'
        P['lcb'][k] = (e[f'{g}_f1'], e[f'{g}_p_oos'], e[f'{g}_q_oos'], e[f'{g}_f3'])
    return P


def _motor(pasta, filtro):
    """SIM+C (plano escolhido) e CAL+R do nosso motor, pela regra pontual e pela LCB."""
    P = {('SIM+C', 'pt'): {}, ('CAL+R', 'pt'): {}, ('SIM+C', 'lcb'): {}, ('CAL+R', 'lcb'): {}}
    for arq, m, g in (('refinamento_C-SIMC', 'SIM+C', 'base'), ('refinamento_CAL', 'CAL+R', 'fin')):
        for regra, suf, gg in (('pt', '', g), ('lcb', '_lcb', 'base' if m == 'SIM+C' else 'finl')):
            p = os.path.join(BASE, pasta, arq + suf + '.json')
            if os.path.exists(p):
                for x in json.load(open(p, encoding='utf-8')):
                    if filtro(x['instance']):
                        P[(m, regra)][(x['instance'], x['seed'])] = (x[f'{gg}_f1'], x[f'{gg}_p_oos'], x[f'{gg}_q_oos'], x[f'{gg}_f3'])
    return P


def analisar():
    from scipy.stats import wilcoxon
    from equivalencia import hodges_lehmann
    from analisar_refinamento import holm
    S = [json.loads(L) for L in open(os.path.join(OUT, 'pyvrp_sim.jsonl'), encoding='utf-8')]
    G = [('Benchmark', lambda i: '_SP' not in i and i.split('_')[2] in ('025', '050', '100'),
          os.path.join(BASE, 'pesquisa', 'resultados_pyvrp_pipeline', 'pipeline.jsonl'), (0, 1, 2, 3, 4),
          'pesquisa/resultados_refinamento'),
         ('São Paulo', lambda i: '_SP' in i, os.path.join(BASE, 'pesquisa', 'resultados_pyvrp_pipeline', 'pipeline.jsonl'),
          (0, 1, 2, 3, 4), 'pesquisa/resultados_caso_sp/refinamento'),
         ('200-400 customers', lambda i: i.split('_')[2] in ('200', '400'),
          os.path.join(BASE, 'pesquisa', 'resultados_escala', 'pipeline_escala.jsonl'), (0, 1, 2),
          'pesquisa/resultados_escala/refinamento')]
    MED = (('custo (%)', 0, 'rel'), ('perda (p.p.)', 2, 'abs'), ('CO2 (%)', 3, 'rel'), ('P_oos (p.p.)', 1, 'abs'))

    def por_inst(D, idx):
        pi = defaultdict(list)
        for (i, _), t in D.items():
            pi[i].append(t[idx])
        return {i: float(np.mean(v)) for i, v in pi.items()}

    def comparar(A, B):
        res = []
        for nm, idx, esc in MED:
            x, y = por_inst(A, idx), por_inst(B, idx)
            I = sorted(set(x) & set(y))
            a = np.array([x[i] for i in I]); c = np.array([y[i] for i in I])
            dd = 100 * (a / c - 1) if esc == 'rel' else 100 * (a - c)
            hl, lo, hi = hodges_lehmann(dd, 0.95)
            p = wilcoxon(dd).pvalue if np.any(np.abs(dd) > 1e-12) else 1.0
            res.append(dict(medida=nm, n=len(I), hl=hl, lo=lo, hi=hi, p=p, menores=int(np.sum(dd < -1e-12)),
                            maiores=int(np.sum(dd > 1e-12))))
        for r, pa in zip(res, holm([r['p'] for r in res])):
            r['p_holm'] = pa
        return res
    out = [f'PY-SIM (simheurística sobre o PyVRP) — {len(S)} execuções; α = {ALPHA}; 1.000 cenários comuns']
    resumo = {}
    for nome, filtro, arq, sems, pasta in G:
        X = [x for x in S if filtro(x['instance'])]
        if not X:
            continue
        PS = {r: {(x['instance'], x['seed']): (x[f'{r}_fin_f1'], x[f'{r}_fin_p_oos'], x[f'{r}_fin_q_oos'], x[f'{r}_fin_f3'])
                  for x in X if x.get(f'{r}_ok')} for r in ('pt', 'lcb')}
        PC = _pycal(arq, filtro, sems)
        PM = _motor(pasta, filtro)
        nres = [x['n_res'] for x in X]; it = [x.get('pt_iter') for x in X if x.get('pt_ok')]
        p0 = [x['its'][0].get('p_busca') for x in X if x['its'][0]['viavel']]
        out.append(f'\n=== {nome}: {len(X)} execuções ===')
        out.append(f'  resoluções usadas: média {np.mean(nres):.2f} (1: {nres.count(1)}, 2: {nres.count(2)}, 3: {nres.count(3)}); '
                   f'inviáveis: {sum(1 for x in X if not all(i["viavel"] for i in x["its"]))}; sem plano certificado: '
                   f'{sum(1 for x in X if not x.get("pt_ok"))}; iteração escolhida: {dict(zip(*np.unique(it, return_counts=True)))}')
        out.append(f'  P na busca da iteração 0 (tempos nominais): mediana {np.median(p0):.3f}')
        R = resumo[nome] = {'n': len(X), 'nres_media': float(np.mean(nres)), 'sem_plano': sum(1 for x in X if not x.get('pt_ok')),
                            'cumpre': {}, 'comp': {}}
        for regra in ('pt', 'lcb'):
            for m, D in (('PY-SIM+R', PS[regra]), ('PY-CAL+R', PC[regra]), ('SIM+C', PM[('SIM+C', regra)]),
                         ('CAL+R', PM[('CAL+R', regra)])):
                if D:
                    c = 100 * np.mean([t[1] >= ALPHA for t in D.values()])
                    R['cumpre'][f'{regra}|{m}'] = c
                    out.append(f'  [{regra}] {m:9s} execuções {len(D):3d} cumpre {c:5.1f}%  custo {np.mean([t[0] for t in D.values()]):9.1f}'
                               f'  perda {100 * np.mean([t[2] for t in D.values()]):.3f}%')
            for ref in ('PY-CAL+R', 'SIM+C', 'CAL+R'):
                B = PC[regra] if ref == 'PY-CAL+R' else PM[(ref, regra)]
                if not B:
                    continue
                res = comparar(PS[regra], B)
                R['comp'][f'{regra}|{ref}'] = res
                out.append(f'  [{regra}] PY-SIM+R − {ref}: ' + '; '.join(
                    f'{r["medida"]} {r["hl"]:+.2f} [{r["lo"]:+.2f}, {r["hi"]:+.2f}] p_H={r["p_holm"]:.2g} ({r["menores"]}/{r["maiores"]})'
                    for r in res))
    txt = '\n'.join(out); print(txt)
    open(os.path.join(OUT, 'relatorio_pyvrp_sim.txt'), 'w', encoding='utf-8').write(txt + '\n')
    json.dump(resumo, open(os.path.join(OUT, 'resumo_pyvrp_sim.json'), 'w', encoding='utf-8'), indent=1)
    tabela(resumo)


def tabela(R):
    """Tabela do suplemento: PY-SIM+R contra PY-CAL+R e contra o SIM+C (regra pontual)."""
    def cel(d, nm):
        r = next(x for x in d if x['medida'] == nm)
        return (f'{r["hl"]:+.2f}' if abs(r['hl']) >= 0.005 else '0.00') + ('$^*$' if r['p_holm'] < 0.05 else '')
    L = [r'\begin{table}[htbp]', r'\centering',
         r'\caption{A simulation loop around PyVRP (PyVRP-SIM): PyVRP solves the cost problem with nominal travel times; '
         r'after each solve the plan is simulated on 200 scenarios, and the deadline of every customer whose 99th-percentile '
         r'arrival is more than $\delta$ late is moved earlier by that amount; at most three solves, the budget of PyVRP, CAL. '
         r'The cheapest certified iterate is refined (+R). Hodges--Lehmann estimate of the paired difference per instance '
         r'(mean over seeds); $^*$: $p_{\mathrm{Holm}}<0.05$ over the four measures. Complies: \% of chosen plans meeting '
         r'$\alpha=0.90$ out of sample; no plan: runs without a certified iterate; solves: mean number of PyVRP solves used.}',
         r'\label{tab:pyvrpsim}', r'\footnotesize\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{@{}l l rrrr rrr@{}}', r'\toprule',
         r'Instances & Comparison & Cost (\%) & Loss (p.p.) & CO$_2$ (\%) & $P$ (p.p.) & Complies (\%) & No plan & Solves \\',
         r'\midrule']
    ROT = {'Benchmark': 'Benchmark', 'São Paulo': r'S\~ao Paulo', '200-400 customers': '200--400 customers'}
    for nome, rot in ROT.items():
        d = R.get(nome)
        if not d:
            continue
        for k, (ref, rref) in enumerate((('PY-CAL+R', 'PyVRP, CAL+R'), ('SIM+C', r'\mSIM{}'))):
            c = d['comp'][f'pt|{ref}']
            ini = [f'{rot} ($n={c[0]["n"]}$)', f'{d["cumpre"]["pt|PY-SIM+R"]:.1f}', f'{d["sem_plano"]}/{d["n"]}',
                   f'{d["nres_media"]:.2f}'] if k == 0 else ['', '', '', '']
            L.append(f'{ini[0]} & PyVRP-SIM+R vs {rref} & ' + ' & '.join(cel(c, nm) for nm in
                     ('custo (%)', 'perda (p.p.)', 'CO2 (%)', 'P_oos (p.p.)')) + f' & {ini[1]} & {ini[2]} & {ini[3]} \\\\')
        L.append(r'\midrule')
    L[-1] = r'\bottomrule'
    L += [r'\end{tabular}', r'\end{table}']
    caminho = os.path.join(BASE, 'pesquisa', 'latex_ejor_c', 'tables', 'tab_pyvrp_sim.tex')
    open(caminho, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('tabela:', caminho)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--analisar', action='store_true')
    ap.add_argument('--rodar', action='store_true')
    ap.add_argument('--conjuntos', nargs='+', default=['bench', 'sp', 'escala'])
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--teste', nargs=2, metavar=('INSTANCIA', 'SEMENTE'), help='uma execução, para teste de fumaça')
    a = ap.parse_args()
    if a.teste:
        _init()
        p = next(f for c in CONJ.values() for f in glob.glob(os.path.join(c[0], a.teste[0] + '.json')))
        r = tarefa((p, int(a.teste[1])))
        print(json.dumps(r, indent=1))
    if a.rodar:
        rodar(a.conjuntos, a.workers)
    if a.analisar:
        analisar()
