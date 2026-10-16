#!/usr/bin/env python3
"""
Pipeline "buffer em qualquer resolvedor" com um resolvedor externo (15/10/2026, análise de fraquezas A2).

Pergunta: o argumento de implementabilidade do artigo — tempos inflados podem alimentar QUALQUER
resolvedor determinístico, e só a checagem final precisa de simulação — funciona com um resolvedor
de estado da arte que nada sabe de simulação, qualidade ou CO2?

Pipeline (sem nenhuma simulação dentro da busca):
  1. PyVRP 0.13.3 resolve o problema de CUSTO com os tempos de viagem inflados ao quantil q da
     lognormal (o mesmo multiplicador dos métodos BUF; q ∈ {0,80; 0,90; 0,95}), com o mesmo tempo
     por resolução da referência PyVRP (referencia_pyvrp.TEMPO); rotas saem no instante 0;
  2. certificação em 500 cenários (mesma amostra de certificação de refinamento_simulacao.py);
     PY-CAL = o degrau q com a solução certificada (P_cert ≥ α) mais barata, como o CAL;
  3. refinamento +R (refinamento_simulacao.refinar_custo e ajustar_partidas, com a mesma porteira);
  4. avaliação nos 1.000 cenários comuns de analisar_c.py.
Comparação, por instância (média das sementes): contra o plano escolhido do SIM+C e o do CAL+R das
mesmas instâncias (pesquisa/resultados_refinamento e resultados_caso_sp/refinamento).
Protocolo fixado antes de rodar: instâncias MEDIUM e HIGH do benchmark (36) e do caso SP (8);
sementes do PyVRP 0-4; α = 0,90.
Uso: python pesquisa/pipeline_pyvrp_buffer.py --rodar [--workers 10]     depois  --analisar
Saída: pesquisa/resultados_pyvrp_pipeline/
"""
import argparse, glob, json, math, os, random, sys, time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(BASE, 'src')); sys.path.insert(0, os.path.join(BASE, 'pesquisa'))
OUT = os.path.join(BASE, 'pesquisa', 'resultados_pyvrp_pipeline')
QS = (0.80, 0.90, 0.95)
SEMENTES = (0, 1, 2, 3, 4)
ALPHA = 0.90


def _contexto_sp(caso_sp):
    if caso_sp:
        sys.path.insert(0, os.path.join(BASE, 'pesquisa', 'caso_sp'))
        import rodar_caso_sp
        rodar_caso_sp.aplicar_matrizes()


def tarefa(args):
    """Resolve, certifica, refina e avalia uma (instância, q, semente)."""
    path, q, sd, caso_sp = args
    _contexto_sp(caso_sp)
    import analisar_c as ac
    # o mesmo processo pode receber tarefas do benchmark e do caso SP: o diretório das instâncias da
    # avaliação fora da amostra é fixado a cada tarefa (build_matrices remendado cai no original
    # quando a instância não traz matriz viária)
    ac.INST_DIR = os.path.dirname(os.path.abspath(path))
    import nsga2_simheuristic as ns
    import ccvrp_c as cc
    import thermal_route as tr
    import referencia_pyvrp as rp
    import refinamento_simulacao as rs
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime
    inst = ns.load_instance(path); nome = inst['name']
    cs = inst['customers']; n = len(cs); fl = inst['fleet']
    ctx_b = cc.Contexto(inst, 0, buffer_q=q, decoder='split', p_ls=0.0)     # tempos de planejamento inflados
    ctx = cc.Contexto(inst, 0, decoder='split', p_ls=0.0)                    # tempos nominais
    fix, var, K, pen = fl['fixed_cost_BRL'], fl['variable_cost_BRL_km'], fl['num_vehicles'], ns.EXTRA_VEHICLE_COST
    u = var / rp.ESC; BIG = 10 ** 12
    m = Model()
    dep = m.add_depot(x=0, y=0, tw_early=0, tw_late=BIG)
    m.add_vehicle_type(num_available=K, capacity=int(round(fl['capacity_kg'] * 100)),
                       fixed_cost=int(round(fix / u)), unit_distance_cost=1, tw_early=0, tw_late=BIG)
    m.add_vehicle_type(num_available=n, capacity=int(round(fl['capacity_kg'] * 100)),
                       fixed_cost=int(round((fix + pen) / u)), unit_distance_cost=1, tw_early=0, tw_late=BIG)
    cl = [m.add_client(x=0, y=0, delivery=int(round(c['demand_kg'] * 100)),
                       service_duration=int(round(c['service_time_min'] * rp.ESC)),
                       tw_early=int(round(c['ready_time_min'] * rp.ESC)),
                       tw_late=int(math.floor(c['due_date_min'] * rp.ESC))) for c in cs]
    locs = [dep] + cl
    for i, a in enumerate(locs):
        for j, b in enumerate(locs):
            if i != j:
                m.add_edge(a, b, distance=int(round(ctx.dist[i][j] * rp.ESC)),
                           duration=int(math.ceil(ctx_b.ttime_plan[i][j] * rp.ESC)))
    t0 = time.time()
    res = m.solve(stop=MaxRuntime(rp.TEMPO[n]), seed=sd + 1, display=False)
    t_sol = time.time() - t0
    R = [[int(v) for v in r.visits()] for r in res.best.routes()]
    viavel = (res.is_feasible() and all(cc._rota_viavel(r, ctx_b) for r in R)
              and sorted(c for r in R for c in r) == list(range(1, n + 1)))
    cv = inst['thermal_profile']['sigma_t_pct']
    # mesmas amostras de refinamento e certificação de refinamento_simulacao.py (por instância e semente)
    tt_ref = tr._lognormal_media1(np.random.default_rng(31337 + 97 * sd + sum(map(ord, nome))), cv, (rs.K_REF, n + 1))
    tt_cer = tr._lognormal_media1(np.random.default_rng(4242 + 89 * sd + sum(map(ord, nome))), cv, (rs.K_CERT, n + 1))
    pl = rs.Plano(ctx, tt_ref); pc = rs.Plano(ctx, tt_cer)
    D0 = [0.0] * len(R)
    p_cert0 = pc.p_conj(R, D0)
    f1, f2, f3 = pl.objetivos(R, D0)
    p_oos, q_oos = rs.avaliar_oos(R, D0, nome, ctx)
    linha = dict(instance=nome, q=q, seed=sd, n=n, viavel=viavel, t_solve_s=round(t_sol, 1), rotas=len(R),
                 base_f1=f1, base_p_cert=p_cert0, base_p_oos=p_oos, base_q_oos=q_oos, base_f3=f3)
    t1 = time.process_time()
    R1, D1 = rs.refinar_custo(R, D0, pl, random.Random(sd))
    D2 = rs.ajustar_partidas(R1, D1, pl)
    p_cert1 = pc.p_conj(R1, D2)
    g1, g2, g3 = pl.objetivos(R1, D2)
    p1, q1 = rs.avaliar_oos(R1, D2, nome, ctx)
    ok = p_cert1 >= ALPHA
    linha.update(ref_p_cert=p_cert1, ref_f1=g1, ref_p_oos=p1, ref_q_oos=q1, ref_f3=g3, ref_aceito=ok,
                 fin_f1=g1 if ok else f1, fin_p_oos=p1 if ok else p_oos, fin_q_oos=q1 if ok else q_oos,
                 fin_f3=g3 if ok else f3, cpu_ref_s=time.process_time() - t1)
    return linha


def rodar(workers):
    import referencia_pyvrp  # noqa: F401  (confere a importação antes de abrir o Pool)
    tarefas = []
    for p in sorted(glob.glob(os.path.join(BASE, 'instances', 'generated', 'CCVRPTW-ST_*.json'))):
        if p.endswith(('_MEDIUM.json', '_HIGH.json')):
            tarefas += [(p, q, s, False) for q in QS for s in SEMENTES]
    for p in sorted(glob.glob(os.path.join(BASE, 'instances', 'caso_sp', 'CCVRPTW-ST_SP*.json'))):
        tarefas += [(p, q, s, True) for q in QS for s in SEMENTES]
    os.makedirs(OUT, exist_ok=True)
    arq = os.path.join(OUT, 'pipeline.jsonl')
    feitas = set()
    if os.path.exists(arq):
        for L in open(arq, encoding='utf-8'):
            x = json.loads(L); feitas.add((x['instance'], x['q'], x['seed']))
    tarefas = [t for t in tarefas if (os.path.basename(t[0])[:-5], t[1], t[2]) not in feitas]
    random.Random(2026).shuffle(tarefas)
    print(f'{len(tarefas)} resoluções a fazer ({len(feitas)} já feitas)', flush=True)
    with Pool(workers) as pool, open(arq, 'a', encoding='utf-8') as fh:
        for i, r in enumerate(pool.imap_unordered(tarefa, tarefas), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            print(f"  [{i}/{len(tarefas)}] {r['instance']} q={r['q']} s{r['seed']} viável={r['viavel']} "
                  f"P_cert {r['base_p_cert']:.3f} -> {r['ref_p_cert']:.3f}", flush=True)


def analisar():
    from scipy.stats import wilcoxon
    from equivalencia import hodges_lehmann
    from analisar_refinamento import holm
    L = [json.loads(x) for x in open(os.path.join(OUT, 'pipeline.jsonl'), encoding='utf-8')]
    L = [x for x in L if x['viavel']]
    por = defaultdict(dict)
    for x in L:
        por[(x['instance'], x['seed'])][x['q']] = x
    planos = defaultdict(dict)       # método -> (inst, seed) -> (f1, p_oos, q_oos, f3)
    for k, d in por.items():
        cert = [x for x in d.values() if x['base_p_cert'] >= ALPHA]
        if not cert:
            continue
        e = min(cert, key=lambda x: x['base_f1'])
        planos['PY-CAL'][k] = (e['base_f1'], e['base_p_oos'], e['base_q_oos'], e['base_f3'])
        planos['PY-CAL+R'][k] = (e['fin_f1'], e['fin_p_oos'], e['fin_q_oos'], e['fin_f3'])
        if 0.95 in d and d[0.95]['base_p_cert'] >= ALPHA:
            x = d[0.95]
            planos['PY-BUF0.95'][k] = (x['base_f1'], x['base_p_oos'], x['base_q_oos'], x['base_f3'])
            planos['PY-BUF0.95+R'][k] = (x['fin_f1'], x['fin_p_oos'], x['fin_q_oos'], x['fin_f3'])
    # referências: SIM+C e CAL+R do nosso motor (planos escolhidos), por instância (média das sementes 0-9)
    ref = defaultdict(dict)
    for d in (os.path.join(BASE, 'pesquisa', 'resultados_refinamento'), os.path.join(BASE, 'pesquisa', 'resultados_caso_sp', 'refinamento')):
        for f, rot, pre in (('refinamento_C-SIMC.json', 'SIM+C', 'base'), ('refinamento_CAL.json', 'CAL+R', 'fin')):
            p = os.path.join(d, f)
            if os.path.exists(p):
                for x in json.load(open(p, encoding='utf-8')):
                    ref[rot].setdefault(x['instance'], []).append((x[f'{pre}_f1'], x[f'{pre}_p_oos'], x[f'{pre}_q_oos'], x[f'{pre}_f3']))
    out = [f'PIPELINE PyVRP + buffer + certificação (+R): {len(L)} resoluções viáveis; α = {ALPHA}; avaliação em 1.000 cenários comuns']
    tab = {}                     # (grupo, método, referência) -> {medida: (HL, p_Holm)}, cumprimento, nº de instâncias
    cumpre = {}                  # (grupo, método) -> % que cumpre
    for grupo, filtro in (('benchmark', lambda i: '_SP' not in i), ('caso SP', lambda i: '_SP' in i)):
        for m, d in planos.items():
            v = [d[k][1] for k in d if filtro(k[0])]
            if v:
                cumpre[(grupo, m)] = 100 * np.mean(np.array(v) >= ALPHA)
        for rot in ('SIM+C', 'CAL+R'):
            v = [t[1] for i in ref[rot] if filtro(i) for t in ref[rot][i]]
            if v:
                cumpre[(grupo, rot)] = 100 * np.mean(np.array(v) >= ALPHA)
        out.append(f'\n== {grupo} ==')
        for m, d in planos.items():
            ks = [k for k in d if filtro(k[0])]
            if not ks:
                continue
            v = np.array([d[k] for k in ks])
            out.append(f'  {m:13s} execuções {len(ks):3d}  cumpre {100*np.mean(v[:,1] >= ALPHA):5.1f}%  custo {v[:,0].mean():8.1f}  '
                       f'perda {100*v[:,2].mean():.3f}%  CO2 {v[:,3].mean():.1f}')
        for rot in ('SIM+C', 'CAL+R'):
            insts = [i for i in ref[rot] if filtro(i)]
            v = np.array([t for i in insts for t in ref[rot][i]])
            if len(v):
                out.append(f'  {rot:13s} execuções {len(v):3d}  cumpre {100*np.mean(v[:,1] >= ALPHA):5.1f}%  custo {v[:,0].mean():8.1f}  '
                           f'perda {100*v[:,2].mean():.3f}%  CO2 {v[:,3].mean():.1f}   (nosso motor)')
        for m in ('PY-CAL+R', 'PY-BUF0.95+R'):
            for rot in ('SIM+C', 'CAL+R'):
                res = []
                for idx, nm, esc in ((0, 'custo (%)', 'rel'), (2, 'perda (p.p.)', 'abs'), (3, 'CO2 (%)', 'rel'), (1, 'P_oos (p.p.)', 'abs')):
                    pi = defaultdict(list)
                    for k, t in planos[m].items():
                        if filtro(k[0]):
                            pi[k[0]].append(t[idx])
                    I = sorted(i for i in pi if i in ref[rot])
                    if len(I) < 5:
                        continue
                    x = np.array([np.mean(pi[i]) for i in I]); y = np.array([np.mean([t[idx] for t in ref[rot][i]]) for i in I])
                    dd = 100 * (x / y - 1) if esc == 'rel' else 100 * (x - y)
                    hl, lo, hi = hodges_lehmann(dd, 0.95)
                    p = wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
                    res.append((nm, len(I), hl, lo, hi, p, int(np.sum(dd < 0)), int(np.sum(dd > 0))))
                if res:
                    out.append(f'  {m} contra {rot} (por instância; HL [IC 95%]; Wilcoxon e Holm nas 4 medidas; nº de instâncias menor/maior)')
                    for (nm, ni, hl, lo, hi, p, mn, mx), pa in zip(res, holm([r[5] for r in res])):
                        out.append(f'      {nm:14s} n={ni:2d}  HL {hl:+7.3f} [{lo:+7.3f}, {hi:+7.3f}]  p={p:.3g}  p_Holm={pa:.3g}  ({mn} menores, {mx} maiores)')
                        tab.setdefault((grupo, m, rot), {})[nm] = (hl, pa, ni, mn, mx)
    txt = '\n'.join(out); print(txt)
    open(os.path.join(OUT, 'relatorio_pipeline.txt'), 'w', encoding='utf-8').write(txt + '\n')
    tabela_tex(tab, cumpre)


def tabela_tex(tab, cumpre):
    """Tabela do artigo: diferenças pareadas por instância (HL; * = p_Holm < 0,05) e cumprimento."""
    NOME = {'PY-BUF0.95+R': r'PyVRP, BUF$_{0.95}$+C+R', 'PY-CAL+R': r'PyVRP, CAL+R', 'SIM+C': r'\mSIM{}', 'CAL+R': r'\mCAL{}+R'}

    def cel(d, nm):
        if nm not in d:
            return '--'
        hl, pa = d[nm][0], d[nm][1]
        return ('0.00' if abs(hl) < 0.005 else f'{hl:+.2f}') + ('$^*$' if pa < 0.05 else '')
    L = [r'\begin{table}[htbp]', r'\centering',
         r'\caption{Buffers with an external solver: PyVRP solves the cost problem with travel times inflated to the $q$-quantile, '
         r'the plan is certified on 500 scenarios and refined (+R); PyVRP, CAL+R takes the cheapest certified rung of $q\in\{0.80,0.90,0.95\}$. '
         r'Hodges--Lehmann estimate of the paired difference per instance (PyVRP plan minus reference; mean over seeds); $^*$: $p_{\mathrm{Holm}}<0.05$ '
         r'(Wilcoxon signed-rank test, Holm correction over the four measures). Complies: \% of chosen plans meeting $\alpha=0.90$ out of sample, '
         r'PyVRP plan / reference.}',
         r'\label{tab:pyvrp-pipeline}', r'\small', r'\begin{tabular}{@{}l l r r r r@{}}', r'\toprule',
         r'Instances & Comparison & Cost (\%) & Quality loss (p.p.) & CO$_2$ (\%) & Complies (\%) \\', r'\midrule']
    for grupo, rot_g in (('benchmark', 'Benchmark'), ('caso SP', r'S\~ao Paulo')):
        primeiro = True
        for m, rot in (('PY-BUF0.95+R', 'SIM+C'), ('PY-CAL+R', 'SIM+C'), ('PY-CAL+R', 'CAL+R')):
            d = tab.get((grupo, m, rot))
            if not d:
                continue
            n = next(iter(d.values()))[2]
            g = f'{rot_g} ($n={n}$)' if primeiro else ''
            primeiro = False
            c1, c2 = cumpre.get((grupo, m)), cumpre.get((grupo, rot))
            L.append(f'{g} & {NOME[m]} vs {NOME[rot]} & {cel(d, "custo (%)")} & {cel(d, "perda (p.p.)")} & {cel(d, "CO2 (%)")} & '
                     f'{c1:.1f} / {c2:.1f} \\\\')
        L.append(r'\midrule')
    L[-1] = r'\bottomrule'
    L += [r'\end{tabular}', r'\end{table}']
    caminho = os.path.join(BASE, 'pesquisa', 'latex_ejor_c', 'tables', 'tab_pyvrp_pipeline.tex')
    open(caminho, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print(f'tabela: {caminho}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rodar', action='store_true')
    ap.add_argument('--analisar', action='store_true')
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    if a.rodar:
        rodar(a.workers)
    if a.analisar:
        analisar()
