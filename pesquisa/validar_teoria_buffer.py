#!/usr/bin/env python3
"""
Validação empírica das propriedades analíticas do buffer calibrado (16/10/2026, diário Entrada 102;
melhoria 3 da análise crítica). As proposições estão no material suplementar (Seção S9).

  Lema 1   chegadas não decrescentes nos multiplicadores ⇒ conjuntos viáveis com buffer ENCAIXADOS
           (B(m') ⊆ B(m) para m' ≥ m) e custo ótimo com buffer C(q) não decrescente em q.
  Prop. 1  (associação, Esary, Proschan & Walkup 1967) Π_j π_j ≤ P_on ≤ min_j π_j, com π_j a
           pontualidade do cliente j (independência entre rotas: P_on = Π_r P_on(r)).
  Prop. 2  (recursão de Lindley + Cantelli) num plano viável com buffer m, o risco do cliente j é
           ≤ Σ_p κ²V_pj / (κ²V_pj + ((m−1)T_pj + W_pj + σ_j + δ)²): diminui com a folga σ_j, as esperas
           planejadas W e a margem do buffer acumulada no caminho.
  Prop. 3  se a margem uniforme de buffer q(x*) do ótimo com restrição de chance for ≥ o menor degrau
           seguro, o CAL é ótimo; senão, a perda é ≤ C(q⁺) − C(q(x*)) (inclinação da curva C(q)).

Medidas, nos planos escolhidos (mais barato certificado) de DET+C, BUF0.70–0.95+C, CAL e SIM+C, com os
1.000 cenários comuns de analisar_c.py (benchmark, execução principal v2, e caso SP):
  V1  o sanduíche da Prop. 1 (com tolerância de Monte Carlo) e quantos clientes são críticos;
  V2  validade e poder da cota da Prop. 2 em cada cliente dos planos BUF; risco máximo por cliente × 1 − q;
  V3  monotonicidade e inclinação de Ĉ(q) (melhor de 10 sementes, degraus DET, 0.70, 0.80, 0.90, 0.95);
      margem q(x) do melhor plano certificado conhecido × degrau escolhido pelo CAL; perda realizada do CAL.
Uso: python pesquisa/validar_teoria_buffer.py   Saída: pesquisa/resultados_teoria_buffer/
"""
import glob, json, math, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import norm

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(BASE, 'src')); sys.path.insert(0, os.path.join(BASE, 'pesquisa'))
sys.path.insert(0, os.path.join(BASE, 'pesquisa', 'caso_sp'))
import rodar_caso_sp           # noqa: E402
rodar_caso_sp.aplicar_matrizes()   # matrizes viárias quando a instância as tem; senão as euclidianas
import analisar_c as ac        # noqa: E402
import ccvrp_c as cc           # noqa: E402

ALPHA = 0.90; DELTA = cc.DELTA_MIN
ESCADA = [('C-BUF70C', 0.70), ('C-BUF80C', 0.80), ('C-BUF90C', 0.90), ('C-BUF95C', 0.95)]
METODOS = ['C-DETC'] + [a for a, _ in ESCADA] + ['C-SIMC']
GRUPOS = [('benchmark', os.path.join(BASE, 'results_c'), os.path.join(BASE, 'instances', 'generated')),
          ('SP', os.path.join(BASE, 'results_c_sp'), os.path.join(BASE, 'instances', 'caso_sp'))]
OUT = os.path.join(BASE, 'pesquisa', 'resultados_teoria_buffer')


def q_de_m(m, cv):
    """Nível de quantil cujo multiplicador da lognormal (média 1, CV cv) é m."""
    s = math.sqrt(math.log(1 + cv * cv))
    return float(norm.cdf((math.log(m) + s * s / 2) / s)) if m > 0 else 0.0


def margem(R, D, inst, T):
    """Maior m tal que o plano (rotas e partidas) fica sem atraso com todos os tempos × m."""
    cust = inst['customers']

    def ok(m):
        for r, d in zip(R, D):
            t = d; prev = 0
            for c in r:
                t += m * T[prev][c]
                if t > cust[c - 1]['due_date_min'] + 1e-9:
                    return False
                t = max(t, cust[c - 1]['ready_time_min']) + cust[c - 1]['service_time_min']; prev = c
        return True
    lo, hi = 0.0, 1.0
    if ok(hi):
        while ok(2 * hi):
            hi *= 2
        lo, hi = hi, 2 * hi
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def por_cliente(r, d, inst, T, tt):
    """Pontualidade por cenário de cada cliente da rota (K×m) — como cc.pontual_rota, sem o 'e' conjunto."""
    cust = inst['customers']; K = tt.shape[0]
    t = np.full(K, float(d)); prev = 0; out = []
    for c in r:
        t = t + T[prev][c] * tt[:, c]
        out.append((t - cust[c - 1]['due_date_min']) <= DELTA)
        t = np.maximum(t, cust[c - 1]['ready_time_min']) + cust[c - 1]['service_time_min']; prev = c
    return np.array(out).T


def cota_prop2(r, d, inst, T, m, cv):
    """Cota da Prop. 2 para cada cliente da rota, no cronograma planejado com tempos × m."""
    cust = inst['customers']
    tau, w, sig = [], [], []
    t = d; prev = 0
    for c in r:
        tau.append(T[prev][c]); t += m * T[prev][c]
        sig.append(cust[c - 1]['due_date_min'] - t)
        wq = max(0.0, cust[c - 1]['ready_time_min'] - t); w.append(wq)
        t += wq + cust[c - 1]['service_time_min']; prev = c
    cotas = []
    for j in range(len(r)):
        tot = 0.0
        for p in range(j + 1):                 # segmento dos arcos p..j (índices 0-based), esperas em p..j-1
            Tp = sum(tau[p:j + 1]); Vp = sum(x * x for x in tau[p:j + 1]); Wp = sum(w[p:j])
            lim = (m - 1) * Tp + Wp + max(0.0, sig[j]) + DELTA
            v = cv * cv * Vp
            tot += v / (v + lim * lim) if lim > 0 else 1.0
        cotas.append(min(1.0, tot))
    return cotas


def escolha(S):
    cert = [s for s in S if s['feasible'] and (s.get('p_cert') or 0) >= ALPHA]
    return min(cert, key=lambda s: s['f1_cost_BRL']) if cert else None


def main():
    os.makedirs(OUT, exist_ok=True)
    rel = open(os.path.join(OUT, 'relatorio_teoria_buffer.txt'), 'w', encoding='utf-8')

    def log(s=''):
        print(s, flush=True); rel.write(s + '\n')
    resumo = {}
    for grupo, res, idir in GRUPOS:
        ac.INST_DIR = idir; ac._cache.clear()
        log(f'\n===== {grupo} ({os.path.relpath(res, BASE)}) =====')
        V1 = defaultdict(list); V2 = defaultdict(list); curva = defaultdict(dict); runs = defaultdict(dict)
        nomes = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(res, 'C-SIMC', '*_solutions.json'))})
        for nome in nomes:
            inst, T, sc, _ = ac._ctx(nome, 1)
            cv = inst['thermal_profile']['sigma_t_pct']; tt = sc['tt']
            for sd in range(10):
                esc = {}
                for a in METODOS + ['C-DET']:
                    f = os.path.join(res, a, f'{nome}__{a}__seed{sd}_solutions.json')
                    if not os.path.exists(f):
                        continue
                    S = json.load(open(f, encoding='utf-8'))
                    viav = [s for s in S if s['feasible']]
                    if viav:                        # Ĉ(q): solução viável mais barata do degrau (sem certificação)
                        curva[nome].setdefault(a, []).append(min(s['f1_cost_BRL'] for s in viav))
                    if a != 'C-DET':
                        e = escolha(S)
                        if e:
                            esc[a] = e
                lad = [(esc[a], q) for a, q in ESCADA if a in esc]
                if lad:
                    esc['CAL'], q_cal = min(lad, key=lambda x: x[0]['f1_cost_BRL'])
                else:
                    q_cal = None
                for a, s in esc.items():
                    R, D = s['routes'], s['partida_min']
                    pc = [por_cliente(r, d, inst, T, tt) for r, d in zip(R, D)]
                    pi = np.concatenate([x.mean(axis=0) for x in pc])
                    P = float(np.prod([x.all(axis=1).mean() for x in pc]))   # rotas independentes
                    Pj = float(np.all(np.concatenate(pc, axis=1), axis=1).mean())
                    prod = float(np.prod(pi)); mn = float(pi.min())
                    tol = 1.96 * math.sqrt(max(Pj * (1 - Pj), 1e-4) / tt.shape[0])
                    V1[a].append(dict(prod=prod, P=Pj, P_indep=P, mn=mn, ok_inf=prod <= Pj + tol, ok_sup=Pj <= mn + 1e-12,
                                      ncrit=int(np.sum(pi < 0.999)), n=len(pi)))
                    m = margem(R, D, inst, T)
                    runs[(nome, sd)][a] = dict(f1=s['f1_cost_BRL'], m=m, q=q_de_m(m, cv))
                    qa = dict(ESCADA).get(a)
                    if qa is not None:
                        mq = cc.multiplicador_quantil(cv, qa)
                        for r, d, x in zip(R, D, pc):
                            risco = 1 - x.mean(axis=0)
                            for cb, rk in zip(cota_prop2(r, d, inst, T, mq, cv), risco):
                                V2[a].append((cb, float(rk)))
                        V2[a + '_max'].append((float((1 - pi).max()), 1 - qa))
                if q_cal is not None:
                    runs[(nome, sd)]['q_cal'] = q_cal
            print(f'  {nome}', flush=True)
        # ---------- V1
        log('\nV1 — Prop. 1 (sanduíche Π π_j ≤ P_on ≤ min π_j, 1.000 cenários)')
        for a, L in V1.items():
            log(f'  {a:9s} planos {len(L):4d}  limite inferior ok {100 * np.mean([x["ok_inf"] for x in L]):5.1f}%  '
                f'superior ok {100 * np.mean([x["ok_sup"] for x in L]):5.1f}%  P mediana {np.median([x["P"] for x in L]):.3f}  '
                f'Π π_j mediana {np.median([x["prod"] for x in L]):.3f}  min π_j mediana {np.median([x["mn"] for x in L]):.3f}  '
                f'clientes críticos (π<0,999) mediana {np.median([x["ncrit"] for x in L]):.0f} de {np.median([x["n"] for x in L]):.0f}')
        # ---------- V2
        log('\nV2 — Prop. 2 (cota de Cantelli por cliente, planos BUF)')
        for a, _ in ESCADA:
            L = V2.get(a, [])
            if not L:
                continue
            cb = np.array([x[0] for x in L]); rk = np.array([x[1] for x in L])
            M = V2[a + '_max']
            log(f'  {a:9s} clientes {len(L):6d}  cota ≥ risco {100 * np.mean(cb + 0.0095 >= rk):5.1f}%  '
                f'cota ≤ 0,001 em {100 * np.mean(cb <= 0.001):5.1f}% (risco ≤ 0,001 em {100 * np.mean(rk <= 0.001):5.1f}%)  '
                f'risco máx. por plano: mediana {np.median([x[0] for x in M]):.3f}, ≤ 1 − q em {100 * np.mean([x[0] <= x[1] + 0.0095 for x in M]):5.1f}%')
        # ---------- V3
        log('\nV3 — Prop. 3 e Lema 1 (Ĉ(q) = melhor de 10 sementes)')
        deg = ['C-DET'] + [a for a, _ in ESCADA]
        mono = []; incl = []
        for nome, c in curva.items():
            v = [min(c[a]) for a in deg if a in c]
            if len(v) == len(deg):
                mono.append(all(x <= y * (1 + 1e-9) for x, y in zip(v, v[1:])))
                incl.append(100 * (v[-1] / v[1] - 1))
        log(f'  Ĉ não decrescente em DET ≤ 0,70 ≤ 0,80 ≤ 0,90 ≤ 0,95: {sum(mono)}/{len(mono)} instâncias; '
            f'Ĉ(0,95)/Ĉ(0,70) − 1: mediana {np.median(incl):.2f}%, máx. {np.max(incl):.2f}%')
        gaps = []; cond = []; gaps_cond = []; gaps_nao = []
        for (nome, sd), d in runs.items():
            if 'CAL' not in d:
                continue
            cands = [(nome2, x) for (nome2, _), dd in runs.items() if nome2 == nome for k, x in dd.items() if isinstance(x, dict)]
            best = min((x for _, x in cands), key=lambda x: x['f1'])
            g = 100 * (d['CAL']['f1'] / best['f1'] - 1)
            gaps.append(g)
            c_ok = best['q'] >= d['q_cal'] - 1e-9
            cond.append(c_ok); (gaps_cond if c_ok else gaps_nao).append(g)
        log(f'  perda do CAL contra o melhor plano certificado conhecido (todas as sementes e métodos): '
            f'mediana {np.median(gaps):.2f}%, P90 {np.percentile(gaps, 90):.2f}%')
        log(f'  margem do melhor plano ≥ degrau do CAL (condição da Prop. 3) em {100 * np.mean(cond):.1f}% das execuções; '
            f'perda mediana {np.median(gaps_cond) if gaps_cond else float("nan"):.2f}% quando vale e '
            f'{np.median(gaps_nao) if gaps_nao else float("nan"):.2f}% quando não vale')
        qs = [x['q'] for d in runs.values() for k, x in d.items() if k == 'C-SIMC']
        log(f'  margem uniforme dos planos escolhidos do SIM+C: q mediano {np.median(qs):.3f} (P10 {np.percentile(qs, 10):.3f}, P90 {np.percentile(qs, 90):.3f})')
        qc = [d['q_cal'] for d in runs.values() if 'q_cal' in d]
        log('  degrau escolhido pelo CAL: ' + ', '.join(f'{q:.2f}: {100 * np.mean(np.isclose(qc, q)):.0f}%' for _, q in ESCADA))
        v2 = {}
        for a, qa in ESCADA:
            L = V2.get(a, [])
            if L:
                cb = np.array([x[0] for x in L]); rk = np.array([x[1] for x in L]); M = V2[a + '_max']
                v2[a] = dict(n=len(L), valida=float(np.mean(cb + 0.0095 >= rk)), cota_seg=float(np.mean(cb <= 0.001)),
                             risco_seg=float(np.mean(rk <= 0.001)), max_med=float(np.median([x[0] for x in M])),
                             max_ok=float(np.mean([x[0] <= x[1] + 0.0095 for x in M])))
        resumo[grupo] = dict(V2=v2, V1={a: dict(n=len(L), ok_inf=float(np.mean([x['ok_inf'] for x in L])),
                                         ok_sup=float(np.mean([x['ok_sup'] for x in L])),
                                         P=float(np.median([x['P'] for x in L])), prod=float(np.median([x['prod'] for x in L])),
                                         mn=float(np.median([x['mn'] for x in L])), ncrit=float(np.median([x['ncrit'] for x in L])))
                                     for a, L in V1.items()},
                             V3=dict(mono=int(sum(mono)), n_inst=len(mono), incl_med=float(np.median(incl)),
                                     incl_max=float(np.max(incl)), gap_med=float(np.median(gaps)),
                                     gap_p90=float(np.percentile(gaps, 90)), cond=float(np.mean(cond)),
                                     gap_cond=float(np.median(gaps_cond)) if gaps_cond else None,
                                     gap_nao=float(np.median(gaps_nao)) if gaps_nao else None,
                                     q_sim=float(np.median(qs)),
                                     degraus={f'{q:.2f}': float(np.mean(np.isclose(qc, q))) for _, q in ESCADA}))
    json.dump(resumo, open(os.path.join(OUT, 'resumo_teoria_buffer.json'), 'w', encoding='utf-8'), indent=1)
    rel.close()
    tabela(resumo)


ROT = {'C-DETC': r'\mDETC{}', 'C-BUF70C': r'\mBUF{0.70}', 'C-BUF80C': r'\mBUF{0.80}', 'C-BUF90C': r'\mBUF{0.90}',
       'C-BUF95C': r'\mBUF{0.95}', 'CAL': r'\mCAL{}', 'C-SIMC': r'\mSIM{}'}


def tabela(resumo):
    """Tabela S9 do material suplementar (tables/tab_teoria.tex), a partir do resumo."""
    pc = r'\%'
    L = [r'\begin{table}[htbp]', r'\centering',
         r'\caption{Empirical check of the properties of Section~4.3 of the paper on the chosen plan (cheapest certified '
         r'solution) of every run, with the 1{,}000 common out-of-sample scenarios. Sandwich: share of plans with '
         r'$\prod_j\pi_j\le P_{\mathrm{on}}\le\min_j\pi_j$ (lower bound with a Monte Carlo tolerance of 1.96 standard errors); '
         r'critical: customers with $\pi_j<0.999$ (median per plan); bound valid: share of customers whose estimated risk does '
         r'not exceed the bound of Proposition~3 (tolerance 0.0095, one standard error at risk 0.1); safe: share of customers '
         r'whose bound, and whose estimated risk, is at most 0.001; max.\ risk: median over plans of the largest customer risk.}',
         r'\label{tab:teoria}', r'\footnotesize', r'\setlength{\tabcolsep}{3.5pt}',
         r'\begin{tabular}{@{}l l r r r r r r r r r@{}}', r'\toprule',
         r' & & & \multicolumn{4}{c}{Proposition 2} & \multicolumn{4}{c}{Proposition 3} \\',
         r'\cmidrule(lr){4-7}\cmidrule(l){8-11}',
         r'Set & Method & Plans & Sandwich & $P_{\mathrm{on}}$ & $\prod_j\pi_j$ & Critical & Bound valid & Safe (bound) & Safe (risk) & Max.\ risk \\',
         r'\midrule']
    for g, rot_g in (('benchmark', 'Benchmark'), ('SP', r'S\~ao Paulo')):
        R = resumo[g]
        for k, a in enumerate(['C-DETC', 'C-BUF70C', 'C-BUF80C', 'C-BUF90C', 'C-BUF95C', 'CAL', 'C-SIMC']):
            v = R['V1'][a]; w = R['V2'].get(a)
            sw = 100 * min(v['ok_inf'], v['ok_sup'])
            p3 = (f"{100 * w['valida']:.1f}{pc} & {100 * w['cota_seg']:.1f}{pc} & {100 * w['risco_seg']:.1f}{pc} & {w['max_med']:.3f}"
                  if w else '-- & -- & -- & --')
            L.append(f"{rot_g if k == 0 else ''} & {ROT[a]} & {v['n']} & {sw:.1f}{pc} & {v['P']:.3f} & {v['prod']:.3f} & "
                     f"{v['ncrit']:.0f} & {p3} " + r'\\')
        if g == 'benchmark':
            L.append(r'\midrule')
    L += [r'\bottomrule', r'\end{tabular}']
    v3 = []
    for g, rot_g in (('benchmark', 'benchmark'), ('SP', r'S\~ao Paulo')):
        x = resumo[g]['V3']
        v3.append(f"{rot_g}: $\\hat C$ non-decreasing in {x['mono']} of {x['n_inst']} instances, median "
                  f"$\\hat C(0.95)/\\hat C(0.70)-1$ of {x['incl_med']:.2f}{pc} (max.\\ {x['incl_max']:.2f}{pc}); gap of "
                  f"\\mCAL{{}} to the best certified plan {x['gap_med']:.2f}{pc} (median; 90th percentile {x['gap_p90']:.2f}{pc}), "
                  f"{x['gap_cond']:.2f}{pc} when the margin condition holds ({100 * x['cond']:.0f}{pc} of the runs) and "
                  f"{x['gap_nao']:.2f}{pc} otherwise")
    L += [r'\par\smallskip\parbox{\textwidth}{\footnotesize Lemma~1 and Proposition~4: ' + '; '.join(v3) + '.}', r'\end{table}']
    arq = os.path.join(BASE, 'pesquisa', 'latex_ejor_c', 'tables', 'tab_teoria.tex')
    open(arq, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('tabela:', arq)


if __name__ == '__main__':
    main()
