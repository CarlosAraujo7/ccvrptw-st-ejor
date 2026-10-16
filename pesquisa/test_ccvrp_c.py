#!/usr/bin/env python3
"""Testes do Caminho C (src/ccvrp_c.py) — 03/10/2026, diário Entrada 74."""
import os, random, sys, time
import numpy as np

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(BASE, 'src'))
import nsga2_simheuristic as ns   # noqa: E402
import thermal_route as tr        # noqa: E402
import ccvrp_c as cc              # noqa: E402

inst = ns.load_instance(os.path.join(BASE, 'instances', 'generated', 'CCVRPTW-ST_R1_050_HIGH.json'))
dist, ttime = ns.build_matrices(inst)
cust = inst['customers']
falhas = 0


def check(cond, msg):
    global falhas
    print(('OK   ' if cond else 'FALHA ') + msg)
    falhas += 0 if cond else 1


# 1) saída adiada: sem atraso nominal, duração não aumenta, Δ>0 em alguma rota
r = random.Random(5); rotas = []
for _ in range(20):
    ch = list(range(1, 51)); r.shuffle(ch); rotas += ns.decode(ch, inst, ttime)[0]
pior_atraso = 0.0; dur_ok = True; algum_delta = False
for rt in rotas:
    d0, dur = cc.plano_rota(rt, inst, ttime)
    _, dur0 = (0.0, None)
    t = 0.0; prev = 0
    for c in rt:
        t += ttime[prev][c]; t = max(t, cust[c - 1]['ready_time_min']) + cust[c - 1]['service_time_min']; prev = c
    dur0 = t + ttime[prev][0]
    t = d0; prev = 0
    for c in rt:
        t += ttime[prev][c]; pior_atraso = max(pior_atraso, t - cust[c - 1]['due_date_min'])
        t = max(t, cust[c - 1]['ready_time_min']) + cust[c - 1]['service_time_min']; prev = c
    dur_ok &= dur <= dur0 + 1e-9
    algum_delta |= d0 > 1e-6
check(pior_atraso <= 1e-9, f"saída adiada não cria atraso nominal (pior {pior_atraso:.2e} min)")
check(dur_ok, "duração com saída adiada ≤ duração saindo em t=0")
check(algum_delta, "há rotas com saída adiada (Δ>0)")

# 2) pontualidade: tt=1 → pontual; e bate com o atraso do modelo térmico nos mesmos cenários
K = 500
sc = tr.sortear_cenarios(inst, K, np.random.default_rng(3), tempo_viagem=True)
ok_nom = all(cc.pontual_rota(rt, inst, ttime, np.ones((3, 51)), cc.plano_rota(rt, inst, ttime)[0]).all() for rt in rotas)
check(ok_nom, "com tempos nominais, toda rota é pontual")
iguais = True
for rt in rotas[:30]:
    d0 = cc.plano_rota(rt, inst, ttime)[0]
    a = cc.pontual_rota(rt, inst, ttime, sc['tt'], d0, 5.0)
    b = (tr.route_quality(rt, inst, ttime, sc, detalhe=True, t0_min=d0)['late'] <= 5.0).all(1)
    iguais &= bool((a == b).all())
check(iguais, "pontual_rota bate com o atraso calculado por thermal_route (mesmos cenários)")

# 3) quantil da folga de segurança
for cv, q in ((0.05, 0.8), (0.2, 0.9)):
    amostra = tr._lognormal_media1(np.random.default_rng(1), cv, (400000,))
    m = cc.multiplicador_quantil(cv, q)
    check(abs(np.mean(amostra <= m) - q) < 0.003, f"quantil {q} com CV {cv}: multiplicador {m:.4f}")

# 4) dominância com restrição
A = cc.IndividualC(chromosome=[1]); B = cc.IndividualC(chromosome=[1])
A.f_cost, A.f_qloss, A.f_co2, A.viol = 10, 1, 1, 0.0
B.f_cost, B.f_qloss, B.f_co2, B.viol = 5, 0.5, 0.5, 0.02
check(A.dominates(B) and not B.dominates(A), "viável domina inviável mesmo com objetivos piores")
A.viol = 0.01
check(A.dominates(B) and not B.dominates(A), "entre inviáveis, menor violação domina")
A.viol = B.viol = 0.0
check(B.dominates(A) and not A.dominates(B), "entre viáveis, vale Pareto")

# 5) execuções curtas de cada variante (n=50, 30 gerações)
for nome, kw in (('DET', {}), ('BUF90', {'buffer_q': 0.9}), ('SIM', {'mc_rel': 30})):
    t0 = time.time()
    pf, _ = cc.run_nsga2_c(inst, pop_size=60, n_gen=60, seed=1, **kw)
    viav = sum(1 for x in pf if x.viol == 0)
    niv = np.mean([np.mean(x.dep_block[:len(x.routes)]) for x in pf])
    pon = np.mean([x.p_on for x in pf]) if nome == 'SIM' else float('nan')
    check(len(pf) > 0 and all(np.isfinite(x.f_qloss) for x in pf),
          f"{nome}: frente {len(pf)} sol., {viav} sem violação (só SIM avalia), P(pontual) média {pon:.2f}, "
          f"nível de saída médio {niv:.2f}/4, custo mín {min(x.f_cost for x in pf):.0f}, {time.time() - t0:.1f}s")

# 6) B2 curto
pf, _ = cc.run_nsga2_c(inst, pop_size=40, n_gen=10, seed=2, mc_rel=30, use_multizone=True, n_zones=2)
check(len(pf) > 0, f"B2 Z=2 com SIM: frente com {len(pf)} soluções")

print(f"\n{falhas} falha(s)")
sys.exit(1 if falhas else 0)
