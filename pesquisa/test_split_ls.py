#!/usr/bin/env python3
"""Testes do Split e da busca local do Caminho C (04/10/2026, diário Entrada 75)."""
import itertools, os, random, sys, time
import numpy as np

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(BASE, 'src'))
import nsga2_simheuristic as ns   # noqa: E402
import ccvrp_c as cc              # noqa: E402

falhas = 0


def check(cond, msg):
    global falhas
    print(('OK   ' if cond else 'FALHA ') + msg, flush=True)
    falhas += 0 if cond else 1


def custo(rotas, ctx):
    return sum(cc._custo_rota(r, ctx) for r in rotas)


inst = ns.load_instance(os.path.join(BASE, 'instances', 'generated', 'CCVRPTW-ST_R1_050_HIGH.json'))
ctx = cc.Contexto(inst, 0, decoder='split', p_ls=0.0)
rng = random.Random(1)

# 1) Split ótimo: força bruta sobre todas as divisões em segmentos consecutivos (8 clientes)
ok_otimo = True
for _ in range(30):
    perm = rng.sample(range(1, 51), 8)
    rotas, _ = cc.split_decode(perm, ctx)
    melhor = np.inf
    for cortes in itertools.product([0, 1], repeat=7):
        segs, ini = [], 0
        for k, c in enumerate(cortes, 1):
            if c:
                segs.append(perm[ini:k]); ini = k
        segs.append(perm[ini:])
        if all(cc._rota_viavel(sg, ctx) for sg in segs):
            melhor = min(melhor, custo(segs, ctx))
    ok_otimo &= abs(custo(rotas, ctx) - melhor) < 1e-6
check(ok_otimo, "Split = ótimo da força bruta em 30 permutações de 8 clientes")

# 2) busca local: não piora, rotas viáveis, todos os clientes presentes; Split reproduz
ok_bl = ok_rep = True; ganhos = []
for _ in range(20):
    ch = list(range(1, 51)); rng.shuffle(ch)
    r0, _ = cc.split_decode(ch, ctx)
    r1 = cc.busca_local(r0, ctx, rng)
    ok_bl &= (custo(r1, ctx) <= custo(r0, ctx) + 1e-9 and all(cc._rota_viavel(r, ctx) for r in r1)
              and sorted(c for r in r1 for c in r) == sorted(c for r in r0 for c in r))
    r2, _ = cc.split_decode([c for r in r1 for c in r], ctx)
    ok_rep &= custo(r2, ctx) <= custo(r1, ctx) + 1e-9
    ganhos.append(1 - custo(r1, ctx) / custo(r0, ctx))
check(ok_bl, f"busca local não piora, mantém viabilidade e clientes (ganho médio {np.mean(ganhos):.1%})")
check(ok_rep, "Split da concatenação das rotas melhoradas tem custo <= o delas")

# 3) gap contra os ótimos exatos (melhor custo da frente, 5 sementes, DET)
OTIMOS = {5: (485.84, 'provado'), 10: (879.64, 'provado'), 15: (1464.02, 'incumbente, não provado'),
          20: (1281.15, 'provado')}
for n, (ot, st) in OTIMOS.items():
    instv = ns.load_instance(os.path.join(BASE, 'instances', 'validacao_exata', f'CCVRPTW-ST_valexata_C1_{n:03d}_LOW.json'))
    res = {}
    for nome, kw in (('firstfit', {}), ('split', {'decoder': 'split'}), ('split+BL', {'decoder': 'split', 'p_ls': 0.5})):
        best = [min(x.f_cost for x in cc.run_nsga2_c(instv, seed=s, **kw)[0]) for s in range(5)]
        res[nome] = (min(best), np.mean(best))
    linha = ' | '.join(f"{k}: melhor {v[0]:.2f} ({100 * (v[0] / ot - 1):+.1f}%), média {100 * (v[1] / ot - 1):+.1f}%"
                       for k, v in res.items())
    print(f"   n={n:2d} ótimo {ot:.2f} ({st}) | {linha}", flush=True)
    if n == 20:
        check(res['split+BL'][0] / ot - 1 < 0.05, "n=20: gap do melhor com Split+busca local < 5% (meta do plano)")

# 4) tempo por execução em n=100 (DET, 200 gerações)
inst100 = ns.load_instance(os.path.join(BASE, 'instances', 'generated', 'CCVRPTW-ST_R1_100_HIGH.json'))
for nome, kw in (('firstfit', {}), ('split+BL p=0,5', {'decoder': 'split', 'p_ls': 0.5})):
    t0 = time.time(); pf, _ = cc.run_nsga2_c(inst100, seed=0, **kw)
    print(f"   n=100 {nome}: {time.time() - t0:.1f}s, custo mín {min(x.f_cost for x in pf):.1f}", flush=True)

print(f"\n{falhas} falha(s)")
sys.exit(1 if falhas else 0)
