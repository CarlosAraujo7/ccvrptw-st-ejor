#!/usr/bin/env python3
"""
Equivalência e velocidade da busca local por variação de custo (09/10/2026, diário
Entrada 86). _passes_delta deve tomar EXATAMENTE as mesmas decisões que _passes_ref
(implementação original): mesmas rotas finais para as mesmas rotas de entrada e a
mesma semente. Testa rotas do Split de permutações aleatórias, com tempos nominais e
com folga (buffer 0,95), em instâncias de janela estreita e larga, n = 25, 50 e 100.
Mede o tempo de CPU (time.process_time, menos sensível à carga da máquina) das duas.
"""
import os, random, sys, time

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(BASE, 'src'))
import nsga2_simheuristic as ns   # noqa: E402
import ccvrp_c as cc              # noqa: E402

INSTS = ['CCVRPTW-ST_RC1_100_MEDIUM', 'CCVRPTW-ST_R2_100_HIGH', 'CCVRPTW-ST_C1_050_HIGH',
         'CCVRPTW-ST_R1_050_LOW', 'CCVRPTW-ST_RC2_025_MEDIUM', 'CCVRPTW-ST_C2_025_LOW']
iguais = difer = 0; t_ref = t_delta = 0.0
for nome in INSTS:
    inst = ns.load_instance(os.path.join(BASE, 'instances', 'generated', nome + '.json'))
    for buf in (None, 0.95):
        ctx = cc.Contexto(inst, 0, buffer_q=buf, decoder='split', p_ls=0.2)
        n = len(inst['customers'])
        for t in range(40):
            rng = random.Random(1000 + t); ch = list(range(1, n + 1)); rng.shuffle(ch)
            R, _ = cc.split_decode(ch, ctx)
            a = time.process_time(); r1 = cc._passes_ref(R, ctx, random.Random(t), 2); t_ref += time.process_time() - a
            a = time.process_time(); r2 = cc._passes_delta(R, ctx, random.Random(t), 2); t_delta += time.process_time() - a
            if r1 == r2:
                iguais += 1
            else:
                difer += 1
                c1 = sum(cc._custo_rota(r, ctx) for r in r1); c2 = sum(cc._custo_rota(r, ctx) for r in r2)
                print(f'DIFERENTE: {nome} buf={buf} t={t}  custo ref {c1:.4f}  delta {c2:.4f}', flush=True)
print(f'casos idênticos: {iguais}  diferentes: {difer}')
print(f'tempo de CPU: referência {t_ref:.1f} s  |  delta {t_delta:.1f} s  |  aceleração {t_ref / t_delta:.1f}x')
