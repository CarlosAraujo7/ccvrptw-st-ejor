#!/usr/bin/env python3
"""
Refinamento por simulação depois da busca com buffer (11/10/2026, diário Entrada 88).

Ideia: a busca com buffer (qualquer resolvedor determinístico) entrega planos com
pontualidade DE SOBRA (o buffer 0,95 entrega ≈ 0,98 para α = 0,90), paga em qualidade e
emissões (partidas cedo) e, possivelmente, em custo (rotas montadas com tempos
inflados). O refinamento usa a simulação para converter essa sobra:
  1. refinamento de CUSTO — busca local (realocação, troca, 2-opt* entre vizinhos
     granulares e eliminação de rotas) com viabilidade NOMINAL; um movimento que reduz
     o custo só é aceito se a pontualidade conjunta, na amostra de refinamento, ficar
     ≥ α + margem (rotas alteradas saem no instante 0, o mais pontual possível);
  2. ajuste das PARTIDAS — atrasa, guloso, a saída de cada rota numa grade de 17
     pontos entre 0 e Δmax nominal (maior atraso sem atraso nominal nem rota mais longa),
     escolhendo a cada passo o atraso que mais reduz a perda de qualidade nominal,
     enquanto a pontualidade conjunta na amostra de refinamento ficar ≥ α + margem;
  3. CERTIFICAÇÃO em 500 cenários novos (independentes dos de refinamento) como
     PORTEIRA: o plano refinado só substitui o original se P_cert >= α (versão pontual)
     ou P_cert >= α + 1,645·√(α(1−α)/500) (versão LCB); avaliação final nos 1.000
     cenários comuns de analisar_c.py (nunca usados no refinamento nem na porteira).
Margem = 1,645·√(α(1−α)/K_ref): limite inferior de confiança unilateral de 95%.

Protocolo fixado ANTES de ver resultados (Lei Geral I.3): ponto de partida = a escolha
do decisor (solução mais barata certificada) de cada execução; comparação com as
escolhas do decisor de SIM+C, BUF0.95+C e CAL nas mesmas execuções, nos mesmos 1.000
cenários; métricas: cumpre α fora da amostra, custo, perda de qualidade esperada, CO2.
Não altera src/. Uso: python pesquisa/refinamento_simulacao.py --res-dir results_c_v1_cindom
  [--niveis MEDIUM HIGH] [--tamanhos 050] [--sementes 0 1 2] [--partida C-BUF95C]
"""
import argparse, glob, json, os, random, sys, time
from collections import defaultdict
import numpy as np

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(BASE, 'src')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nsga2_simheuristic as ns   # noqa: E402
import thermal_route as tr        # noqa: E402
import ccvrp_c as cc              # noqa: E402
import analisar_c as ac           # noqa: E402

ALPHA = 0.90; K_REF = 500; K_CERT = 500
MARGEM = 1.645 * np.sqrt(ALPHA * (1 - ALPHA) / K_REF)
ALVO = ALPHA + MARGEM
# Porteira (Entrada 88, definida no conjunto de desenvolvimento = execuções v1): o plano
# refinado só substitui o original se passar na certificação independente; senão fica o
# original. Duas versões: estimativa pontual (P_cert >= α) e limite de confiança (LCB).
ALVO_CERT_LCB = ALPHA + 1.645 * np.sqrt(ALPHA * (1 - ALPHA) / K_CERT)
GRADE = 16


class Plano:
    """Rotas + partidas com caches por rota (pontualidade na amostra, qualidade, duração)."""

    def __init__(self, ctx, tt_ref):
        self.ctx = ctx; self.inst = ctx.inst; self.tt = tt_ref
        fl = self.inst['fleet']; co2p = self.inst.get('co2_params', {})
        self.a_co2 = co2p.get('alpha_kg_per_km', 0.21); self.b_co2 = co2p.get('beta_kg_per_kWh', 0.092)
        self.P_ref = co2p.get('P_refrigeration_kW', 3.5); self.K = fl['num_vehicles']
        self.cust = self.inst['customers']
        self.c_on, self.c_q = {}, {}

    def on(self, r, d0):
        k = (tuple(r), round(d0, 6))
        if k not in self.c_on:
            self.c_on[k] = cc.pontual_rota(r, self.inst, self.ctx.ttime, self.tt, d0)
        return self.c_on[k]

    def q(self, r, d0):
        k = (tuple(r), round(d0, 6))
        if k not in self.c_q:
            self.c_q[k] = tr.route_quality(r, self.inst, self.ctx.ttime, self.ctx.nom,
                                           per_customer_kinetics=self.ctx.cin, t0_min=d0)[0]
        return self.c_q[k]

    def valor(self, r):
        return sum(self.cust[c - 1].get('load_value_BRL', 1.0) for c in r)

    def objetivos(self, R, D):
        custo = sum(cc._custo_rota(r, self.ctx) for r in R) + ns.EXTRA_VEHICLE_COST * max(0, len(R) - self.K)
        v = [self.valor(r) for r in R]
        f2 = 1.0 - sum(self.q(r, d) * w for r, d, w in zip(R, D, v)) / sum(v)
        f3 = 0.0
        for r, d in zip(R, D):
            km = sum(self.ctx.dist[a][b] for a, b in zip([0] + r, r + [0]))
            f3 += self.a_co2 * km + self.b_co2 * self.P_ref * cc.duracao(r, self.inst, self.ctx.ttime, d) / 60.0
        return custo, f2, f3

    def p_conj(self, R, D):
        on = np.ones(self.tt.shape[0], dtype=bool)
        for r, d in zip(R, D):
            on &= self.on(r, d)
        return float(on.mean())


def refinar_custo(R, D, pl, rng, max_passes=3):
    """Busca local de custo com aceitação condicionada à pontualidade simulada."""
    ctx = pl.ctx; R = [r[:] for r in R]; D = list(D)
    for _ in range(max_passes):
        melhorou = False
        pos = {c: (ri, k) for ri, r in enumerate(R) for k, c in enumerate(r)}
        ordem = list(pos); rng.shuffle(ordem)
        for u in ordem:
            for v in ctx.viz[u]:
                if v not in pos:
                    continue
                ru, iu = pos[u]; rv, iv = pos[v]
                cands = []
                A = R[ru][:iu] + R[ru][iu + 1:]
                if ru == rv:
                    B = A[:]; B.insert(B.index(v) + 1, u); cands.append({ru: B})
                else:
                    cands.append({ru: A, rv: R[rv][:iv + 1] + [u] + R[rv][iv + 1:]})
                    A2 = R[ru][:]; B2 = R[rv][:]; A2[iu] = v; B2[iv] = u; cands.append({ru: A2, rv: B2})
                    cands.append({ru: R[ru][:iu + 1] + R[rv][iv + 1:], rv: R[rv][:iv + 1] + R[ru][iu + 1:]})
                aceito = False
                for mv in cands:
                    antes = sum(cc._custo_rota(R[k], ctx) for k in mv)
                    depois = sum(cc._custo_rota(r, ctx) for r in mv.values())
                    if depois >= antes - 1e-9 or not all(cc._rota_viavel(r, ctx) for r in mv.values() if r):
                        continue
                    R2 = [mv.get(k, r) for k, r in enumerate(R)]; D2 = [0.0 if k in mv else d for k, d in enumerate(D)]
                    par = [(r, d) for r, d in zip(R2, D2) if r]
                    R2 = [r for r, _ in par]; D2 = [d for _, d in par]
                    if pl.p_conj(R2, D2) >= ALVO:
                        R, D = R2, D2; aceito = melhorou = True
                        pos = {c: (ri, k) for ri, r in enumerate(R) for k, c in enumerate(r)}
                        break
                if aceito:
                    break
        # eliminação de rotas (mesmo operador do motor), com a condição de pontualidade
        while True:
            novo = cc._eliminar_rota(R, ctx)
            if novo is None:
                break
            Dn = [0.0] * len(novo)
            if pl.p_conj(novo, Dn) >= ALVO:
                R, D = novo, Dn; melhorou = True
            else:
                break
        if not melhorou:
            break
    return R, D


def ajustar_partidas(R, D, pl):
    """Atrasos gulosos na grade [0, Δmax nominal], mantendo a pontualidade ≥ ALVO."""
    ctx = pl.ctx
    dmax = [cc.delta_max(r, pl.inst, ctx.ttime) for r in R]
    grade = [[dm * j / GRADE for j in range(GRADE + 1)] for dm in dmax]
    j = [max(0, max(k for k, g in enumerate(gr) if g <= d + 1e-9)) for gr, d in zip(grade, D)]
    D = [grade[i][j[i]] for i in range(len(R))]
    # se a partida inicial não cumpre o alvo, recua (as rotas mais arriscadas primeiro) até cumprir
    while pl.p_conj(R, D) < ALVO and any(x > 0 for x in j):
        i = max((i for i in range(len(R)) if j[i] > 0), key=lambda i: 1 - pl.on(R[i], D[i]).mean())
        j[i] -= 1; D[i] = grade[i][j[i]]
    v = [pl.valor(r) for r in R]
    while True:
        melhor = None
        for i in range(len(R)):
            if j[i] >= GRADE:
                continue
            dn = grade[i][j[i] + 1]
            ganho = (pl.q(R[i], dn) - pl.q(R[i], D[i])) * v[i]
            if ganho <= 1e-12:
                continue
            D2 = D[:]; D2[i] = dn
            if pl.p_conj(R, D2) >= ALVO and (melhor is None or ganho > melhor[0]):
                melhor = (ganho, i, dn)
        if melhor is None:
            break
        _, i, dn = melhor; j[i] += 1; D[i] = dn
    return D


def avaliar_oos(R, D, inst_nome, ctx):
    inst, ttime, sc, _ = ac._ctx(inst_nome, 1)
    on = np.ones(ac.K, dtype=bool); acc = np.zeros(ac.K); tot = 0.0
    for r, d in zip(R, D):
        on &= cc.pontual_rota(r, inst, ttime, sc['tt'], d)
        q = tr.route_quality(r, inst, ttime, sc, per_customer_kinetics=ctx.cin, t0_min=d)
        v = sum(inst['customers'][c - 1].get('load_value_BRL', 1.0) for c in r)
        acc += q * v; tot += v
    return float(on.mean()), float(1 - (acc / tot).mean())


def escolha_decisor(path_sol, path_oos):
    S = json.load(open(path_sol, encoding='utf-8')); O = {o['sol_id']: o for o in json.load(open(path_oos, encoding='utf-8'))['solutions']}
    cert = [s for s in S if s['feasible'] and (s.get('p_cert') or 0) >= ALPHA]
    if not cert:
        return None
    s = min(cert, key=lambda s: s['f1_cost_BRL']); o = O[s['sol_id']]
    return dict(s=s, f1=s['f1_cost_BRL'], p_oos=o['p_on'], q_oos=o['q_oos'], f3=s['f3_co2_kg'], p_cert=s['p_cert'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res-dir', default='results_c')
    ap.add_argument('--niveis', nargs='+', default=['MEDIUM', 'HIGH'])
    ap.add_argument('--tamanhos', nargs='+', default=['050'])
    ap.add_argument('--sementes', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--partida', default='C-BUF95C')
    ap.add_argument('--saida', default=os.path.join(BASE, 'pesquisa', 'resultados_refinamento'))
    ap.add_argument('--caso-sp', action='store_true', help='instâncias do caso São Paulo (matrizes viárias)')
    a = ap.parse_args()
    inst_dir = os.path.join(BASE, 'instances', 'generated')
    if a.caso_sp:
        sys.path.insert(0, os.path.join(BASE, 'pesquisa', 'caso_sp'))
        import rodar_caso_sp
        rodar_caso_sp.aplicar_matrizes(); inst_dir = rodar_caso_sp.INST_DIR
    RD = os.path.join(BASE, a.res_dir)
    os.makedirs(a.saida, exist_ok=True)
    linhas = []
    ESCADA = ['C-BUF70C', 'C-BUF80C', 'C-BUF90C', 'C-BUF95C']   # --partida CAL: a mais barata certificada da escada
    pasta_nomes = 'C-BUF95C' if a.partida == 'CAL' else a.partida
    for nome in sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(RD, pasta_nomes, '*_solutions.json'))}):
        _, tam, niv = nome.split('_')[-3:]
        if niv not in a.niveis or tam not in a.tamanhos:
            continue
        inst = ns.load_instance(os.path.join(inst_dir, nome + '.json'))
        ctx = cc.Contexto(inst, 0, decoder='split', p_ls=0.0)
        cv = inst['thermal_profile']['sigma_t_pct']; n = len(inst['customers'])
        for sd in a.sementes:
            if a.partida == 'CAL':
                cands = []
                for deg in ESCADA:
                    arq = os.path.join(RD, deg, f'{nome}__{deg}__seed{sd}_solutions.json')
                    if os.path.exists(arq):
                        e = escolha_decisor(arq, arq.replace('_solutions.json', '_oosC.json'))
                        if e:
                            cands.append(e)
                esc = min(cands, key=lambda e: e['f1']) if cands else None
            else:
                arq = os.path.join(RD, a.partida, f'{nome}__{a.partida}__seed{sd}_solutions.json')
                if not os.path.exists(arq):
                    continue
                esc = escolha_decisor(arq, arq.replace('_solutions.json', '_oosC.json'))
            if esc is None:
                continue
            tt_ref = tr._lognormal_media1(np.random.default_rng(31337 + 97 * sd + sum(map(ord, nome))), cv, (K_REF, n + 1))
            tt_cer = tr._lognormal_media1(np.random.default_rng(4242 + 89 * sd + sum(map(ord, nome))), cv, (K_CERT, n + 1))
            pl = Plano(ctx, tt_ref)
            t0 = time.process_time()
            R0 = [r[:] for r in esc['s']['routes']]; D0 = list(esc['s']['partida_min'])
            R1, D1 = refinar_custo(R0, D0, pl, random.Random(sd))
            D2 = ajustar_partidas(R1, D1, pl)
            t_ref = time.process_time() - t0
            p_cert = Plano(ctx, tt_cer).p_conj(R1, D2)
            f1, f2, f3 = pl.objetivos(R1, D2)
            p_oos, q_oos = avaliar_oos(R1, D2, nome, ctx)
            linha = dict(instance=nome, seed=sd, nivel=niv,
                         base_f1=esc['f1'], base_p_oos=esc['p_oos'], base_q_oos=esc['q_oos'], base_f3=esc['f3'],
                         ref_f1=f1, ref_p_cert=p_cert, ref_p_oos=p_oos, ref_q_oos=q_oos, ref_f3=f3, cpu_s=t_ref)
            for suf, lim in (('fin', ALPHA), ('finl', ALVO_CERT_LCB)):
                ok = p_cert >= lim
                linha[f'{suf}_aceito'] = bool(ok)
                for k in ('f1', 'p_oos', 'q_oos', 'f3'):
                    linha[f'{suf}_{k}'] = linha[f'ref_{k}'] if ok else linha[f'base_{k}']
            for outro in ('C-SIMC', 'C-BUF90C', 'C-BUF70C', 'C-BUF80C'):
                o = os.path.join(RD, outro, f'{nome}__{outro}__seed{sd}_solutions.json')
                e = escolha_decisor(o, o.replace('_solutions.json', '_oosC.json')) if os.path.exists(o) else None
                if e:
                    linha.update({f'{outro}_f1': e['f1'], f'{outro}_p_oos': e['p_oos'], f'{outro}_q_oos': e['q_oos'], f'{outro}_f3': e['f3']})
            linhas.append(linha)
            print(f"{nome} s{sd}: base f1 {esc['f1']:.1f} P {esc['p_oos']:.3f} q {100*esc['q_oos']:.2f}% | "
                  f"refinado f1 {f1:.1f} P {p_oos:.3f} (cert {p_cert:.3f}) q {100*q_oos:.2f}% | "
                  f"SIM f1 {linha.get('C-SIMC_f1', float('nan')):.1f} P {linha.get('C-SIMC_p_oos', float('nan')):.3f} "
                  f"q {100*linha.get('C-SIMC_q_oos', float('nan')):.2f}% | {t_ref:.1f} s", flush=True)
    json.dump(linhas, open(os.path.join(a.saida, f'refinamento_{a.partida}.json'), 'w', encoding='utf-8'), indent=1)
    resumo(linhas, a.saida, a.partida)


def resumo(L, saida, partida):
    def med(xs):
        xs = [x for x in xs if np.isfinite(x)]
        return np.median(xs) if xs else np.nan
    out = [f'REFINAMENTO POR SIMULAÇÃO a partir da escolha do decisor de {partida} ({len(L)} execuções); avaliação em 1.000 cenários comuns']
    for nome, f1, p, q, f3 in (('base (' + partida + ')', 'base_f1', 'base_p_oos', 'base_q_oos', 'base_f3'),
                               ('refinado', 'ref_f1', 'ref_p_oos', 'ref_q_oos', 'ref_f3'),
                               ('SIM+C', 'C-SIMC_f1', 'C-SIMC_p_oos', 'C-SIMC_q_oos', 'C-SIMC_f3')):
        LL = [x for x in L if f1 in x]
        out.append(f'  {nome:22s} cumpre {100*np.mean([x[p] >= ALPHA for x in LL]):5.1f}%  P média {np.mean([x[p] for x in LL]):.3f}  '
                   f'custo médio {np.mean([x[f1] for x in LL]):8.1f}  perda {100*np.mean([x[q] for x in LL]):.3f}%  CO2 {np.mean([x[f3] for x in LL]):.1f}')
    LL = [x for x in L if 'C-SIMC_f1' in x]
    for rot, a_, b_ in (('refinado vs base', 'ref', 'base'), ('refinado vs SIM+C', 'ref', 'C-SIMC')):
        dc = [100 * (x[f'{a_}_f1'] / x[f'{b_}_f1'] - 1) for x in LL]
        dq = [100 * (x[f'{a_}_q_oos'] - x[f'{b_}_q_oos']) for x in LL]
        de = [100 * (x[f'{a_}_f3'] / x[f'{b_}_f3'] - 1) for x in LL]
        out.append(f'  {rot:22s} Δcusto mediana {med(dc):+.2f}% média {np.mean(dc):+.2f}%  Δperda mediana {med(dq):+.3f} p.p.  ΔCO2 mediana {med(de):+.2f}%')
    out.append(f'  tempo de CPU do refinamento: média {np.mean([x["cpu_s"] for x in L]):.1f} s por execução')
    txt = '\n'.join(out); print('\n' + txt)
    open(os.path.join(saida, f'relatorio_refinamento_{partida}.txt'), 'w', encoding='utf-8').write(txt + '\n')


if __name__ == '__main__':
    main()
