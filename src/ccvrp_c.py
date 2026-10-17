#!/usr/bin/env python3
"""
CCVRPTW-ST — Caminho C: "a qualidade se planeja, a pontualidade se simula"
==========================================================================
03/10/2026 — pesquisa/plano_caminho_C.md (aprovado pelo pesquisador com as
recomendações D1-D6), diário Entradas 73-74.

Módulo NOVO e separado: nada aqui altera nsga2_simheuristic.py; os resultados
já publicados continuam reprodutíveis byte a byte. Reaproveita o decoder, os
operadores genéticos, a inicialização, a eliminação de duplicatas e o
hipervolume de nsga2_simheuristic.

Formulação (problema triobjetivo com restrição de chance):
  f1 custo   — fixo + distância + veículo extra (+ custo por zona em B2).
  f2 qualidade — perda média ponderada por valor, avaliada no cenário NOMINAL
                 do modelo térmico dependente da rota (thermal_route: porta,
                 espera e serviço, resfriamento, clima horário real).
  f3 CO2     — deslocamento + refrigeração durante TODA a duração da rota
                 (viagem + espera + serviço), não só na viagem (decisão D6).
  Restrição de chance (D1): P(nenhum cliente atendido com atraso > δ) ≥ α,
                 δ = 5 min (D2), α ∈ {0,90; 0,95}. Tempo de viagem
                 lognormal de média 1 por trecho, CV = sigma_t_pct da
                 instância.

Saída do depósito (DECISÃO, bloco 3 do cromossomo): cada rota sai em
(g/4)·Δmax, g ∈ {0,...,4}, onde Δmax é o maior adiamento que não cria atraso
no plano nem aumenta a duração (folga para frente; Savelsbergh 1992
[VERIFICAR]): Δmax = min(F0, espera total), F0 = min_j (Σ_{i<j} w_i + l_j − A_j).
Sair cedo deixa folga contra atrasos, mas o produto fica mais tempo no
veículo (pior f2) e a refrigeração funciona mais (pior f3); sair tarde
preserva a qualidade, mas zera a folga de algum cliente. O DET, que não vê a
pontualidade, tende ao adiamento máximo — a armadilha que a SIM deve evitar.

Variantes (experiment_runner, runner 'nsga2c'):
  SIM   mc_rel > 0: estima P(pontual) com mc_rel cenários FIXOS por execução
        (números aleatórios comuns, semente da execução) e trata a restrição
        pela regra de dominância com restrição de Deb (2000).
  DET   mc_rel = 0, buffer_q = None: planeja com tempos nominais e não vê a
        restrição.
  BUF-q mc_rel = 0, buffer_q = q: planeja (decoder e saída) com tempos de
        viagem inflados ao quantil q da lognormal — a "folga de segurança"
        usual na prática. Executa com os tempos reais.
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

import nsga2_simheuristic as ns
import tempo_dependente as td
import thermal_route as tr

DELTA_MIN = 5.0      # tolerância de atraso (min), decisão D2
ALPHA     = 0.90     # nível de serviço, decisão D1
MC_REL    = 30       # cenários de tempo de viagem por execução (SIM)
N_NIVEIS_PARTIDA = 5  # saída = (g/4)·Δmax, g ∈ {0,...,4} (0 = sai em t=0; 4 = adiamento máximo)
# Cinética no compartimento ÚNICO (diário Entradas 83 e 87). Padrão False = carga
# refrigerada HOMOGÊNEA: toda a carga segue a categoria dominante (experimento principal,
# como na versão de conferência). True = carga MISTA, cada produto pela cinética da
# própria categoria no set-point do veículo — usado só na base de compartimento único do
# B2 (variantes *-MIX), por execução, via o parâmetro cinetica_por_cliente. O B2 com
# zonas sempre usa a cinética de cada produto.
CINETICA_POR_CLIENTE = False
# Eliminação de rotas na busca local (09/10/2026, diário Entrada 85): fecha o gap de
# custo contra o PyVRP nas famílias de janela estreita. False reproduz o motor das
# execuções anteriores (results_c_v1_cindom/).
ELIMINAR_ROTAS = True
# Busca local por variação de custo (09/10/2026, diário Entrada 86): mesmas decisões,
# muito menos recálculo. False usa a implementação original (_passes_ref).
BUSCA_LOCAL_DELTA = True


# ─────────────────────────────────────────────────────────────────────────────
# Indivíduo com restrição
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IndividualC(ns.Individual):
    viol: float = 0.0              # max(0, α − P(pontual)); 0 = cumpre (ou não avaliado)
    p_on: float = float('nan')     # P(pontual) estimada na amostra da execução
    partida: Tuple[float, ...] = ()  # saída do depósito por rota (min)
    dep_block: List[int] = field(default_factory=list)  # bloco 3: nível de saída por rota (posicional)
    p_cert: float = float('nan')   # P(pontual) na simulação longa de certificação

    def dominates(self, other) -> bool:
        """Dominância com restrição (Deb, 2000): viável domina inviável; entre
        inviáveis, a menor violação domina; entre viáveis, Pareto."""
        if self.viol == 0.0 and other.viol > 0.0:
            return True
        if self.viol > 0.0 and other.viol == 0.0:
            return False
        if self.viol > 0.0 and other.viol > 0.0:
            return self.viol < other.viol
        return ns.Individual.dominates(self, other)


# ─────────────────────────────────────────────────────────────────────────────
# Cronograma, saída do depósito e pontualidade
# ─────────────────────────────────────────────────────────────────────────────

def delta_max(route: List[int], inst: dict, ttime) -> float:
    """Maior adiamento da saída que não cria atraso no plano nem aumenta a duração
    (folga para frente; Savelsbergh 1992 [VERIFICAR]): min(F0, espera total)."""
    if td.eh_td(ttime):          # tempos dependentes do horário (Entrada 101)
        return td.delta_max(route, inst, ttime)
    cust = inst['customers']
    t = 0.0; prev = 0; cumw = 0.0; F0 = math.inf
    for c in route:
        t += ttime[prev][c]
        F0 = min(F0, cumw + cust[c - 1]['due_date_min'] - t)
        w = max(0.0, cust[c - 1]['ready_time_min'] - t)
        cumw += w
        t += w + cust[c - 1]['service_time_min']
        prev = c
    return max(0.0, min(F0, cumw))


def duracao(route: List[int], inst: dict, ttime, t0: float) -> float:
    """Duração (min) da rota saindo em t0, com retorno ao depósito."""
    if td.eh_td(ttime):
        return td.duracao(route, inst, ttime, t0)
    cust = inst['customers']
    t = t0; prev = 0
    for c in route:
        t += ttime[prev][c]
        t = max(t, cust[c - 1]['ready_time_min']) + cust[c - 1]['service_time_min']
        prev = c
    return t + ttime[prev][0] - t0


def plano_rota(route: List[int], inst: dict, ttime) -> Tuple[float, float]:
    """(Δmax, duração saindo em Δmax) — usado nos testes."""
    d = delta_max(route, inst, ttime)
    return d, duracao(route, inst, ttime, d)


def pontual_rota(route: List[int], inst: dict, ttime, tt: np.ndarray,
                 t0: float, delta_tol: float = DELTA_MIN) -> np.ndarray:
    """Por cenário (K,): True se nenhum cliente da rota chega com atraso > delta_tol.
    tt: multiplicadores K×(n+1) indexados pelo cliente de DESTINO do trecho."""
    if td.eh_td(ttime):
        return td.pontual_rota(route, inst, ttime, tt, t0, delta_tol)
    cust = inst['customers']
    K = tt.shape[0]
    t = np.full(K, float(t0)); ok = np.ones(K, dtype=bool); prev = 0
    for c in route:
        t = t + ttime[prev][c] * tt[:, c]
        ok &= (t - cust[c - 1]['due_date_min']) <= delta_tol
        t = np.maximum(t, cust[c - 1]['ready_time_min']) + cust[c - 1]['service_time_min']
        prev = c
    return ok


def multiplicador_quantil(cv: float, q: float) -> float:
    """Quantil q da lognormal de média 1 e coeficiente de variação cv."""
    if cv <= 0:
        return 1.0
    s2 = math.log(1.0 + cv * cv)
    return math.exp(-s2 / 2.0 + math.sqrt(s2) * norm.ppf(q))


# ─────────────────────────────────────────────────────────────────────────────
# Contexto de uma execução (matrizes, cenários, caches por rota)
# ─────────────────────────────────────────────────────────────────────────────

class Contexto:
    def __init__(self, inst: dict, seed: int, mc_rel: int = 0, buffer_q: Optional[float] = None,
                 alpha: float = ALPHA, delta_tol: float = DELTA_MIN,
                 use_multizone: bool = False, n_zones: int = 1,
                 estacao: str = tr.CLIMA_PADRAO, t_start_h: float = tr.T_START_PADRAO,
                 decoder: str = 'firstfit', p_ls: float = 0.0, k_viz: int = 10,
                 cinetica_por_cliente: Optional[bool] = None, m_L: Optional[float] = None,
                 m_U: Optional[float] = None, gamma: Optional[float] = None):
        self.inst = inst
        # Entrada 87: parâmetros por execução (None = padrão do módulo)
        self.cin = CINETICA_POR_CLIENTE if cinetica_por_cliente is None else bool(cinetica_por_cliente)
        self.m_L = ns.ZONE_LOADING_COST_KM_MULTIPLE if m_L is None else m_L
        self.m_U = ns.ZONE_UNLOADING_COST_KM_MULTIPLE if m_U is None else m_U
        self.gamma = ns.GAMMA_ZONE_COLD if gamma is None else gamma
        self.decoder, self.p_ls = decoder, p_ls
        self.dist, self.ttime = ns.build_matrices(inst)
        cv = inst['thermal_profile'].get('sigma_t_pct', 0.0)
        self.td = td.eh_td(self.ttime)   # tempos dependentes do horário (Entrada 101): só com o Split
        assert not self.td or decoder == 'split', 'tempos dependentes do horário exigem o decoder Split'
        if buffer_q:
            m = multiplicador_quantil(cv, buffer_q)
            self.ttime_plan = (self.ttime.escalada(m) if self.td
                               else [[x * m for x in row] for row in self.ttime])
        else:
            self.ttime_plan = self.ttime
        self.use_multizone, self.n_zones = use_multizone, n_zones
        self.cat_index = ns._cat_index_map(inst) if use_multizone else None
        self.alpha, self.delta_tol = alpha, delta_tol
        self.t_start_h, self.estacao = t_start_h, estacao
        self.nom = tr.cenario_nominal(inst, n_zones=n_zones, estacao=estacao)
        n = len(inst['customers'])
        self.tt = (tr._lognormal_media1(np.random.default_rng(seed + 7919), cv, (mc_rel, n + 1))
                   if mc_rel > 0 else None)
        self.c_plano: Dict = {}; self.c_q: Dict = {}; self.c_on: Dict = {}
        # vetores por cliente (índice = id, 0 = depósito) para Split e busca local
        cs = inst['customers']; fl = inst['fleet']
        self.dem = [0.0] + [c['demand_kg'] for c in cs]
        self.rdy = [0.0] + [c['ready_time_min'] for c in cs]
        self.due = [math.inf] + [c['due_date_min'] for c in cs]
        self.svc = [0.0] + [c['service_time_min'] for c in cs]
        self.cap, self.fix, self.var = fl['capacity_kg'], fl['fixed_cost_BRL'], fl['variable_cost_BRL_km']
        self.viz = [[]] + [sorted((j for j in range(1, n + 1) if j != i), key=lambda j: self.dist[i][j])[:k_viz]
                           for i in range(1, n + 1)]
        # clientes que nem sozinhos cabem na janela (planejamento): o Split os deixa de fora
        if self.td:
            self.sozinho_inviavel = {i for i in range(1, n + 1) if self.ttime_plan.chegada(0, i, 0.0) > self.due[i]}
        else:
            self.sozinho_inviavel = {i for i in range(1, n + 1) if self.ttime_plan[0][i] > self.due[i]}


def split_decode(chrom: List[int], ctx: Contexto) -> Tuple[List[List[int]], bool]:
    """Split (Prins, 2004 [VERIFICAR]) com capacidade e janelas de tempo: divide a
    permutação em rotas CONSECUTIVAS de custo mínimo (fixo + variável × km),
    por programação dinâmica em grafo acíclico. Usa os tempos de PLANEJAMENTO
    (com folga, nas variantes BUF). Clientes inviáveis sozinhos ficam de fora
    (feasible=False), como no decoder original."""
    if ctx.td:
        return td.split_decode(chrom, ctx)
    perm = [c for c in chrom if c not in ctx.sozinho_inviavel]
    n = len(perm); tt = ctx.ttime_plan; D = ctx.dist
    V = [math.inf] * (n + 1); P = [-1] * (n + 1); V[0] = 0.0
    for i in range(n):
        if V[i] == math.inf:
            continue
        load = 0.0; t = 0.0; d = 0.0; prev = 0
        for j in range(i, n):
            c = perm[j]
            load += ctx.dem[c]
            if load > ctx.cap:
                break
            t += tt[prev][c]
            if t > ctx.due[c]:
                break
            t = max(t, ctx.rdy[c]) + ctx.svc[c]
            d += D[prev][c]
            custo = V[i] + ctx.fix + ctx.var * (d + D[c][0])
            if custo < V[j + 1]:
                V[j + 1] = custo; P[j + 1] = i
            prev = c
    rotas = []; j = n
    while j > 0:
        i = P[j]; rotas.append(perm[i:j]); j = i
    rotas.reverse()
    return rotas, not ctx.sozinho_inviavel


def _rota_viavel(r: List[int], ctx: Contexto) -> bool:
    if ctx.td:
        return td.viavel(r, ctx)
    load = 0.0; t = 0.0; prev = 0; tt = ctx.ttime_plan
    for c in r:
        load += ctx.dem[c]
        if load > ctx.cap:
            return False
        t += tt[prev][c]
        if t > ctx.due[c]:
            return False
        t = max(t, ctx.rdy[c]) + ctx.svc[c]; prev = c
    return True


def _custo_rota(r: List[int], ctx: Contexto) -> float:
    if not r:
        return 0.0
    D = ctx.dist
    return ctx.fix + ctx.var * (D[0][r[0]] + sum(D[a][b] for a, b in zip(r, r[1:])) + D[r[-1]][0])


def busca_local(rotas: List[List[int]], ctx: Contexto, rng: random.Random, max_passes: int = 2) -> List[List[int]]:
    """Busca local granular de primeira melhoria sobre o CUSTO (f1), só com
    movimentos viáveis (capacidade e janelas, tempos de planejamento):
    realocação de u logo depois de v, troca u<->v e 2-opt* entre as rotas de u
    e v, para v entre os k vizinhos mais próximos de u (Toth & Vigo [VERIFICAR])."""
    R = (_passes_delta if BUSCA_LOCAL_DELTA else _passes_ref)(rotas, ctx, rng, max_passes)
    if ELIMINAR_ROTAS:
        for _ in range(2):                       # no máximo duas eliminações por chamada
            novo = _eliminar_rota(R, ctx)
            if novo is None:
                break
            R = novo
    return R


def _passes_delta(rotas: List[List[int]], ctx: Contexto, rng: random.Random, max_passes: int) -> List[List[int]]:
    """Mesmas decisões de _passes_ref (mesma ordem de clientes, vizinhos e movimentos,
    mesma regra de aceitação), mas o custo de cada movimento é calculado pela variação
    das arestas, O(1), em vez de recalcular as rotas inteiras; as rotas candidatas só
    são montadas, e a viabilidade só é testada, quando o movimento melhora o custo
    (09/10/2026, diário Entrada 86). Validado contra _passes_ref em
    pesquisa/teste_busca_local_delta.py."""
    R = [r[:] for r in rotas]
    custo = [_custo_rota(r, ctx) for r in R]
    D = ctx.dist; var = ctx.var
    for _ in range(max_passes):
        melhorou = False
        pos = {c: (ri, k) for ri, r in enumerate(R) for k, c in enumerate(r)}
        ordem = list(pos); rng.shuffle(ordem)
        for u in ordem:
            for v in ctx.viz[u]:
                if v not in pos:
                    continue
                ru, iu = pos[u]; rv, iv = pos[v]
                Ru = R[ru]; Lu = len(Ru)
                pu = Ru[iu - 1] if iu > 0 else 0
                su = Ru[iu + 1] if iu < Lu - 1 else 0
                mv = None
                if ru == rv:                              # realocação dentro da mesma rota
                    sv = su if v == pu else (Ru[iv + 1] if iv < Lu - 1 else 0)
                    d = var * ((D[pu][su] - D[pu][u] - D[u][su]) + (D[v][u] + D[u][sv] - D[v][sv]))
                    if d < -1e-9:
                        B = Ru[:iu] + Ru[iu + 1:]
                        B.insert(B.index(v) + 1, u)
                        if _rota_viavel(B, ctx):
                            mv = {ru: B}
                else:
                    Rv = R[rv]; Lv = len(Rv)
                    pv = Rv[iv - 1] if iv > 0 else 0
                    sv = Rv[iv + 1] if iv < Lv - 1 else 0
                    # 1) realocação de u logo depois de v
                    d_rem = -custo[ru] if Lu == 1 else var * (D[pu][su] - D[pu][u] - D[u][su])
                    if d_rem + var * (D[v][u] + D[u][sv] - D[v][sv]) < -1e-9:
                        A = Ru[:iu] + Ru[iu + 1:]
                        B = Rv[:iv + 1] + [u] + Rv[iv + 1:]
                        if (not A or _rota_viavel(A, ctx)) and _rota_viavel(B, ctx):
                            mv = {ru: A, rv: B}
                    # 2) troca u <-> v
                    if mv is None and var * ((D[pu][v] + D[v][su] - D[pu][u] - D[u][su])
                                             + (D[pv][u] + D[u][sv] - D[pv][v] - D[v][sv])) < -1e-9:
                        A2 = Ru[:]; B2 = Rv[:]; A2[iu] = v; B2[iv] = u
                        if _rota_viavel(A2, ctx) and _rota_viavel(B2, ctx):
                            mv = {ru: A2, rv: B2}
                    # 3) 2-opt* (troca das caudas)
                    if mv is None and var * (D[u][sv] + D[v][su] - D[u][su] - D[v][sv]) < -1e-9:
                        A3 = Ru[:iu + 1] + Rv[iv + 1:]; B3 = Rv[:iv + 1] + Ru[iu + 1:]
                        if _rota_viavel(A3, ctx) and _rota_viavel(B3, ctx):
                            mv = {ru: A3, rv: B3}
                if mv is not None:
                    for k, r in mv.items():
                        R[k] = r; custo[k] = _custo_rota(r, ctx)
                    melhorou = True
                    pos = {c: (ri, k) for ri, r in enumerate(R) for k, c in enumerate(r)}
                    break
        par = [(r, c) for r, c in zip(R, custo) if r]
        R = [r for r, _ in par]; custo = [c for _, c in par]
        if not melhorou:
            break
    return R


def _passes_ref(rotas: List[List[int]], ctx: Contexto, rng: random.Random, max_passes: int) -> List[List[int]]:
    """Implementação original das passadas (recalcula o custo das rotas inteiras a cada
    movimento candidato). Mantida como referência de teste para _passes_delta."""
    R = [r[:] for r in rotas]
    custo = [_custo_rota(r, ctx) for r in R]
    for _ in range(max_passes):
        melhorou = False
        pos = {c: (ri, k) for ri, r in enumerate(R) for k, c in enumerate(r)}
        ordem = list(pos); rng.shuffle(ordem)
        for u in ordem:
            for v in ctx.viz[u]:
                if v not in pos:
                    continue
                ru, iu = pos[u]; rv, iv = pos[v]
                cand = []
                A = R[ru][:iu] + R[ru][iu + 1:]          # u sai da rota dela
                B = (A if ru == rv else R[rv])[:]
                B.insert(B.index(v) + 1, u)              # u entra logo depois de v
                cand.append({ru: B} if ru == rv else {ru: A, rv: B})
                if ru != rv:
                    A2 = R[ru][:]; B2 = R[rv][:]; A2[iu] = v; B2[iv] = u          # troca
                    cand.append({ru: A2, rv: B2})
                    cand.append({ru: R[ru][:iu + 1] + R[rv][iv + 1:],             # 2-opt*
                                 rv: R[rv][:iv + 1] + R[ru][iu + 1:]})
                for mv in cand:
                    novo = sum(_custo_rota(r, ctx) for r in mv.values())
                    if (novo < sum(custo[k] for k in mv) - 1e-9
                            and all(_rota_viavel(r, ctx) for r in mv.values() if r)):
                        for k, r in mv.items():
                            R[k] = r; custo[k] = _custo_rota(r, ctx)
                        melhorou = True
                        pos = {c: (ri, k) for ri, r in enumerate(R) for k, c in enumerate(r)}
                        break
                else:
                    continue
                break
        par = [(r, c) for r, c in zip(R, custo) if r]
        R = [r for r, _ in par]; custo = [c for _, c in par]
        if not melhorou:
            break
    return R


def _eliminar_rota(R: List[List[int]], ctx: Contexto) -> Optional[List[List[int]]]:
    """Eliminação de rota (09/10/2026, diário Entrada 85): tenta esvaziar a rota com
    menos clientes inserindo cada um deles, da janela mais estreita para a mais larga,
    na posição viável mais barata de qualquer outra rota (capacidade e janelas, tempos
    de planejamento). Aceita só se o custo total cair; senão devolve None. Os
    movimentos de um cliente por vez da busca local não conseguem esvaziar rotas com
    vários clientes, o que deixava rotas a mais nas famílias de janela estreita."""
    if len(R) < 2:
        return None
    alvo = min(range(len(R)), key=lambda i: (len(R[i]), _custo_rota(R[i], ctx)))
    outras = [r[:] for i, r in enumerate(R) if i != alvo]
    carga = [sum(ctx.dem[x] for x in r) for r in outras]
    for c in sorted(R[alvo], key=lambda c: ctx.due[c] - ctx.rdy[c]):
        melhor = None
        for ri, r in enumerate(outras):
            if carga[ri] + ctx.dem[c] > ctx.cap:
                continue
            base = _custo_rota(r, ctx)
            for k in range(len(r) + 1):
                nr = r[:k] + [c] + r[k:]
                if _rota_viavel(nr, ctx):
                    d = _custo_rota(nr, ctx) - base
                    if melhor is None or d < melhor[0]:
                        melhor = (d, ri, nr)
        if melhor is None:
            return None
        outras[melhor[1]] = melhor[2]; carga[melhor[1]] += ctx.dem[c]
    if sum(_custo_rota(r, ctx) for r in outras) < sum(_custo_rota(r, ctx) for r in R) - 1e-9:
        return outras
    return None


def _talvez_busca_local(ch: List[int], ctx: Contexto, rng: random.Random) -> List[int]:
    """Com probabilidade p_ls (só com o decoder Split): Split -> busca local ->
    cromossomo = concatenação das rotas melhoradas (+ clientes deixados de
    fora, no fim). O Split desse cromossomo tem custo <= o das rotas melhoradas."""
    if ctx.decoder != 'split' or ctx.p_ls <= 0 or rng.random() >= ctx.p_ls:
        return ch
    rotas, _ = split_decode(ch, ctx)
    rotas = busca_local(rotas, ctx, rng)
    usados = [c for r in rotas for c in r]
    ja = set(usados)
    return usados + [c for c in ch if c not in ja]


def evaluate_c(ind: IndividualC, ctx: Contexto) -> None:
    inst = ctx.inst
    routes, feasible = (split_decode(ind.chromosome, ctx) if ctx.decoder == 'split'
                        else ns.decode(ind.chromosome, inst, ctx.ttime_plan))
    ind.routes, ind.feasible = routes, feasible
    if not routes:
        ind.f_cost, ind.f_qloss, ind.f_co2 = ns.PENALTY_INFEAS, 1.0, ns.PENALTY_INFEAS
        ind.viol, ind.p_on, ind.partida = 0.0, float('nan'), ()
        return
    fleet = inst['fleet']; co2p = inst.get('co2_params', {})
    fix, var = fleet['fixed_cost_BRL'], fleet['variable_cost_BRL_km']
    a_co2 = co2p.get('alpha_kg_per_km', 0.21); b_co2 = co2p.get('beta_kg_per_kWh', 0.092)
    P_ref = co2p.get('P_refrigeration_kW', 3.5)
    cust = inst['customers']
    zones = (ns.assign_zones(routes, ind.zone_block, inst, ctx.cat_index) if ctx.use_multizone else None)
    T_ref = inst['thermal_profile']['T_set_C']
    cost = co2 = wq = val = 0.0
    on = np.ones(ctx.tt.shape[0], dtype=bool) if ctx.tt is not None else None
    partidas = []
    for i, route in enumerate(routes):
        g = ind.dep_block[i] if i < len(ind.dep_block) else N_NIVEIS_PARTIDA - 1
        key = (tuple(route), g)
        if key not in ctx.c_plano:
            dmax = delta_max(route, inst, ctx.ttime_plan)
            d0 = dmax * g / (N_NIVEIS_PARTIDA - 1)
            km = sum(ctx.dist[a][b] for a, b in zip([0] + route, route + [0]))
            ctx.c_plano[key] = (d0, duracao(route, inst, ctx.ttime, d0), km)
        d0, dur, km = ctx.c_plano[key]
        partidas.append(d0)
        if ctx.use_multizone:
            zi = zones[i]
            zkey = (key, tuple(sorted(zi['zone_of_category'].items())))
            P_route = sum(ns._p_ref_zone(z['T_set_C'], P_ref, T_ref, ctx.gamma)
                          for z in zi['zone_temp'].values())
            nz = len(zi['zone_temp'])
            zcost = (ctx.m_L * var * (ns.ZONE_LOADING_FACTOR[nz] - 1.0)
                     + ctx.m_U * var * (ns.ZONE_UNLOADING_FACTOR[nz] - 1.0) * len(route))
        else:
            zkey, P_route, zcost = key, P_ref, 0.0
        if zkey not in ctx.c_q:
            if ctx.use_multizone:
                zoc = [zi['zone_of_category'][cust[c - 1]['product_category']] for c in route]
                q = tr.route_quality(route, inst, ctx.ttime, ctx.nom, zoc, zi['zone_temp'],
                                     per_customer_kinetics=True, t_start_h=ctx.t_start_h, t0_min=d0)[0]
            else:
                q = tr.route_quality(route, inst, ctx.ttime, ctx.nom,
                                     per_customer_kinetics=ctx.cin,
                                     t_start_h=ctx.t_start_h, t0_min=d0)[0]
            ctx.c_q[zkey] = q
        r_val = sum(cust[c - 1].get('load_value_BRL', 1.0) for c in route)
        wq += ctx.c_q[zkey] * r_val; val += r_val
        cost += fix + var * km + zcost
        co2 += a_co2 * km + b_co2 * P_route * (dur / 60.0)
        if on is not None:
            if key not in ctx.c_on:
                ctx.c_on[key] = pontual_rota(route, inst, ctx.ttime, ctx.tt, d0, ctx.delta_tol)
            on &= ctx.c_on[key]
    n_extra = max(0, len(routes) - fleet['num_vehicles'])
    pen = 0.0 if feasible else ns.PENALTY_INFEAS
    ind.f_cost = cost + n_extra * ns.EXTRA_VEHICLE_COST + pen
    ind.f_qloss = 1.0 - (wq / val if val > 0 else 0.0)
    ind.f_co2 = co2 + pen
    ind.partida = tuple(partidas)
    if on is not None:
        ind.p_on = float(on.mean())
        ind.viol = max(0.0, ctx.alpha - ind.p_on)
    else:
        ind.p_on, ind.viol = float('nan'), 0.0


# ─────────────────────────────────────────────────────────────────────────────
# NSGA-II do Caminho C (eliminação de duplicatas sempre ligada)
# ─────────────────────────────────────────────────────────────────────────────

def _assinatura(ind, use_multizone: bool, n_cat: int):
    R = len(ind.routes)
    sig = (tuple(tuple(r) for r in ind.routes), tuple(ind.dep_block[:R]))
    return (sig, tuple(ind.zone_block[:R * n_cat])) if use_multizone else sig


def certificar(cands: List[IndividualC], ctx: Contexto, mc_cert: int, seed: int) -> List[IndividualC]:
    """Etapa de certificação da simheurística (simulação longa; Juan et al. 2015
    [VERIFICAR]): reavalia cada candidata em mc_cert cenários NOVOS de tempo de
    viagem, independentes dos da busca e dos da análise fora da amostra, e
    devolve a frente não dominada (objetivos) das que cumprem P ≥ α. Se
    nenhuma cumpre, devolve a de maior P(pontual) certificada."""
    inst = ctx.inst; n = len(inst['customers'])
    cv = inst['thermal_profile'].get('sigma_t_pct', 0.0)
    tt = tr._lognormal_media1(np.random.default_rng(seed + 104729), cv, (mc_cert, n + 1))
    memo = {}
    for ind in cands:
        on = np.ones(mc_cert, dtype=bool)
        for r, d0 in zip(ind.routes, ind.partida):
            k = (tuple(r), round(d0, 6))
            if k not in memo:
                memo[k] = pontual_rota(r, inst, ctx.ttime, tt, d0, ctx.delta_tol)
            on &= memo[k]
        ind.p_cert = float(on.mean())
    ok = [x for x in cands if x.feasible and x.p_cert >= ctx.alpha]
    if not ok:
        return [max(cands, key=lambda x: x.p_cert)] if cands else []
    return [a for a in ok if not any(ns.Individual.dominates(b, a) for b in ok if b is not a)]


def _split_duplicates_c(pop, use_multizone: bool, n_cat: int):
    """Como ns._split_duplicates, mas a assinatura inclui o nível de saída das
    rotas usadas — mesmas rotas com saídas diferentes NÃO são duplicatas."""
    seen = set(); unique = []; dups = []
    for ind in pop:
        R = len(ind.routes)
        sig = (tuple(tuple(r) for r in ind.routes), tuple(ind.dep_block[:R]))
        if use_multizone:
            sig = (sig, tuple(ind.zone_block[:R * n_cat]))
        (dups if sig in seen else unique).append(ind)
        seen.add(sig)
    return unique, dups


def _init_pop(ctx: Contexto, pop_size: int, rng_py: random.Random) -> List[IndividualC]:
    inst = ctx.inst; n = len(inst['customers'])
    n_cat = len(ctx.cat_index) if ctx.cat_index else 0
    n_edd = max(1, pop_size // 5); n_nn = pop_size // 2 - n_edd; n_rnd = pop_size - n_edd - n_nn
    chroms = ([ns._edd_perturbed(inst, rng_py) for _ in range(n_edd)]
              + [ns._nn_chromosome(inst, ctx.dist, rng_py) for _ in range(n_nn)])
    for _ in range(n_rnd):
        c = list(range(1, n + 1)); rng_py.shuffle(c); chroms.append(c)
    pop = []
    for c in chroms:
        ind = IndividualC(chromosome=c,
                          dep_block=[rng_py.randrange(N_NIVEIS_PARTIDA) for _ in range(n)])
        if ctx.use_multizone:
            ind.zone_block = ns._init_zone_block(n, n_cat, ctx.n_zones, rng_py)
        evaluate_c(ind, ctx); pop.append(ind)
    return pop


def run_nsga2_c(inst: dict, pop_size: int = ns.POP_SIZE, n_gen: int = ns.N_GEN, seed: int = 0,
                mc_rel: int = 0, buffer_q: Optional[float] = None, alpha: float = ALPHA,
                delta_tol: float = DELTA_MIN, use_multizone: bool = False, n_zones: int = 1,
                estacao: str = tr.CLIMA_PADRAO, t_start_h: float = tr.T_START_PADRAO,
                hv_ref_point: Optional[Tuple] = None, verbose: bool = False, mc_cert: int = 0,
                decoder: str = 'firstfit', p_ls: float = 0.0,
                cinetica_por_cliente: Optional[bool] = None, m_L: Optional[float] = None,
                m_U: Optional[float] = None, gamma: Optional[float] = None):
    rng_py = random.Random(seed)
    ctx = Contexto(inst, seed, mc_rel, buffer_q, alpha, delta_tol, use_multizone, n_zones,
                   estacao, t_start_h, decoder=decoder, p_ls=p_ls, cinetica_por_cliente=cinetica_por_cliente,
                   m_L=m_L, m_U=m_U, gamma=gamma)
    n_cat = len(ctx.cat_index) if ctx.cat_index else 0
    pop = _init_pop(ctx, pop_size, rng_py)
    arquivo: Dict = {}   # rank-1 que cumprem a restrição na busca (para certificação)
    hv_hist = []; t0 = time.time()
    for gen in range(1, n_gen + 1):
        for front in ns.fast_nondominated_sort(pop):
            ns.crowding_distance_assign(front)
        off = []
        while len(off) < pop_size:
            p1 = ns.tournament_select(pop, rng_py); p2 = ns.tournament_select(pop, rng_py)
            do_cx = rng_py.random() < ns.CROSSOVER_RATE
            ch = ns.ox_crossover(p1, p2, rng_py).chromosome if do_cx else p1.chromosome[:]
            if rng_py.random() < ns.MUTATION_RATE:
                ch = ns.mutate(ns.Individual(chromosome=ch), rng_py).chromosome
            db = ns.zone_block_crossover(p1.dep_block, p2.dep_block, rng_py) if do_cx else p1.dep_block[:]
            ch = _talvez_busca_local(ch, ctx, rng_py)
            child = IndividualC(chromosome=ch, dep_block=ns.zone_block_mutate(db, N_NIVEIS_PARTIDA, rng_py))
            if use_multizone:
                zb = ns.zone_block_crossover(p1.zone_block, p2.zone_block, rng_py) if do_cx else p1.zone_block[:]
                child.zone_block = ns.zone_block_mutate(zb, n_zones, rng_py)
            evaluate_c(child, ctx); off.append(child)
        comb, dups = _split_duplicates_c(pop + off, use_multizone, n_cat)
        fronts = ns.fast_nondominated_sort(comb)
        for front in fronts:
            ns.crowding_distance_assign(front)
        new = []
        for front in fronts:
            if len(new) + len(front) <= pop_size:
                new.extend(front)
            else:
                new.extend(sorted(front, key=lambda x: -x.crowding)[:pop_size - len(new)])
                break
        for d in dups:
            if len(new) >= pop_size:
                break
            d.rank, d.crowding = ns.DUP_RANK, 0.0
            new.append(d)
        pop = new
        if mc_cert:
            for x in pop:
                if x.rank == 1 and x.viol == 0.0 and x.feasible:
                    arquivo.setdefault(_assinatura(x, use_multizone, n_cat), x)
        if gen % 10 == 0 or gen == n_gen:
            pf = [x for x in pop if x.rank == 1]
            hv_hist.append((gen, ns.compute_hypervolume(pf, ref_point=hv_ref_point)))
            if verbose:
                print(f"  Gen {gen:4d} | frente {len(pf):3d} | viáveis(chance) "
                      f"{sum(1 for x in pf if x.viol == 0)} | {time.time() - t0:.1f}s", flush=True)
    pareto = [x for x in pop if x.rank == 1]
    if mc_cert:
        pareto = certificar(list(arquivo.values()) or pareto, ctx, mc_cert, seed)
    return pareto, hv_hist


# ─────────────────────────────────────────────────────────────────────────────
# MOEA/D do Caminho C (baseline evolutivo externo, mesma avaliação da SIM)
# ─────────────────────────────────────────────────────────────────────────────

def run_moead_c(inst: dict, pop_size: int = ns.POP_SIZE, n_gen: int = ns.N_GEN, seed: int = 0,
                mc_rel: int = MC_REL, buffer_q: Optional[float] = None, alpha: float = ALPHA,
                delta_tol: float = DELTA_MIN, use_multizone: bool = False, n_zones: int = 1,
                estacao: str = tr.CLIMA_PADRAO, t_start_h: float = tr.T_START_PADRAO,
                hv_ref_point: Optional[Tuple] = None, verbose: bool = False, mc_cert: int = 0,
                decoder: str = 'firstfit', p_ls: float = 0.0,
                cinetica_por_cliente: Optional[bool] = None, m_L: Optional[float] = None,
                m_U: Optional[float] = None, gamma: Optional[float] = None):
    """MOEA/D (Tchebycheff normalizado, mesmos MOEAD_H/T/DELTA/NR de ns.run_moead)
    com regra de viabilidade na substituição: quem cumpre a restrição de chance
    substitui quem não cumpre; entre não-cumpridoras, menor violação; entre
    cumpridoras, Tchebycheff. pop_size é ignorado (105 subproblemas)."""
    rng_py = random.Random(seed)
    ctx = Contexto(inst, seed, mc_rel, buffer_q, alpha, delta_tol, use_multizone, n_zones,
                   estacao, t_start_h, decoder=decoder, p_ls=p_ls, cinetica_por_cliente=cinetica_por_cliente,
                   m_L=m_L, m_U=m_U, gamma=gamma)
    W = ns._das_dennis_weights(3, ns.MOEAD_H); n_sub = len(W)
    T = min(ns.MOEAD_T, n_sub); viz = ns._compute_neighbors(W, T)
    pop = _init_pop(ctx, n_sub, rng_py)
    n_cat = len(ctx.cat_index) if ctx.cat_index else 0
    arquivo: Dict = {}
    objs = lambda x: np.array([x.f_cost, x.f_qloss, x.f_co2])   # noqa: E731
    z_star = np.min([objs(x) for x in pop], axis=0)
    viav = [x for x in pop if x.feasible] or pop
    z_nad = np.max([objs(x) for x in viav], axis=0)

    def tch(x, w):
        return float(np.max(w * (objs(x) - z_star) / np.maximum(z_nad - z_star, 1e-9)))

    def melhor(a, b, w):   # a substitui b?
        if a.viol != b.viol:
            return a.viol < b.viol
        return tch(a, w) <= tch(b, w)

    hv_hist = []
    for gen in range(1, n_gen + 1):
        for i in range(n_sub):
            pool = viz[i] if rng_py.random() < ns.MOEAD_DELTA else list(range(n_sub))
            p1, p2 = (pop[k] for k in rng_py.sample(pool, 2))
            do_cx = rng_py.random() < ns.CROSSOVER_RATE
            ch = ns.ox_crossover(p1, p2, rng_py).chromosome if do_cx else p1.chromosome[:]
            if rng_py.random() < ns.MUTATION_RATE:
                ch = ns.mutate(ns.Individual(chromosome=ch), rng_py).chromosome
            db = ns.zone_block_crossover(p1.dep_block, p2.dep_block, rng_py) if do_cx else p1.dep_block[:]
            ch = _talvez_busca_local(ch, ctx, rng_py)
            child = IndividualC(chromosome=ch, dep_block=ns.zone_block_mutate(db, N_NIVEIS_PARTIDA, rng_py))
            if use_multizone:
                zb = ns.zone_block_crossover(p1.zone_block, p2.zone_block, rng_py) if do_cx else p1.zone_block[:]
                child.zone_block = ns.zone_block_mutate(zb, n_zones, rng_py)
            evaluate_c(child, ctx)
            z_star = np.minimum(z_star, objs(child))
            if child.feasible:
                z_nad = np.maximum(z_nad, objs(child))
            ordem = viz[i][:]; rng_py.shuffle(ordem); n_upd = 0
            for j in ordem:
                if n_upd >= ns.MOEAD_NR:
                    break
                if melhor(child, pop[j], W[j]):
                    pop[j] = child; n_upd += 1
        if mc_cert:
            for x in ns._nondominated_subset(pop):
                if x.viol == 0.0 and x.feasible:
                    arquivo.setdefault(_assinatura(x, use_multizone, n_cat), x)
        if gen % 10 == 0 or gen == n_gen:
            nd = ns._nondominated_subset(pop)
            hv_hist.append((gen, ns.compute_hypervolume(nd, ref_point=hv_ref_point)))
    final = ns._nondominated_subset(pop)
    if mc_cert:
        final = certificar(list(arquivo.values()) or final, ctx, mc_cert, seed)
    return final, hv_hist
