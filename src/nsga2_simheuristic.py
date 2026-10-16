"""
NSGA-II com Simulação Monte Carlo — CCVRPTW-ST
===============================================
Simheurística multiobjetivo para distribuição em cadeia de frio
sob incerteza estocástica de temperatura.

Objetivos (todos minimizados):
  f1 — Custo total de transporte (R$): custo fixo por veículo + variável por km
  f2 — Perda de qualidade (adimensional, 0–1): 1 - qualidade residual média
       ponderada por valor da carga, avaliada via Monte Carlo
  f3 — Emissões de CO₂ (kg CO₂-eq): motor + sistema de refrigeração

Algoritmo:
  1. Inicialização: heurística de vizinho mais próximo com perturbação
     aleatória (biased-randomized) + permutações aleatórias puras
  2. Avaliação: Monte Carlo embarcado (MC_SAMPLES replicações de temperatura)
  3. NSGA-II: classificação não-dominada rápida + distância de aglomeração
     + seleção por torneio binário + crossover OX + mutação 2-opt / or-opt
  4. Métricas: Hipervolume (HV), Generational Distance (GD), Spread (Δ)

Representação da solução:
  Giant-tour — permutação de {1, …, n} decodificada em rotas por
  inserção sequencial respeitando capacidade e janelas de tempo.

Referências:
  [1] Deb et al. (2002) IEEE Trans. Evol. Comput. 6(2):182–197.
  [2] Juan et al. (2021) Operations Research Perspectives 8:100124.
  [3] Ndraha et al. (2018) Food Control 89:12–21.
  [4] Solomon (1987) Oper. Res. 35(2):254–265.
  [5] Zitzler & Thiele (1999) Evol. Comput. 7(2):173–195.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1. PARÂMETROS DO ALGORITMO
# ─────────────────────────────────────────────────────────────────────────────

POP_SIZE         = 100      # tamanho da população
N_GEN            = 200      # número de gerações
CROSSOVER_RATE   = 0.90     # probabilidade de crossover (OX)
MUTATION_RATE    = 0.30     # probabilidade de mutação (2-opt / or-opt)
MC_SAMPLES       = 30       # replicações Monte Carlo por avaliação
TOURNAMENT_K     = 2        # tamanho do torneio
PENALTY_INFEAS   = 1e9      # penalidade de infeasibilidade
EXTRA_VEHICLE_COST = 400.0  # custo de locacao por veiculo adicional (R$)
                            # > 2x custo fixo, incentiva uso do frota nominal
R_GAS            = 8.314    # constante universal dos gases (J mol⁻¹ K⁻¹)

# --- B2: multi-compartimento térmico (opcional — use_multizone=True) -------
# Ver pesquisa/proposta_extensao_ejor.md §4.5.1 e pesquisa/plano_implementacao_b2.md.
# Com use_multizone=False (default em toda função abaixo), nenhum destes é usado
# e o comportamento é idêntico ao código anterior a B2 (compatibilidade retroativa
# deliberada — nada dos 1.080 resultados já reprocessados nesta pesquisa deve mudar
# de valor só por B2 existir no código).
GAMMA_ZONE_COLD  = 0.0135   # ESTIMATIVA NÃO VERIFICADA (°C⁻¹), extensão de f3, variante (ii).
                            # Derivada grosseiramente de um achado de +31% de consumo em carga
                            # congelada (fonte não confirmada por leitura direta) sobre ΔT≈23°C.
                            # NÃO usar no manuscrito sem confirmar a fonte ou substituí-la (LEI II.3).

# --- B2: extensão de f1 — custo de carga/descarga por zona ativa (23/08/2026) --
# Fundamentado em Hübner & Ostermeier (2018, Transportation Science 53(1):282-300),
# Tabela 4 — fatores de custo RELATIVOS (% de SCV=1 compartimento), calibrados com
# estudo de tempo-e-movimento real num grande varejista alemão. Valores absolutos
# são protegidos por acordo de confidencialidade no artigo-fonte — só os fatores
# relativos abaixo são publicados; ver literatura_mcvrp_compartimentos.md §10.2.
# ZONE_LOADING_FACTOR: custo de carga por veículo cresce ~linear com nº de
#   compartimentos ativos (100%→206%→306% para 1→2→3, dados do artigo).
# ZONE_UNLOADING_FACTOR: custo de descarga por cliente atendido fica ~flat
#   (100%→112%→112% para 1→2→3 — não escala além do primeiro compartimento extra).
ZONE_LOADING_FACTOR   = {1: 1.00, 2: 2.06, 3: 3.06}
ZONE_UNLOADING_FACTOR = {1: 1.00, 2: 1.12, 3: 1.12}
# Âncora de magnitude — mesma convenção do artigo-fonte ("custo de carregar 1
# compartimento = 2,60x o custo de 1 km de deslocamento"; descarregar = 2,05x).
# ESTIMATIVA DE SENSIBILIDADE (mesmo tratamento dado a GAMMA_ZONE_COLD acima) —
# o artigo-fonte é de um varejista alemão (EUR), não do contexto do CCVRPTW-ST
# (BRL); a ESTRUTURA relativa é citável, a MAGNITUDE absoluta não foi verificada
# para este problema. NÃO usar no manuscrito sem marcar explicitamente como tal.
ZONE_LOADING_COST_KM_MULTIPLE   = 2.60
ZONE_UNLOADING_COST_KM_MULTIPLE = 2.05
# Só se aplica com use_multizone=True (baseline permanece com contribuição zero,
# compatibilidade retroativa preservada — ver evaluate() abaixo).

# --- MOEA/D: baseline externo (frente 4.1, 23/08/2026) ---------------------
# Zhang & Li (2007) — decomposição Tchebycheff. Defaults padrão da literatura,
# NÃO tunados (mesmo tratamento dado aos parâmetros do NSGA-II acima) — o
# objetivo é um checkpoint de robustez do NSGA-II+MC, não otimizar o MOEA/D
# em si (ver proposta_extensao_ejor.md §3.2, plano_implementacao_moead.md).
# Citações (Zhang & Li 2007; Das & Dennis 1998/via Deb & Jain 2014) são
# conhecimento geral da área, NÃO verificadas contra fonte primária nesta
# pesquisa — diferente das citações já lidas na íntegra neste projeto.
MOEAD_H     = 13    # divisões da malha de Das-Dennis (3 obj. -> C(15,2)=105 subproblemas)
MOEAD_T     = 20    # tamanho da vizinhança (subproblemas mais próximos por peso)
MOEAD_DELTA = 0.90  # prob. de restringir os pais à vizinhança (mating restriction)
MOEAD_NR    = 2     # nº máx. de vizinhos atualizados por filho gerado


# ─────────────────────────────────────────────────────────────────────────────
# 2. REPRESENTAÇÃO DE SOLUÇÃO
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Individual:
    """
    Indivíduo do NSGA-II.

    Cromossomo (bloco 1): giant-tour — permutação dos IDs de clientes (1-indexados).
    Decodificado em rotas por inserção sequencial com cortes baseados em
    capacidade e janelas de tempo.

    zone_block (bloco 2, só usado se use_multizone=True — B2a, ver
    pesquisa/plano_implementacao_b2.md §4): vetor auxiliar SEPARADO do
    cromossomo (não concatenado — ox_crossover() assume permutação e
    corromperia valores repetidos). Tamanho n_customers × n_categorias,
    gene = zona escolhida (0..n_zones-1) para (i-ésima rota decodificada
    × categoria). Mapeamento posicional: slot i do bloco corresponde à
    i-ésima rota que decode() produzir, não a um "veículo físico" fixo.
    """
    chromosome: List[int]
    zone_block: List[int] = field(default_factory=list)
    routes:     List[List[int]] = field(default_factory=list)
    f_cost:     float = float('inf')
    f_qloss:    float = float('inf')
    f_co2:      float = float('inf')
    rank:       int   = 0
    crowding:   float = 0.0
    feasible:   bool  = True

    def obj(self) -> Tuple[float, float, float]:
        return (self.f_cost, self.f_qloss, self.f_co2)

    def dominates(self, other: Individual) -> bool:
        """True se self domina other em todos os objetivos (pelo menos um estrito)."""
        a, b = self.obj(), other.obj()
        return (all(ai <= bi for ai, bi in zip(a, b))
                and any(ai < bi for ai, bi in zip(a, b)))


# ─────────────────────────────────────────────────────────────────────────────
# 3. CARREGAMENTO DA INSTÂNCIA E MATRIZES
# ─────────────────────────────────────────────────────────────────────────────

def load_instance(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def build_matrices(inst: dict) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Constrói matrizes de distância euclidiana (km) e tempo de percurso (min).
    Índice 0 = depósito; índices 1..n = clientes.
    """
    nodes = [inst['depot']] + inst['customers']
    n     = len(nodes)
    speed = inst['fleet']['speed_km_h']
    dist  = [[0.0] * n for _ in range(n)]
    ttime = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = nodes[i]['x'], nodes[i]['y']
        for j in range(n):
            if i != j:
                dx = xi - nodes[j]['x']
                dy = yi - nodes[j]['y']
                d  = math.hypot(dx, dy)
                dist[i][j]  = round(d, 4)
                ttime[i][j] = round(d / speed * 60.0, 4)
    return dist, ttime


# ─────────────────────────────────────────────────────────────────────────────
# 4. DECODIFICADOR GIANT-TOUR → ROTAS
# ─────────────────────────────────────────────────────────────────────────────

def decode(
    chromosome: List[int],
    inst: dict,
    ttime: List[List[float]],
) -> Tuple[List[List[int]], bool]:
    """
    Decoder paralelo (parallel-route insertion) para giant-tour CVRPTW.

    Para cada cliente no cromossomo, tenta inserir em QUALQUER rota aberta
    (first-fit: primeira que respeite capacidade e janela de tempo).
    Só abre nova rota se nenhuma existente aceitar o cliente.
    Se o cliente não puder ser servido nem com veículo dedicado (chega após
    due_time mesmo saindo direto do depósito), marca infeasível.

    Esta estratégia é muito mais eficiente que o decoder sequencial (single-route)
    e produz soluções com número de rotas próximo ao ótimo greedy EDD.

    Referência: Nagata & Bräysy (2009) Networks 54(4):232-245.
    """
    customers = inst['customers']
    capacity  = inst['fleet']['capacity_kg']

    # Estado de cada rota aberta: (load, current_time, last_node)
    r_load: List[float] = []
    r_time: List[float] = []
    r_last: List[int]   = []
    routes: List[List[int]] = []
    feasible = True

    for cid in chromosome:
        c   = customers[cid - 1]
        dem = c['demand_kg']
        rt  = c['ready_time_min']
        due = c['due_date_min']
        st  = c['service_time_min']

        inserted = False
        for k in range(len(routes)):
            if r_load[k] + dem > capacity:
                continue
            travel  = ttime[r_last[k]][cid]
            arrival = r_time[k] + travel
            if arrival > due:
                continue
            wait       = max(0.0, rt - arrival)
            r_load[k] += dem
            r_time[k]  = arrival + wait + st
            r_last[k]  = cid
            routes[k].append(cid)
            inserted = True
            break

        if not inserted:
            # Abre nova rota saindo do depósito
            arr0 = ttime[0][cid]
            if arr0 > due:
                feasible = False
                continue          # cliente não atendível (due_time muito curto)
            wait = max(0.0, rt - arr0)
            r_load.append(dem)
            r_time.append(arr0 + wait + st)
            r_last.append(cid)
            routes.append([cid])

    return routes, feasible


def _cat_index_map(inst: dict) -> Dict[str, int]:
    """
    Mapa categoria térmica → índice fixo (ordem alfabética das chaves de
    thermal_profile.per_category), usado para indexar o bloco 2 (zone_block)
    do cromossomo B2a. Derivado da própria instância, não hardcoded — não
    assume nomes de categoria específicos.
    """
    cats = sorted(inst['thermal_profile']['per_category'].keys())
    return {c: i for i, c in enumerate(cats)}


def assign_zones(
    routes:     List[List[int]],
    zone_block: List[int],
    inst:       dict,
    cat_index:  Dict[str, int],
) -> List[dict]:
    """
    Segunda passada do decodificador (B2a) — atribui cada categoria térmica
    presente em cada rota a uma zona de temperatura, usando o bloco 2 do
    cromossomo. Não reabre nem reordena as rotas produzidas por decode().

    Mapeamento posicional: a i-ésima rota (na ordem de `routes`) usa os genes
    do slot i do bloco 2 — ver docstring de Individual.zone_block.

    Não há reparo guloso aqui (ver pesquisa/plano_implementacao_b2.md §6):
    sob o modelo de capacidade de zona contínua (adotado), a atribuição
    categoria→zona é rotulagem pura — decode() já garante capacidade total
    da rota, e não existe combinação categoria→zona que viole isso.

    Retorna, por rota (mesma ordem/índice de `routes`), um dict:
      {
        'zone_of_category': {categoria: zona, ...}   — só categorias presentes,
        'zone_temp':         {zona: {T_set_C, T_max_C, T_min_C, sigma_T_C,
                                      source_cat}, ...} — só zonas ativas,
                              onde source_cat é a categoria mais fria que
                              caiu na zona (governa ruído/clamps, §4.5.1).
      }
    """
    customers = inst['customers']
    per_cat   = inst['thermal_profile']['per_category']
    n_cat     = len(cat_index)
    result: List[dict] = []

    for i, route in enumerate(routes):
        cats_presentes = sorted({customers[cid - 1]['product_category'] for cid in route})

        zone_of: Dict[str, int] = {}
        for c in cats_presentes:
            gene_idx = i * n_cat + cat_index[c]
            zone_of[c] = zone_block[gene_idx] if gene_idx < len(zone_block) else 0

        zone_temp: Dict[int, dict] = {}
        for c, z in zone_of.items():
            T_set_c = per_cat[c]['T_set_C']
            if z not in zone_temp or T_set_c < zone_temp[z]['T_set_C']:
                zone_temp[z] = {
                    'T_set_C':    T_set_c,
                    'T_max_C':    per_cat[c]['T_max_C'],
                    # per_category não tem T_min_C — mesma convenção de fallback
                    # já usada hoje no nível agregado (linha ~230 abaixo).
                    'T_min_C':    per_cat[c].get('T_min_C', T_set_c - 3.0),
                    'sigma_T_C':  per_cat[c]['sigma_T_C'],
                    'source_cat': c,
                }

        result.append({'zone_of_category': zone_of, 'zone_temp': zone_temp})

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. AVALIAÇÃO DE OBJETIVOS
# ─────────────────────────────────────────────────────────────────────────────

def _route_quality_mc(
    route: List[int],
    inst:  dict,
    ttime: List[List[float]],
    rng:   np.random.Generator,
    mc_samples: int,
) -> float:
    """
    Qualidade residual média ponderada por valor para uma rota, via Monte Carlo.

    Modelo de degradação de Arrhenius de primeira ordem:
      Q(t) = exp(-k(T) · t),   k(T) = A₀ · exp(-Eₐ / (R · T_K))

    Temperatura em cada trecho: T_nominal + ε, ε ~ N(0, σ_T²),
    grampeada em [T_min, T_max] para realismo físico.

    Retorna: E[Σᵢ valueᵢ · Qᵢ] / Σᵢ valueᵢ   (escalar em [0, 1]).
    """
    if not route:
        return 1.0

    customers = inst['customers']
    tp        = inst['thermal_profile']
    sigma_T   = tp['sigma_T_C']
    T_set     = tp['T_set_C']
    T_min     = tp.get('T_min_C', T_set - 3.0)
    T_max     = tp.get('T_max_C', T_set + 5.0)
    Ea        = tp['Ea_J_mol']
    A0        = tp['A0']

    # Sequência de nós: depósito (0) → c1 → c2 → ... → ck
    nodes     = [0] + route
    n_legs    = len(nodes) - 1

    # Tempos de percurso por trecho (minutos → horas)
    leg_t_h   = np.array([ttime[nodes[i]][nodes[i + 1]] / 60.0
                           for i in range(n_legs)])

    # Pesos de valor (para média ponderada)
    values    = np.array([customers[cid - 1].get('load_value_BRL', 1.0)
                           for cid in route])
    total_val = values.sum() or 1.0

    # Modo determinístico (mc_samples=0): temperatura fixa em T_set
    if mc_samples == 0:
        T_K_det = np.full(n_legs, T_set + 273.15)
        k_det   = A0 * np.exp(-Ea / (R_GAS * T_K_det))
        cum_kdt = np.cumsum(k_det * leg_t_h)
        Q_det   = np.exp(-cum_kdt)
        return float((Q_det * values).sum() / total_val)

    # Monte Carlo vetorizado — shape (mc_samples, n_legs)
    eps   = rng.normal(0.0, sigma_T, size=(mc_samples, n_legs))
    T_mat = np.clip(T_set + eps, T_min, T_max)   # temperaturas por trecho

    # Taxa de degradação k(T) por trecho (vetorizado)
    T_K   = T_mat + 273.15
    k_mat = A0 * np.exp(-Ea / (R_GAS * T_K))     # (mc_samples, n_legs)

    # Degradação acumulada: Q_i = exp(-Σ_{j≤i} k_j · t_j)
    kdt         = k_mat * leg_t_h                 # (mc_samples, n_legs)
    cum_kdt     = np.cumsum(kdt, axis=1)          # degradação até cliente i
    Q_mat       = np.exp(-cum_kdt)                # qualidade por cliente

    # Qualidade ponderada por replicação e média
    wq_per_rep = (Q_mat * values).sum(axis=1) / total_val   # (mc_samples,)
    return float(wq_per_rep.mean())


def _route_quality_mc_multizone(
    route:            List[int],
    inst:             dict,
    ttime:            List[List[float]],
    rng:              np.random.Generator,
    mc_samples:       int,
    zone_of_category: Dict[str, int],
    zone_temp:        Dict[int, dict],
) -> float:
    """
    Versão B2 (multi-compartimento) de _route_quality_mc().

    Ponto tecnicamente delicado (ver pesquisa/plano_implementacao_b2.md §7):
    um cliente na zona z está exposto à temperatura de z DESDE A SAÍDA DO
    DEPÓSITO, não só nos trechos entre clientes da mesma zona — o produto
    fica no compartimento o tempo todo, incluindo enquanto o veículo para
    em clientes de OUTRAS zonas no meio do caminho. Por isso a degradação
    acumulada de cada cliente usa TODOS os trechos até sua posição na rota
    (exatamente como o código original), só que com a série de temperatura
    da zona dele e a cinética (Ea, A0) da categoria dele — não mais uma
    cinética/temperatura únicas para a rota inteira.

    Dentro de uma mesma zona, categorias diferentes podem coexistir (duas
    categorias podem cair no mesmo gene de zona) — todas compartilham a
    MESMA série de temperatura amostrada da zona, mas cada uma aplica sua
    própria cinética sobre essa série (agrupamento por zona primeiro, depois
    por categoria dentro da zona, preservando a vetorização Monte Carlo).
    """
    if not route:
        return 1.0

    customers = inst['customers']
    per_cat   = inst['thermal_profile']['per_category']

    nodes   = [0] + route
    n_legs  = len(nodes) - 1
    leg_t_h = np.array([ttime[nodes[i]][nodes[i + 1]] / 60.0 for i in range(n_legs)])

    values    = np.array([customers[cid - 1].get('load_value_BRL', 1.0) for cid in route])
    total_val = values.sum() or 1.0

    client_cat  = [customers[cid - 1]['product_category'] for cid in route]
    client_zone = [zone_of_category[c] for c in client_cat]

    # Modo determinístico (mc_samples=0): temperatura fixa em T_set da zona.
    # Usado de verdade pela variante NSGA2-DET do benchmark, não só em teste.
    if mc_samples == 0:
        Q_row = np.zeros(len(route))
        for z, ztemp in zone_temp.items():
            idxs = [i for i, zz in enumerate(client_zone) if zz == z]
            if not idxs:
                continue
            T_K_det = np.full(n_legs, ztemp['T_set_C'] + 273.15)
            for c in {client_cat[i] for i in idxs}:
                Ea_c, A0_c = per_cat[c]['Ea_J_mol'], per_cat[c]['A0']
                k_det   = A0_c * np.exp(-Ea_c / (R_GAS * T_K_det))
                cum_kdt = np.cumsum(k_det * leg_t_h)
                Q_det   = np.exp(-cum_kdt)
                for i in idxs:
                    if client_cat[i] == c:
                        Q_row[i] = Q_det[i]
        return float((Q_row * values).sum() / total_val)

    # Monte Carlo vetorizado, por zona (série de temperatura) × categoria (cinética)
    Q_mat = np.zeros((mc_samples, len(route)))
    for z, ztemp in zone_temp.items():
        idxs = [i for i, zz in enumerate(client_zone) if zz == z]
        if not idxs:
            continue
        eps   = rng.normal(0.0, ztemp['sigma_T_C'], size=(mc_samples, n_legs))
        T_mat = np.clip(ztemp['T_set_C'] + eps, ztemp['T_min_C'], ztemp['T_max_C'])
        T_K   = T_mat + 273.15   # (mc_samples, n_legs) — série da zona, compartilhada

        for c in {client_cat[i] for i in idxs}:
            Ea_c, A0_c = per_cat[c]['Ea_J_mol'], per_cat[c]['A0']
            k_mat   = A0_c * np.exp(-Ea_c / (R_GAS * T_K))       # (mc_samples, n_legs)
            cum_kdt = np.cumsum(k_mat * leg_t_h, axis=1)         # (mc_samples, n_legs)
            Q_c     = np.exp(-cum_kdt)
            for i in idxs:
                if client_cat[i] == c:
                    Q_mat[:, i] = Q_c[:, i]

    wq_per_rep = (Q_mat * values).sum(axis=1) / total_val
    return float(wq_per_rep.mean())


def _p_ref_zone(T_zone_C: float, P_ref_base: float, T_ref_C: float, gamma: float) -> float:
    """
    Potência de refrigeração de uma zona, extensão de f3 variante (ii)
    (proposta_extensao_ejor.md §4.5.1): cresce linearmente conforme a zona
    fica mais fria que T_ref_C; nunca cai abaixo de P_ref_base (max(0,...)).
    """
    return P_ref_base * (1.0 + gamma * max(0.0, T_ref_C - T_zone_C))


def evaluate(
    ind:        Individual,
    inst:       dict,
    dist:       List[List[float]],
    ttime:      List[List[float]],
    rng:        np.random.Generator,
    mc_samples: int = MC_SAMPLES,
    use_multizone: bool = False,
    n_zones:       int  = 1,
    cat_index:     Optional[Dict[str, int]] = None,
    gamma_zone_cold:         float = GAMMA_ZONE_COLD,
    zone_loading_km_mult:    float = ZONE_LOADING_COST_KM_MULTIPLE,
    zone_unloading_km_mult:  float = ZONE_UNLOADING_COST_KM_MULTIPLE,
):
    """
    Avalia os três objetivos de um indivíduo:
      f1 = custo total (R$)
      f2 = perda de qualidade ponderada por valor (Monte Carlo)
      f3 = emissões de CO₂ (kg CO₂-eq)

    Infeasibilidade é penalizada adicionando PENALTY_INFEAS a f1 e f3.

    use_multizone=False (default): comportamento idêntico ao código anterior
    a B2 — f1 não tem termo de custo por zona (contribuição zero por
    construção, ver zone_cost_route abaixo), compatibilidade retroativa
    preservada.
    use_multizone=True: f2 usa a cinética de Arrhenius da categoria PRÓPRIA
    de cada cliente na temperatura da SUA zona (corrige e generaliza, como
    efeito colateral, o achado original de categoria térmica ignorada); f3
    soma P_ref por zona ativa em vez de um P_ref fixo por veículo; f1 soma um
    termo de custo de carga/descarga por zona ativa (Hübner & Ostermeier 2018
    — ESTIMATIVA DE SENSIBILIDADE na magnitude, estrutura relativa citável;
    ver ZONE_LOADING_FACTOR/ZONE_UNLOADING_FACTOR acima e diário, Entrada 25).

    gamma_zone_cold/zone_loading_km_mult/zone_unloading_km_mult (novos,
    29/08/2026, sensibilidade B2): defaults reproduzem exatamente os módulos-
    constante GAMMA_ZONE_COLD/ZONE_LOADING_COST_KM_MULTIPLE/
    ZONE_UNLOADING_COST_KM_MULTIPLE — só têm efeito com use_multizone=True
    (ambos os ramos abaixo). Nenhum resultado já existente depende de passar
    estes argumentos.
    """
    routes, feasible = decode(ind.chromosome, inst, ttime)
    ind.routes   = routes
    ind.feasible = feasible

    if not routes:
        ind.f_cost  = PENALTY_INFEAS
        ind.f_qloss = 1.0
        ind.f_co2   = PENALTY_INFEAS
        return

    fleet    = inst['fleet']
    co2p     = inst.get('co2_params', {})
    fix_cost = fleet['fixed_cost_BRL']
    var_cost = fleet['variable_cost_BRL_km']
    speed    = fleet['speed_km_h']
    alpha    = co2p.get('alpha_kg_per_km', 0.21)
    beta     = co2p.get('beta_kg_per_kWh', 0.092)
    P_ref    = co2p.get('P_refrigeration_kW', 3.5)

    zones_by_route = (assign_zones(routes, ind.zone_block, inst, cat_index)
                       if use_multizone else None)
    T_ref_C = inst['thermal_profile']['T_set_C'] if use_multizone else None

    customers  = inst['customers']
    total_cost = 0.0
    total_co2  = 0.0
    total_wq   = 0.0
    total_val  = 0.0

    for route_idx, route in enumerate(routes):
        nodes  = [0] + route + [0]
        r_dist = sum(dist[nodes[i]][nodes[i + 1]] for i in range(len(nodes) - 1))
        r_time_h = r_dist / speed

        if use_multizone:
            zinfo   = zones_by_route[route_idx]
            P_ref_route = sum(
                _p_ref_zone(zt['T_set_C'], P_ref, T_ref_C, gamma_zone_cold)
                for zt in zinfo['zone_temp'].values()
            )
            r_wq = _route_quality_mc_multizone(
                route, inst, ttime, rng, mc_samples,
                zinfo['zone_of_category'], zinfo['zone_temp'],
            )
            # Custo de carga/descarga por zona ativa (Hübner & Ostermeier 2018,
            # ver constantes ZONE_*_FACTOR/ZONE_*_KM_MULTIPLE acima). Paga só o
            # INCREMENTO sobre o caso de 1 zona ativa (fator-1 = 0 quando a rota
            # usa 1 zona só, mesmo sob use_multizone=True — sem caso especial
            # necessário). n_zonas_ativas nunca excede n_zones (2 ou 3, por
            # construção de zone_block), então os dicts cobrem todo caso real.
            n_zonas_ativas = len(zinfo['zone_temp'])
            zone_loading_extra = (zone_loading_km_mult * var_cost
                                   * (ZONE_LOADING_FACTOR[n_zonas_ativas] - 1.0))
            zone_unloading_extra = (zone_unloading_km_mult * var_cost
                                     * (ZONE_UNLOADING_FACTOR[n_zonas_ativas] - 1.0)
                                     * len(route))
            zone_cost_route = zone_loading_extra + zone_unloading_extra
        else:
            P_ref_route = P_ref
            r_wq = _route_quality_mc(route, inst, ttime, rng, mc_samples)
            zone_cost_route = 0.0

        total_cost += fix_cost + var_cost * r_dist + zone_cost_route
        total_co2  += alpha * r_dist + beta * P_ref_route * r_time_h

        r_val = sum(customers[cid - 1].get('load_value_BRL', 1.0) for cid in route)
        total_wq  += r_wq * r_val
        total_val += r_val

    wq_global = total_wq / total_val if total_val > 0 else 0.0
    pen = PENALTY_INFEAS if not feasible else 0.0

    # Custo de veiculos extras alem da frota nominal
    n_nominal   = inst['fleet']['num_vehicles']
    n_extra     = max(0, len(routes) - n_nominal)
    extra_cost  = n_extra * EXTRA_VEHICLE_COST

    ind.f_cost  = total_cost + extra_cost + pen
    ind.f_qloss = 1.0 - wq_global
    ind.f_co2   = total_co2 + pen


# ─────────────────────────────────────────────────────────────────────────────
# 6. OPERADORES DO NSGA-II
# ─────────────────────────────────────────────────────────────────────────────

def fast_nondominated_sort(pop: List[Individual]) -> List[List[Individual]]:
    """
    Classificação rápida não-dominada — Deb et al. (2002), Seção III-A.
    Complexidade O(M·N²) onde M = número de objetivos, N = tamanho da pop.
    Retorna lista de fronts (front[0] é a Frente de Pareto).
    """
    n     = len(pop)
    S     = [[] for _ in range(n)]   # S[i]: conjunto dominado por i
    n_dom = [0] * n                  # n_dom[i]: quantos dominam i
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if pop[i].dominates(pop[j]):
                S[i].append(j)
                n_dom[j] += 1
            elif pop[j].dominates(pop[i]):
                S[j].append(i)
                n_dom[i] += 1
        if n_dom[i] == 0:
            pop[i].rank = 1
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in S[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    pop[j].rank = k + 2
                    nxt.append(j)
        fronts.append(nxt)
        k += 1

    return [[pop[i] for i in front] for front in fronts if front]


def crowding_distance_assign(front: List[Individual]):
    """
    Atribui distância de aglomeração (crowding distance) aos indivíduos de um front.
    Soluções extremas recebem distância infinita.
    """
    n = len(front)
    for ind in front:
        ind.crowding = 0.0
    if n <= 2:
        for ind in front:
            ind.crowding = float('inf')
        return

    for get_f in (lambda x: x.f_cost, lambda x: x.f_qloss, lambda x: x.f_co2):
        sf      = sorted(front, key=get_f)
        sf[0].crowding  = float('inf')
        sf[-1].crowding = float('inf')
        f_range = get_f(sf[-1]) - get_f(sf[0])
        if f_range == 0:
            continue
        for i in range(1, n - 1):
            sf[i].crowding += (get_f(sf[i + 1]) - get_f(sf[i - 1])) / f_range


def tournament_select(pop: List[Individual], rng_py: random.Random) -> Individual:
    """Seleção por torneio binário (rank menor → melhor; empate → crowding maior)."""
    candidates = rng_py.sample(pop, TOURNAMENT_K)
    best = candidates[0]
    for c in candidates[1:]:
        if (c.rank < best.rank
                or (c.rank == best.rank and c.crowding > best.crowding)):
            best = c
    return best


def ox_crossover(
    p1: Individual, p2: Individual, rng_py: random.Random
) -> Individual:
    """
    Order Crossover (OX) para permutações [Davis, 1985].
    Preserva uma sub-sequência de p1 e preenche o restante com a ordem de p2.
    """
    n    = len(p1.chromosome)
    a, b = sorted(rng_py.sample(range(n), 2))

    child = [-1] * n
    child[a : b + 1] = p1.chromosome[a : b + 1]
    segment = set(child[a : b + 1])

    fill = [g for g in p2.chromosome if g not in segment]
    pos  = [(b + 1 + i) % n for i in range(n - (b - a + 1))]
    for i, p in enumerate(pos):
        child[p] = fill[i]

    return Individual(chromosome=child)


def mutate(ind: Individual, rng_py: random.Random, operator: str = "mix") -> Individual:
    """
    Mutação com dois operadores:
      - 2-opt intra: inverte uma sub-sequência (explora vizinhança local)
      - or-opt(1): remove e reinsere um gene em posição aleatória
    Implementação baseada em Laporte et al. (2000).

    operator (novo, frente 4.3 componente 3, 29/08/2026):
      - "mix" (default, comportamento original): 50/50 por sorteio, byte-a-
        byte idêntico ao código anterior a este parâmetro existir.
      - "2opt": sempre 2-opt.
      - "oropt": sempre or-opt(1).
    """
    chrom = ind.chromosome[:]
    n     = len(chrom)
    if n < 3:
        return Individual(chromosome=chrom)

    if operator == "2opt":
        usar_2opt = True
    elif operator == "oropt":
        usar_2opt = False
    elif operator == "mix":
        usar_2opt = rng_py.random() < 0.5
    else:
        raise ValueError(f"mutate(): operator inválido {operator!r} (use 'mix'/'2opt'/'oropt')")

    if usar_2opt:
        # 2-opt: inversão de sub-segmento
        i, j = sorted(rng_py.sample(range(n), 2))
        chrom[i : j + 1] = chrom[i : j + 1][::-1]
    else:
        # or-opt(1): relocação de elemento
        i    = rng_py.randrange(n)
        gene = chrom.pop(i)
        j    = rng_py.randrange(n)
        chrom.insert(j, gene)

    return Individual(chromosome=chrom)


def zone_block_crossover(
    p1_block: List[int], p2_block: List[int], rng_py: random.Random
) -> List[int]:
    """
    Crossover uniforme do bloco 2 (B2a) — por gene, 50% de chance de herdar
    de cada pai. zone_block não é uma permutação (valores pequenos, repetição
    normal), então o OX do bloco 1 não se aplica; este é o operador padrão
    para vetores de alfabeto pequeno e tamanho fixo (sem análogo direto em
    Koch et al. 2016, cujo crossover doador/receptor foi desenhado para o
    cromossomo fundido deles, não para este bloco auxiliar).
    """
    return [p1_block[i] if rng_py.random() < 0.5 else p2_block[i]
            for i in range(len(p1_block))]


def zone_block_mutate(
    block: List[int],
    n_zones: int,
    rng_py: random.Random,
    p_swap: float = 0.05,
    p_ins: float = 0.05,
) -> List[int]:
    """
    Mutação do bloco 2 (B2a) — probabilidades herdadas de Koch et al. (2016,
    p_swap=p_inv=0,05), concretizadas para um vetor não-permutação:
      - swap: troca o valor do gene com o de outro gene sorteado.
      - insertion: reamostra o valor do gene uniformemente em {0,...,Z-1}
        (uma "reinserção" não muda estrutura num vetor sem ordem relevante,
        então foi concretizada como resample — ver plano de implementação §5).
    As duas são independentes por gene (podem ambas, uma, ou nenhuma ocorrer).
    """
    b = block[:]
    n = len(b)
    for i in range(n):
        if rng_py.random() < p_swap:
            j = rng_py.randrange(n)
            b[i], b[j] = b[j], b[i]
        if rng_py.random() < p_ins:
            b[i] = rng_py.randrange(n_zones)
    return b


# ─────────────────────────────────────────────────────────────────────────────
# 7. INICIALIZAÇÃO DA POPULAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def _nn_chromosome(
    inst: dict,
    dist: List[List[float]],
    rng_py: random.Random,
) -> List[int]:
    """
    Heurística de vizinho mais próximo com perturbação biased-randomized.
    Com probabilidade 0.15, escolhe aleatoriamente entre os 3 clientes
    mais próximos (em vez do mais próximo exato).
    Ref.: Juan et al. (2021), Seção 3.
    """
    n         = len(inst['customers'])
    unvisited = list(range(1, n + 1))
    rng_py.shuffle(unvisited)
    chrom   = []
    current = 0

    while unvisited:
        by_dist = sorted(unvisited, key=lambda c: dist[current][c])
        k       = min(3, len(by_dist))
        chosen  = rng_py.choice(by_dist[:k]) if rng_py.random() < 0.15 else by_dist[0]
        chrom.append(chosen)
        unvisited.remove(chosen)
        current = chosen

    return chrom


def _edd_chromosome(inst: dict) -> List[int]:
    """
    Cromossomo ordenado por Earliest Due Date (EDD).
    Garante que o decodificador produza poucas rotas respeitando janelas.
    Usado para semear a população inicial com soluções estruturalmente boas.
    """
    customers = inst['customers']
    return sorted(range(1, len(customers) + 1),
                  key=lambda c: customers[c - 1]['due_date_min'])


def _edd_perturbed(inst: dict, rng_py: random.Random, perturb: float = 0.15) -> List[int]:
    """
    EDD com perturbação: cada cliente tem probabilidade `perturb` de trocar
    posição com um vizinho aleatório. Cria diversidade mantendo estrutura EDD.
    """
    chrom = _edd_chromosome(inst)
    n = len(chrom)
    for i in range(n):
        if rng_py.random() < perturb:
            j = rng_py.randrange(n)
            chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom


def _init_zone_block(
    n_customers: int, n_categories: int, n_zones: int, rng_py: random.Random
) -> List[int]:
    """
    Inicializa o bloco 2 (B2a) uniformemente ao acaso em {0,...,n_zones-1}.
    Tamanho n_customers × n_categories — cota superior segura, já que o
    número de rotas produzidas por decode() nunca excede o número de
    clientes (ver pesquisa/plano_implementacao_b2.md §4). Independente de
    qual heurística gerou o bloco 1 (EDD/NN/aleatório) — não há heurística
    "esperta" de zona ainda; possível refinamento futuro, não bloqueia B2.
    """
    size = n_customers * n_categories
    return [rng_py.randrange(n_zones) for _ in range(size)]


def init_population(
    inst:    dict,
    dist:    List[List[float]],
    ttime:   List[List[float]],
    pop_size: int,
    rng_py:  random.Random,
    rng_np:  np.random.Generator,
    mc_samples:    int  = MC_SAMPLES,
    use_multizone: bool = False,
    n_zones:       int  = 1,
    cat_index:     Optional[Dict[str, int]] = None,
    init_mix:      Optional[Tuple[float, float, float]] = None,
    gamma_zone_cold:         float = GAMMA_ZONE_COLD,
    zone_loading_km_mult:    float = ZONE_LOADING_COST_KM_MULTIPLE,
    zone_unloading_km_mult:  float = ZONE_UNLOADING_COST_KM_MULTIPLE,
) -> List[Individual]:
    """
    Inicializa população com três componentes:
      - 20% EDD perturbado (soluções estruturalmente eficientes em janelas de tempo)
      - 40% vizinho mais próximo biased-randomized (Solomon-style)
      - 40% permutação aleatória pura (diversidade)

    Toda solução é avaliada antes de retornar.

    use_multizone=False (default): comportamento idêntico ao código anterior
    a B2 — mc_samples aqui só existe para ser repassado a evaluate() de forma
    explícita (antes vinha implícito do default da própria evaluate()).

    init_mix (novo, frente 4.3 componente 4, 29/08/2026): tupla
    (frac_edd, frac_nn, frac_rnd) somando 1.0. Default None preserva EXATAMENTE
    a fórmula original (n_edd=pop_size//5, n_nn=pop_size//2-n_edd,
    n_rnd=resto) -- não só "20/40/40 aproximado", para não mudar nenhum
    resultado já existente por arredondamento diferente.
    """
    n = len(inst['customers'])
    if init_mix is None:
        n_edd  = max(1, pop_size // 5)         # 20% EDD perturbado
        n_nn   = pop_size // 2 - n_edd         # 40% vizinho mais próximo
        n_rnd  = pop_size - n_edd - n_nn       # 40% aleatório
    else:
        frac_edd, frac_nn, frac_rnd = init_mix
        if abs(frac_edd + frac_nn + frac_rnd - 1.0) > 1e-6:
            raise ValueError(f"init_mix precisa somar 1.0, recebeu {init_mix} (soma={frac_edd+frac_nn+frac_rnd})")
        n_edd  = max(1, round(pop_size * frac_edd)) if frac_edd > 0 else 0
        n_nn   = round(pop_size * frac_nn)
        n_rnd  = pop_size - n_edd - n_nn       # resto absorve arredondamento, sempre soma pop_size
    pop    = []
    n_cat  = len(cat_index) if cat_index else 0

    def _make(chrom: List[int]) -> Individual:
        ind = Individual(chromosome=chrom)
        if use_multizone:
            ind.zone_block = _init_zone_block(n, n_cat, n_zones, rng_py)
        evaluate(ind, inst, dist, ttime, rng_np, mc_samples,
                 use_multizone=use_multizone, n_zones=n_zones, cat_index=cat_index,
                 gamma_zone_cold=gamma_zone_cold,
                 zone_loading_km_mult=zone_loading_km_mult,
                 zone_unloading_km_mult=zone_unloading_km_mult)
        return ind

    # Soluções EDD perturbadas
    for _ in range(n_edd):
        pop.append(_make(_edd_perturbed(inst, rng_py)))

    # Soluções vizinho mais próximo
    for _ in range(n_nn):
        pop.append(_make(_nn_chromosome(inst, dist, rng_py)))

    # Soluções aleatórias puras
    for _ in range(n_rnd):
        chrom = list(range(1, n + 1))
        rng_py.shuffle(chrom)
        pop.append(_make(chrom))

    return pop


# ─────────────────────────────────────────────────────────────────────────────
# 8. MÉTRICAS DE QUALIDADE MULTIOBJETIVO
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(pareto: List[Individual]) -> np.ndarray:
    """Normaliza objetivos da frente para [0, 1] pelo range observado."""
    objs = np.array([ind.obj() for ind in pareto], dtype=float)
    mn   = objs.min(axis=0)
    mx   = objs.max(axis=0)
    rng  = np.where(mx > mn, mx - mn, 1.0)
    return (objs - mn) / rng


def _nondom_2d(pts: list) -> list:
    """Pontos não-dominados em 2D (minimização), ordenados por f1."""
    pts_s   = sorted(pts, key=lambda p: (p[0], p[1]))
    nd      = []
    best_f2 = float('inf')
    for p in pts_s:
        if p[1] < best_f2:
            nd.append(p)
            best_f2 = p[1]
    return nd


def _hv2d(pts: list, ref: np.ndarray) -> float:
    """Hipervolume 2D por varredura (sweep line) em relação ao ponto de referência."""
    hv      = 0.0
    prev_f1 = ref[0]
    for p in sorted(pts, key=lambda x: x[0], reverse=True):
        if p[0] < prev_f1 and p[1] < ref[1]:
            hv     += (prev_f1 - p[0]) * (ref[1] - p[1])
            prev_f1 = p[0]
    return hv


def compute_hypervolume(
    pareto:    List[Individual],
    ref_point: Optional[Tuple] = None,
) -> float:
    """
    Hipervolume 3D (indicador S de Zitzler & Thiele, 1999).
    Algoritmo: decomposição por fatias em f3 com HV 2D por varredura.
    Ponto de referência: nadir × 1.1 (ou valor fornecido pelo usuário).

    CORRIGIDO em 22/08/2026 (diário de pesquisa, Entradas 11/13/15) — a versão
    anterior varria os pontos em ordem DECRESCENTE de f3 e acumulava o front 2D
    nessa mesma ordem, o que atribuía a cada fatia um front incompleto (fatias
    "de cima", perto do ponto de referência, deveriam contar com TODOS os
    pontos, já que a caixa de qualquer ponto se estende dele até a referência).
    Isso fazia pontos não-dominados em 3D, mas 2D-dominados por outro ponto de
    f3 maior, perderem contribuição em fatias onde a caixa do "dominador" nem
    alcançava. Confirmado por 3 métodos independentes (inclusão-exclusão exata,
    Monte Carlo, pymoo.indicators.hv.HV) que a versão anterior divergia em
    ~13-17% em média nas 1.080 execuções reais do benchmark (até 59% no pior
    caso). A correção varre em ordem CRESCENTE de f3, atribuindo a cada ponto
    processado a fatia que vai dele até o PRÓXIMO ponto (ou até a referência,
    para o último) — testado e validado contra pymoo/inclusão-exclusão/Monte
    Carlo com diferença < 1e-6.
    """
    if len(pareto) < 2:
        return 0.0

    objs = np.array([ind.obj() for ind in pareto], dtype=float)
    ref  = (objs.max(axis=0) * 1.1 + 1e-9
            if ref_point is None else np.array(ref_point, dtype=float))

    # Ordena por f3 CRESCENTE — o front 2D acumulado cresce à medida que
    # avançamos, e cada ponto "entra" no front exatamente na fatia que
    # começa na sua própria coordenada de f3.
    idx      = np.argsort(objs[:, 2])
    sorted_o = objs[idx]
    n        = len(sorted_o)

    hv       = 0.0
    front_2d = []

    for i, row in enumerate(sorted_o):
        front_2d.append(row[:2])
        front_2d = _nondom_2d(front_2d)
        hv2d     = _hv2d(front_2d, ref[:2])
        next_f3  = sorted_o[i + 1][2] if i + 1 < n else ref[2]
        delta    = next_f3 - row[2]
        hv      += hv2d * max(delta, 0.0)

    return float(hv)


def compute_gd(
    pareto:   List[Individual],
    ref_set:  List[Individual],
) -> float:
    """
    Generational Distance (GD) — distância média da frente calculada
    ao conjunto de referência, usando objetivos normalizados.
    GD = (1/|P|) Σᵢ min_{r ∈ R} ‖pᵢ − r‖₂
    """
    if not pareto or not ref_set:
        return float('inf')
    p_objs = _normalize(pareto)
    r_objs = _normalize(ref_set)
    dists  = [min(float(np.linalg.norm(p - r)) for r in r_objs) for p in p_objs]
    return float(np.mean(dists))


def compute_spread(pareto: List[Individual]) -> float:
    """
    Spread (Δ) — diversidade da frente de Pareto.
    Métrica de Deb et al. (2002), estendida para 3 objetivos:
    as soluções são ordenadas por f1 (custo) e as distâncias entre
    vizinhos consecutivos são comparadas com sua média.

    Δ = (d_f + d_l + Σ|dᵢ − d̄|) / (d_f + d_l + (n−1)d̄)
    """
    if len(pareto) < 3:
        return 0.0

    objs_s = _normalize(pareto)
    objs_s = objs_s[np.argsort(objs_s[:, 0])]
    dists  = [float(np.linalg.norm(objs_s[i + 1] - objs_s[i]))
               for i in range(len(objs_s) - 1)]
    if not dists:
        return 0.0

    d_bar = float(np.mean(dists))
    d_f, d_l = dists[0], dists[-1]
    num  = d_f + d_l + sum(abs(d - d_bar) for d in dists)
    den  = d_f + d_l + len(dists) * d_bar
    return float(num / den) if den > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 9. LOOP PRINCIPAL DO NSGA-II
# ─────────────────────────────────────────────────────────────────────────────

def run_nsga2(
    inst:       dict,
    pop_size:   int  = POP_SIZE,
    n_gen:      int  = N_GEN,
    mc_samples: int  = MC_SAMPLES,
    seed:       int  = 0,
    verbose:    bool = True,
    use_multizone: bool = False,
    n_zones:       int  = 1,
    crossover_rate:    float = CROSSOVER_RATE,
    mutation_operator: str   = "mix",
    init_mix: Optional[Tuple[float, float, float]] = None,
    gamma_zone_cold:         float = GAMMA_ZONE_COLD,
    zone_loading_km_mult:    float = ZONE_LOADING_COST_KM_MULTIPLE,
    zone_unloading_km_mult:  float = ZONE_UNLOADING_COST_KM_MULTIPLE,
    hv_ref_point: Optional[Tuple] = None,
    dedup: bool = False,
) -> Tuple[List[Individual], List[Tuple[int, float]]]:
    """
    Executa o NSGA-II com Monte Carlo embarcado para o CCVRPTW-ST.

    dedup (novo, 29/09/2026): elimina duplicatas estruturais (mesmas rotas)
    na sobrevivência — ver _split_duplicates(). Default False reproduz o
    comportamento de sempre byte a byte.

    A cada 10 gerações calcula o Hipervolume da frente corrente.

    hv_ref_point (novo, 06/09/2026 -- diário Entrada 52, curvas de
    convergência): ponto de referência FIXO opcional para o HV de checkpoint.
    Default None preserva o comportamento de sempre -- compute_hypervolume()
    recalcula o ponto de referência a cada checkpoint a partir do máximo da
    própria frente corrente (auto-referenciado), o que NÃO é comparável entre
    gerações (curva não-monotônica, verificado). Passar um ponto fixo (ex.:
    o mesmo 'ref_point_global' por instância que reprocessar_hipervolume.py
    já usa pro HV final comparável) produz uma curva de convergência válida.
    Não afeta a busca em si (mesma seed -> mesma frente final byte a byte,
    verificado) -- só a métrica de HV registrada em cada checkpoint.

    use_multizone/n_zones (B2, opcional): liga o cromossomo B2a (bloco 2 de
    atribuição categoria→zona) e a avaliação multi-compartimento. Default
    (False/1) preserva o comportamento anterior a B2 byte-a-byte — nenhum
    resultado já reprocessado nesta pesquisa depende deste parâmetro.

    crossover_rate/mutation_operator/init_mix (novos, frente 4.3 componentes
    2/3/4, 29/08/2026): defaults reproduzem exatamente o comportamento
    anterior a estes parâmetros existirem (CROSSOVER_RATE=0.90, mutação
    50/50 2-opt/or-opt, mix de inicialização 20/40/40 original) — nenhum
    resultado já existente depende de passar estes argumentos.

    gamma_zone_cold/zone_loading_km_mult/zone_unloading_km_mult (novos,
    29/08/2026, sensibilidade B2): só têm efeito com use_multizone=True;
    defaults reproduzem exatamente GAMMA_ZONE_COLD/ZONE_LOADING_COST_KM_MULTIPLE/
    ZONE_UNLOADING_COST_KM_MULTIPLE.

    Retorna:
      pareto     — frente de Pareto final
      hv_history — lista de (geração, HV) a cada 10 gerações
    """
    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)

    dist, ttime = build_matrices(inst)
    cat_index = _cat_index_map(inst) if use_multizone else None

    if verbose:
        print(f"  Inicializando população ({pop_size} indivíduos)...", flush=True)
    pop = init_population(inst, dist, ttime, pop_size, rng_py, rng_np, mc_samples,
                           use_multizone=use_multizone, n_zones=n_zones, cat_index=cat_index,
                           init_mix=init_mix,
                           gamma_zone_cold=gamma_zone_cold,
                           zone_loading_km_mult=zone_loading_km_mult,
                           zone_unloading_km_mult=zone_unloading_km_mult)

    hv_history: List[Tuple[int, float]] = []
    t0 = time.time()

    for gen in range(1, n_gen + 1):
        # Classifica a população atual
        fronts = fast_nondominated_sort(pop)
        for front in fronts:
            crowding_distance_assign(front)

        # Geração de filhos
        offspring: List[Individual] = []
        while len(offspring) < pop_size:
            p1 = tournament_select(pop, rng_py)
            p2 = tournament_select(pop, rng_py)
            do_cx = rng_py.random() < crossover_rate
            child = ox_crossover(p1, p2, rng_py) if do_cx \
                    else Individual(chromosome=p1.chromosome[:])
            if rng_py.random() < MUTATION_RATE:
                child = mutate(child, rng_py, operator=mutation_operator)

            # Bloco 2 (B2a): cruza junto com o bloco 1 (mesmo do_cx — mesmo par
            # de pais/filho, ver plano de implementação §5), mutação sempre
            # tentada (suas próprias probabilidades p_swap/p_ins já são a taxa,
            # sem gate extra de MUTATION_RATE — diferente do bloco 1 de propósito).
            if use_multizone:
                child.zone_block = zone_block_crossover(p1.zone_block, p2.zone_block, rng_py) \
                        if do_cx else p1.zone_block[:]
                child.zone_block = zone_block_mutate(child.zone_block, n_zones, rng_py)

            evaluate(child, inst, dist, ttime, rng_np, mc_samples,
                     use_multizone=use_multizone, n_zones=n_zones, cat_index=cat_index,
                     gamma_zone_cold=gamma_zone_cold,
                     zone_loading_km_mult=zone_loading_km_mult,
                     zone_unloading_km_mult=zone_unloading_km_mult)
            offspring.append(child)

        # União (pais + filhos) e seleção elitista — Deb et al. (2002), Seção III-C
        combined = pop + offspring
        dups: List[Individual] = []
        if dedup:
            combined, dups = _split_duplicates(combined, use_multizone,
                                               len(cat_index) if cat_index else 0)
        fronts   = fast_nondominated_sort(combined)
        for front in fronts:
            crowding_distance_assign(front)

        new_pop: List[Individual] = []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend(front)
            else:
                remaining = pop_size - len(new_pop)
                sf = sorted(front, key=lambda x: -x.crowding)
                new_pop.extend(sf[:remaining])
                break
        # dedup: só completa com duplicatas se faltarem soluções distintas;
        # elas recebem o pior posto possível (nunca entram na frente final e
        # perdem qualquer torneio contra uma solução distinta).
        for d in dups:
            if len(new_pop) >= pop_size:
                break
            d.rank, d.crowding = DUP_RANK, 0.0
            new_pop.append(d)
        pop = new_pop

        # Hipervolume a cada 10 gerações
        if gen % 10 == 0 or gen == n_gen:
            pf = [ind for ind in pop if ind.rank == 1]
            hv = compute_hypervolume(pf, ref_point=hv_ref_point)
            hv_history.append((gen, hv))
            if verbose:
                elapsed = time.time() - t0
                n_feas  = sum(1 for ind in pf if ind.feasible)
                print(
                    f"  Gen {gen:4d}/{n_gen} | Pareto: {len(pf):3d} sol "
                    f"({n_feas} viáveis) | HV: {hv:.4e} | {elapsed:.1f}s",
                    flush=True,
                )

    pareto = [ind for ind in pop if ind.rank == 1]
    return pareto, hv_history


DUP_RANK = 10**6


def _split_duplicates(
    pop: List[Individual], use_multizone: bool, n_cat: int,
) -> Tuple[List[Individual], List[Individual]]:
    """
    Separa soluções estruturalmente duplicadas (29/09/2026, ver
    pesquisa/plano_reavaliacao_oos.md). Duas soluções são duplicatas se
    decodificam nas MESMAS rotas (e, em B2, na mesma atribuição de zonas das
    rotas usadas) — cromossomos diferentes podem gerar a mesma solução.
    Critério estrutural, não por vetor de objetivos, para valer igualmente
    para as variantes determinística e Monte Carlo (nestas, o ruído faz a
    mesma solução receber f2 ligeiramente diferente, escondendo o clone).
    Mantém a primeira ocorrência (pais vêm antes dos filhos em `pop`).
    """
    seen = set()
    unique: List[Individual] = []
    dups: List[Individual] = []
    for ind in pop:
        sig = tuple(tuple(r) for r in ind.routes)
        if use_multizone:
            sig = (sig, tuple(ind.zone_block[:len(ind.routes) * n_cat]))
        if sig in seen:
            dups.append(ind)
        else:
            seen.add(sig)
            unique.append(ind)
    return unique, dups


# ─────────────────────────────────────────────────────────────────────────────
# 9b. LOOP PRINCIPAL DO MOEA/D (baseline externo, frente 4.1, 23/08/2026)
# ─────────────────────────────────────────────────────────────────────────────
# Ver pesquisa/plano_implementacao_moead.md para o desenho completo e
# pesquisa/proposta_extensao_ejor.md §3.2 para o papel desta comparação
# (checkpoint de robustez, não tentativa de "vencer o SOTA" — não existe SOTA
# publicado para a combinação exata do CCVRPTW-ST). Reaproveita literalmente
# a mesma representação/decodificador/avaliação/operadores do NSGA-II acima —
# só a lógica de seleção/atualização muda (decomposição em vez de Pareto +
# crowding).

def _das_dennis_weights(n_obj: int, h: int) -> np.ndarray:
    """
    Malha sistemática de Das & Dennis (simplex-lattice) — vetores de peso
    uniformemente distribuídos no simplex de n_obj objetivos, h divisões.
    Gera C(h+n_obj-1, n_obj-1) vetores (h=13, n_obj=3 -> C(15,2)=105).
    Ver ressalva de citação junto a MOEAD_H acima.
    """
    def _recurse(k: int, remaining: int) -> List[List[int]]:
        if k == 1:
            return [[remaining]]
        return [[i] + tail
                for i in range(remaining + 1)
                for tail in _recurse(k - 1, remaining - i)]

    combos = _recurse(n_obj, h)
    return np.array(combos, dtype=float) / h


def _compute_neighbors(weights: np.ndarray, T: int) -> List[List[int]]:
    """
    Índices dos T vetores de peso mais próximos de cada subproblema
    (distância euclidiana), incluindo o próprio índice — convenção padrão
    do MOEA/D. Calculado uma única vez (os pesos são fixos; só o conteúdo
    de cada posição da população muda ao longo das gerações).
    """
    n = len(weights)
    neighbors = []
    for i in range(n):
        dists = np.linalg.norm(weights - weights[i], axis=1)
        neighbors.append(np.argsort(dists)[:T].tolist())
    return neighbors


def _nondominated_subset(pop: List[Individual]) -> List[Individual]:
    """
    Subconjunto não-dominado de uma população — reaproveita
    fast_nondominated_sort() só como filtro final (o MOEA/D não usa
    ordenação não-dominada durante a busca, só a decomposição escalar).
    """
    fronts = fast_nondominated_sort(pop)
    return fronts[0] if fronts else []


def run_moead(
    inst:       dict,
    pop_size:   int  = POP_SIZE,
    n_gen:      int  = N_GEN,
    mc_samples: int  = MC_SAMPLES,
    seed:       int  = 0,
    verbose:    bool = True,
    use_multizone: bool = False,
    n_zones:       int  = 1,
    hv_ref_point: Optional[Tuple] = None,
) -> Tuple[List[Individual], List[Tuple[int, float]]]:
    """
    Executa o MOEA/D (Zhang & Li, 2007) com o mesmo Monte Carlo embarcado do
    NSGA-II — decomposição Tchebycheff, vizinhança MOEAD_T, mating
    restriction MOEAD_DELTA, limite de atualização MOEAD_NR. Baseline
    externo (frente 4.1): checkpoint de robustez do NSGA-II+MC, não
    tentativa de "vencer o SOTA" (proposta_extensao_ejor.md §3.2).

    Reaproveita literalmente decode()/evaluate(), os operadores genéticos
    (ox_crossover, mutate, zone_block_crossover/mutate) e init_population()
    já usados pelo NSGA-II — só a seleção/atualização muda.

    pop_size é IGNORADO como valor livre: o nº real de subproblemas vem da
    malha de Das-Dennis (MOEAD_H=13 -> 105 vetores de peso p/ 3 objetivos).
    Mantido como parâmetro só por compatibilidade de assinatura com
    run_nsga2()/_worker() em experiment_runner.py.

    use_multizone/n_zones (B2, opcional): mesmo significado de run_nsga2().

    hv_ref_point (novo, 06/09/2026 -- diário Entrada 52): mesmo significado
    de run_nsga2() -- ponto de referência fixo opcional pro HV de checkpoint;
    default None preserva o comportamento auto-referenciado de sempre.

    Retorna:
      pareto     — subconjunto não-dominado da população final
      hv_history — lista de (geração, HV) a cada 10 gerações
    """
    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)

    dist, ttime = build_matrices(inst)
    cat_index = _cat_index_map(inst) if use_multizone else None

    weights   = _das_dennis_weights(n_obj=3, h=MOEAD_H)
    n_sub     = len(weights)
    T         = min(MOEAD_T, n_sub)
    neighbors = _compute_neighbors(weights, T)

    if verbose:
        print(f"  Inicializando MOEA/D ({n_sub} subproblemas, T={T})...", flush=True)
    pop = init_population(inst, dist, ttime, n_sub, rng_py, rng_np, mc_samples,
                           use_multizone=use_multizone, n_zones=n_zones, cat_index=cat_index)

    # z_star (ideal, min por objetivo) e z_nadir (pior por objetivo, só entre
    # VIAVEIS) -- normalização necessária porque os 3 objetivos têm escalas
    # brutalmente diferentes (custo ~R$5.000, qloss ~0-0,02, CO2 ~100-300kg);
    # sem normalizar, o max() do Tchebycheff é dominado quase sempre pelo
    # custo, colapsando a frente (achado real: pareto_size=3/105 numa
    # instância n=50 real, antes deste ajuste -- ver diario, Entrada 29).
    # z_nadir só considera indivíduos VIÁVEIS -- senão a penalidade de
    # infactibilidade (PENALTY_INFEAS=1e9 em f_cost/f_co2) contamina a escala
    # inteira. z_star não precisa dessa restrição (min() já ignora a
    # penalidade alta naturalmente).
    z_star = np.array([
        min(ind.f_cost  for ind in pop),
        min(ind.f_qloss for ind in pop),
        min(ind.f_co2   for ind in pop),
    ])
    _feas_init = [ind for ind in pop if ind.feasible] or pop  # fallback se nada viável
    z_nadir = np.array([
        max(ind.f_cost  for ind in _feas_init),
        max(ind.f_qloss for ind in _feas_init),
        max(ind.f_co2   for ind in _feas_init),
    ])

    def _tcheby(ind: Individual, w: np.ndarray) -> float:
        f = np.array([ind.f_cost, ind.f_qloss, ind.f_co2])
        denom = np.maximum(z_nadir - z_star, 1e-9)
        f_norm = (f - z_star) / denom
        return float(np.max(w * f_norm))

    hv_history: List[Tuple[int, float]] = []
    t0 = time.time()

    for gen in range(1, n_gen + 1):
        for i in range(n_sub):
            pool   = neighbors[i] if rng_py.random() < MOEAD_DELTA else list(range(n_sub))
            i1, i2 = rng_py.sample(pool, 2)
            p1, p2 = pop[i1], pop[i2]

            do_cx = rng_py.random() < CROSSOVER_RATE
            child = ox_crossover(p1, p2, rng_py) if do_cx \
                    else Individual(chromosome=p1.chromosome[:])
            if rng_py.random() < MUTATION_RATE:
                child = mutate(child, rng_py)

            if use_multizone:
                child.zone_block = zone_block_crossover(p1.zone_block, p2.zone_block, rng_py) \
                        if do_cx else p1.zone_block[:]
                child.zone_block = zone_block_mutate(child.zone_block, n_zones, rng_py)

            evaluate(child, inst, dist, ttime, rng_np, mc_samples,
                     use_multizone=use_multizone, n_zones=n_zones, cat_index=cat_index)

            z_star[0] = min(z_star[0], child.f_cost)
            z_star[1] = min(z_star[1], child.f_qloss)
            z_star[2] = min(z_star[2], child.f_co2)
            if child.feasible:
                z_nadir[0] = max(z_nadir[0], child.f_cost)
                z_nadir[1] = max(z_nadir[1], child.f_qloss)
                z_nadir[2] = max(z_nadir[2], child.f_co2)

            update_order = neighbors[i][:]
            rng_py.shuffle(update_order)
            n_updated = 0
            for j in update_order:
                if n_updated >= MOEAD_NR:
                    break
                if _tcheby(child, weights[j]) <= _tcheby(pop[j], weights[j]):
                    pop[j] = child
                    n_updated += 1

        if gen % 10 == 0 or gen == n_gen:
            nd = _nondominated_subset(pop)
            hv = compute_hypervolume(nd, ref_point=hv_ref_point)
            hv_history.append((gen, hv))
            if verbose:
                elapsed = time.time() - t0
                n_feas  = sum(1 for ind in nd if ind.feasible)
                print(
                    f"  Gen {gen:4d}/{n_gen} | Pareto: {len(nd):3d} sol "
                    f"({n_feas} viáveis) | HV: {hv:.4e} | {elapsed:.1f}s",
                    flush=True,
                )

    pareto = _nondominated_subset(pop)
    return pareto, hv_history


# ─────────────────────────────────────────────────────────────────────────────
# 10. SAÍDA DE RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    pareto:     List[Individual],
    hv_history: List[Tuple[int, float]],
    metrics:    Dict,
    inst_name:  str,
    output_dir: str,
):
    """Salva frente de Pareto (CSV), histórico de HV (CSV) e métricas (JSON)."""
    os.makedirs(output_dir, exist_ok=True)

    # Frente de Pareto
    pf_path = os.path.join(output_dir, f"{inst_name}_pareto.csv")
    with open(pf_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['sol_id', 'f1_cost_BRL', 'f2_quality_loss',
                    'f3_co2_kg', 'n_routes', 'feasible', 'routes'])
        for i, ind in enumerate(sorted(pareto, key=lambda x: x.f_cost)):
            w.writerow([i + 1, round(ind.f_cost, 2), round(ind.f_qloss, 6),
                        round(ind.f_co2, 4), len(ind.routes), ind.feasible,
                        str(ind.routes)])

    # Histórico de hipervolume
    hv_path = os.path.join(output_dir, f"{inst_name}_hv_history.csv")
    with open(hv_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['generation', 'hypervolume'])
        for gen, hv in hv_history:
            w.writerow([gen, round(hv, 8)])

    # Métricas finais
    m_path = os.path.join(output_dir, f"{inst_name}_metrics.json")
    with open(m_path, 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    print(f"  Salvo: {pf_path}")
    print(f"         {hv_path}")
    print(f"         {m_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. RUNNER POR INSTÂNCIA
# ─────────────────────────────────────────────────────────────────────────────

def run_instance(
    instance_path: str,
    output_dir:    str,
    seed:          int  = 0,
    pop_size:      int  = POP_SIZE,
    n_gen:         int  = N_GEN,
    mc_samples:    int  = MC_SAMPLES,
    verbose:       bool = True,
    use_multizone: bool = False,
    n_zones:       int  = 1,
) -> Tuple[List[Individual], Dict]:
    """Carrega instância, executa NSGA-II e salva resultados."""
    inst      = load_instance(instance_path)
    inst_name = inst['name']

    if verbose:
        print(f"\n{'='*65}")
        print(f"  Instância : {inst_name}")
        print(f"  Família   : {inst['family']}   n={inst['size']}   "
              f"Incerteza: {inst['uncertainty_level'].upper()}")
        print(f"  σ_T={inst['thermal_profile']['sigma_T_C']}°C  |  "
              f"{inst['fleet']['num_vehicles']} veículos × "
              f"{inst['fleet']['capacity_kg']:.0f} kg")
        print(f"{'='*65}")

    t_start = time.time()
    pareto, hv_history = run_nsga2(
        inst, pop_size=pop_size, n_gen=n_gen,
        mc_samples=mc_samples, seed=seed, verbose=verbose,
        use_multizone=use_multizone, n_zones=n_zones,
    )
    elapsed = time.time() - t_start

    hv_final = compute_hypervolume(pareto)
    spread   = compute_spread(pareto)
    gd       = compute_gd(pareto, pareto)    # auto-referência (sem Pareto verdadeiro)

    metrics = {
        "instance":          inst_name,
        "n_customers":       inst['size'],
        "family":            inst['family'],
        "uncertainty":       inst['uncertainty_level'],
        "sigma_T_C":         inst['thermal_profile']['sigma_T_C'],
        "algorithm":         "NSGA-II + Monte Carlo (Simheurística)",
        "pop_size":          pop_size,
        "n_generations":     n_gen,
        "mc_samples":        mc_samples,
        "seed":              seed,
        "use_multizone":     use_multizone,          # B2 — traceabilidade (LEI I.2)
        "n_zones":           n_zones if use_multizone else None,
        "elapsed_seconds":   round(elapsed, 2),
        "pareto_front_size": len(pareto),
        "n_feasible":        sum(1 for ind in pareto if ind.feasible),
        "hypervolume":       round(hv_final, 8),
        "spread_delta":      round(spread, 6),
        "gd_self":           round(gd, 6),
        "best_cost_BRL":     round(min(ind.f_cost  for ind in pareto), 2),
        "best_quality_pct":  round((1 - min(ind.f_qloss for ind in pareto)) * 100, 3),
        "best_co2_kg":       round(min(ind.f_co2   for ind in pareto), 4),
    }

    save_results(pareto, hv_history, metrics, inst_name, output_dir)

    if verbose:
        print(f"\n  ✓ Concluído em {elapsed:.1f}s")
        print(f"  Frente Pareto : {len(pareto)} soluções "
              f"({metrics['n_feasible']} viáveis)")
        print(f"  Hipervolume   : {hv_final:.4e}")
        print(f"  Spread (Δ)    : {spread:.4f}")
        print(f"  Melhor custo  : R${metrics['best_cost_BRL']:,.2f}")
        print(f"  Melhor qualid.: {metrics['best_quality_pct']:.2f}%")
        print(f"  Melhor CO₂    : {metrics['best_co2_kg']:.2f} kg CO₂-eq")

    return pareto, metrics


# ─────────────────────────────────────────────────────────────────────────────
# 11b. BUSCA ALEATÓRIA MULTIOBJETIVO (BASELINE INFERIOR)
# ─────────────────────────────────────────────────────────────────────────────

def _update_pareto_archive(archive: List[Individual], candidate: Individual) -> List[Individual]:
    """Mantém arquivo de soluções não-dominadas."""
    new_archive = [ind for ind in archive if not candidate.dominates(ind)]
    if all(not ind.dominates(candidate) for ind in archive):
        new_archive.append(candidate)
    return new_archive


def run_random_search(
    inst:       dict,
    pop_size:   int  = POP_SIZE,
    n_gen:      int  = N_GEN,
    mc_samples: int  = MC_SAMPLES,
    seed:       int  = 0,
    verbose:    bool = True,
    use_multizone: bool = False,
    n_zones:       int  = 1,
    hv_ref_point: Optional[Tuple] = None,
) -> Tuple[List[Individual], List[Tuple[int, float]]]:
    """
    Busca aleatória multiobjetivo com Monte Carlo (MOEA-RS).
    Gera pop_size permutações aleatórias por iteração e mantém arquivo Pareto.
    Serve como baseline inferior: sem inteligência evolucionária.

    hv_ref_point (novo, 06/09/2026 -- diário Entrada 52): mesmo significado
    de run_nsga2() -- ponto de referência fixo opcional pro HV de checkpoint;
    default None preserva o comportamento auto-referenciado de sempre.
    """
    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)
    dist, ttime = build_matrices(inst)
    n = len(inst['customers'])
    cat_index = _cat_index_map(inst) if use_multizone else None
    n_cat = len(cat_index) if cat_index else 0

    archive: List[Individual] = []
    hv_history: List[Tuple[int, float]] = []
    t0 = time.time()

    for gen in range(1, n_gen + 1):
        for _ in range(pop_size):
            chrom = list(range(1, n + 1))
            rng_py.shuffle(chrom)
            ind = Individual(chromosome=chrom)
            if use_multizone:
                ind.zone_block = _init_zone_block(n, n_cat, n_zones, rng_py)
            evaluate(ind, inst, dist, ttime, rng_np, mc_samples,
                     use_multizone=use_multizone, n_zones=n_zones, cat_index=cat_index)
            archive = _update_pareto_archive(archive, ind)

        if gen % 10 == 0 or gen == n_gen:
            hv = compute_hypervolume(archive, ref_point=hv_ref_point)
            hv_history.append((gen, hv))
            if verbose:
                print(f"  Iter {gen:4d}/{n_gen} | Arquivo: {len(archive):3d} sol | HV: {hv:.4e} | {time.time()-t0:.1f}s", flush=True)

    return archive, hv_history


# ─────────────────────────────────────────────────────────────────────────────
# 12. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
    INST_DIR    = os.path.normpath(os.path.join(BASE_DIR, '..', 'instances', 'generated'))
    RESULTS_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'results'))

    # Argumento opcional: caminho da instância
    if len(sys.argv) > 1:
        inst_path = sys.argv[1]
    else:
        # Por padrão: menor instância (n=25, família C1, incerteza baixa)
        inst_path = os.path.join(INST_DIR, 'CCVRPTW-ST_C1_025_LOW.json')

    if not os.path.exists(inst_path):
        print(f"Arquivo não encontrado: {inst_path}")
        sys.exit(1)

    run_instance(inst_path, RESULTS_DIR, seed=42, verbose=True)
