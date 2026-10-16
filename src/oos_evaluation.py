#!/usr/bin/env python3
"""
Reavaliação fora da amostra (out-of-sample) de f2 — CCVRPTW-ST
================================================================
29/09/2026 — ver pesquisa/plano_reavaliacao_oos.md.

Motivação: o f2 gravado por cada algoritmo é medido com a própria régua da
busca — NSGA2-DET em T_set fixo (otimista, já que k(T) de Arrhenius é
convexa em T), variantes MC pela média das mesmas 10/30 amostras usadas na
seleção (viés de "winner's curse"). Comparar frentes com essas réguas
diferentes não é uma comparação justa. Aqui toda solução final é reavaliada
num conjunto COMUM e grande de K cenários, independente da busca.

Números aleatórios comuns (CRN): um tensor fixo de normais-padrão
Z[s, c, z] por instância (semente derivada do nome da instância). O trecho
que CHEGA ao cliente c, na zona z, no cenário s, usa Z[s, c, z]. Como cada
cliente é destino de exatamente um trecho em qualquer solução, cada trecho
recebe uma normal independente — a mesma distribuição do modelo original
(_route_quality_mc/_route_quality_mc_multizone) — e soluções diferentes são
medidas nos mesmos cenários. Mesmo modelo físico da busca: trechos
depósito→c1→...→ck (sem o retorno), T = clip(T_set + σ·Z, T_min, T_max).
"""

import zlib
from typing import Dict, List, Optional

import numpy as np

import nsga2_simheuristic as ns

DEFAULT_K = 1000


def scenario_tensor(inst: dict, K: int = DEFAULT_K, n_zones: int = 1,
                    base_seed: int = 20260929) -> np.ndarray:
    """Z[s, c, z] ~ N(0,1), c = 0..n (0 = depósito, nunca usado como destino)."""
    name = inst.get('name') or inst.get('instance_name') or ''
    seed = (base_seed + zlib.crc32(name.encode('utf-8'))) % (2**32)
    rng = np.random.default_rng(seed)
    n = len(inst['customers'])
    return rng.standard_normal(size=(K, n + 1, max(1, n_zones)))


def _leg_times_h(route: List[int], ttime) -> np.ndarray:
    nodes = [0] + route
    return np.array([ttime[nodes[i]][nodes[i + 1]] / 60.0 for i in range(len(route))])


def _route_wq_single(route, inst, ttime, Z, nominal: bool) -> np.ndarray:
    """Qualidade ponderada por valor da rota, por cenário: shape (K,) (ou (1,) se nominal)."""
    tp = inst['thermal_profile']
    T_set = tp['T_set_C']
    T_min = tp.get('T_min_C', T_set - 3.0)
    T_max = tp.get('T_max_C', T_set + 5.0)
    Ea, A0 = tp['Ea_J_mol'], tp['A0']
    cust = inst['customers']
    leg_t = _leg_times_h(route, ttime)
    values = np.array([cust[c - 1].get('load_value_BRL', 1.0) for c in route])
    tot = values.sum() or 1.0
    if nominal:
        T = np.full((1, len(route)), T_set)
    else:
        T = np.clip(T_set + tp['sigma_T_C'] * Z[:, route, 0], T_min, T_max)
    k = A0 * np.exp(-Ea / (ns.R_GAS * (T + 273.15)))
    Q = np.exp(-np.cumsum(k * leg_t, axis=1))
    return (Q * values).sum(axis=1) / tot


def _route_wq_multizone(route, inst, ttime, Z, zinfo, nominal: bool) -> np.ndarray:
    per_cat = inst['thermal_profile']['per_category']
    cust = inst['customers']
    leg_t = _leg_times_h(route, ttime)
    values = np.array([cust[c - 1].get('load_value_BRL', 1.0) for c in route])
    tot = values.sum() or 1.0
    cats = [cust[c - 1]['product_category'] for c in route]
    zone_of = zinfo['zone_of_category']
    K = 1 if nominal else Z.shape[0]
    Q = np.zeros((K, len(route)))
    for z, zt in zinfo['zone_temp'].items():
        idxs = [i for i, c in enumerate(cats) if zone_of[c] == z]
        if not idxs:
            continue
        if nominal:
            T = np.full((1, len(route)), zt['T_set_C'])
        else:
            T = np.clip(zt['T_set_C'] + zt['sigma_T_C'] * Z[:, route, z],
                        zt['T_min_C'], zt['T_max_C'])
        TK = T + 273.15
        for c in {cats[i] for i in idxs}:
            k = per_cat[c]['A0'] * np.exp(-per_cat[c]['Ea_J_mol'] / (ns.R_GAS * TK))
            Qc = np.exp(-np.cumsum(k * leg_t, axis=1))
            for i in idxs:
                if cats[i] == c:
                    Q[:, i] = Qc[:, i]
    return (Q * values).sum(axis=1) / tot


def qloss_scenarios(routes: List[List[int]], inst: dict, ttime, Z: np.ndarray,
                    zone_block: Optional[List[int]] = None, use_multizone: bool = False,
                    cat_index: Optional[Dict[str, int]] = None,
                    nominal: bool = False) -> np.ndarray:
    """f2 (perda de qualidade ponderada por valor) da solução em cada cenário: shape (K,)."""
    cust = inst['customers']
    zinfo_all = (ns.assign_zones(routes, zone_block or [], inst, cat_index)
                 if use_multizone else None)
    K = 1 if nominal else Z.shape[0]
    acc = np.zeros(K)
    tot = 0.0
    for r_idx, route in enumerate(routes):
        if not route:
            continue
        if use_multizone:
            wq = _route_wq_multizone(route, inst, ttime, Z, zinfo_all[r_idx], nominal)
        else:
            wq = _route_wq_single(route, inst, ttime, Z, nominal)
        r_val = sum(cust[c - 1].get('load_value_BRL', 1.0) for c in route)
        acc += wq * r_val
        tot += r_val
    return 1.0 - (acc / tot if tot > 0 else np.ones(K))


def summarize(q: np.ndarray, alpha: float = 0.95) -> dict:
    """Média, erro-padrão e CVaR_alpha (média dos piores (1-alpha) cenários) de f2."""
    qs = np.sort(q)
    tail = qs[int(np.floor(alpha * len(qs))):] if len(qs) > 1 else qs
    return {
        'f2_oos_mean': float(q.mean()),
        'f2_oos_se':   float(q.std(ddof=1) / np.sqrt(len(q))) if len(q) > 1 else 0.0,
        'f2_oos_cvar95': float(tail.mean()),
    }
