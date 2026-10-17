#!/usr/bin/env python3
"""
Modelo térmico dependente da rota — CCVRPTW-ST (01/10/2026, diário Entrada 70)
=============================================================================
Substitui, como opção, o ruído iid por trecho do modelo original, que mostrou
tornar f2 esperado ≈ constante × f2 nominal (Entrada 69). Aqui a exposição
térmica depende da rota: posição de cada cliente, carga a bordo, hora do dia.

Mecanismo (Stellingwerf et al., 2021, IJPE 231:107849, eqs. 2-9; cinética de
primeira ordem, como no resto do CCVRPTW-ST):
  • Em cada parada j a porta do compartimento do cliente abre durante
    d_j = τ·q_j (s). O ar aquece em direção à temperatura externa Ta da hora
    do atendimento (eq. 4) e o produto aquece a partir do ar (eq. 6):
        T_ar   = T0 + min(1, 0.5·d·φ)·(Ta − T0)
        T_prod = T0 + min(1, 0.5·d·φ)·(T_ar − T0)
  • Durante o tempo de serviço, os produtos ainda a bordo daquele
    compartimento degradam a T_prod (eq. 2). A refrigeração fica inativa
    durante a descarga (Novaes et al., 2015).
  • No trecho seguinte, o equipamento leva c·L segundos para baixar a
    temperatura (L = carga restante no compartimento), período em que a
    temperatura é 0,5·(T_ar + T0) (eq. 8); o resto do trecho, e a espera
    antes da janela, ficam no set-point mais a oscilação do termostato
    (σ_T, o ruído do modelo original).
  • Em B2 só abre o compartimento da zona do cliente atendido.

Incerteza (sorteada por cenário):
  • Temperatura externa: um dia REAL sorteado do mês mais quente de
    2011-2020 da cidade escolhida (NOAA ISD, ver pesquisa/clima/). Nominal =
    perfil horário médio do mês.
  • Duração da porta aberta: multiplicador lognormal de média 1 e coeficiente
    de variação por nível de incerteza (faixa 0,05-0,60 de Novaes et al., 2015).
  • Oscilação do termostato: N(0, σ_T) por trecho, como antes.

Horário: o relógio da instância (minutos desde a abertura do depósito)
começa em T_START_H. Partidas do depósito em t=0 e serviço nominal, igual a
nsga2_simheuristic.decode(), então o cronograma não depende do cenário.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np

import tempo_dependente as td

R_GAS = 8.314

# Parâmetros físicos (fonte em cada linha)
TAU_UNLOAD_S_PER_KG = 0.8      # Stellingwerf et al. (2021), caso-base; faixa testada 0,4-1,2
PHI_HEAT_PER_S      = 0.0027   # Tso et al. (2002), via Stellingwerf et al. (2021) eq. 4
COOL_S_PER_KG       = 0.4      # Stellingwerf et al. (2021), caso-base
DOOR_CV = {'LOW': 0.10, 'MEDIUM': 0.30, 'HIGH': 0.60}   # Novaes et al. (2015): CV 0,05-0,60

CLIMA_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '..', 'pesquisa', 'clima'))
CLIMA_PADRAO = '83780099999'   # São Paulo (Congonhas) — provisório, ver Entrada 70
T_START_PADRAO = 6.0

_clima_cache: Dict[str, dict] = {}


def carregar_clima(estacao: str = CLIMA_PADRAO) -> dict:
    if estacao not in _clima_cache:
        with open(os.path.join(CLIMA_DIR, f'perfis_{estacao}.json'), encoding='utf-8') as fh:
            d = json.load(fh)
        d['perfis'] = np.array(d['perfis_diarios_C'], dtype=float)
        d['nominal'] = np.array(d['perfil_nominal_C'], dtype=float)
        _clima_cache[estacao] = d
    return _clima_cache[estacao]


def _nivel(inst: dict) -> str:
    return str(inst.get('uncertainty_level', 'MEDIUM')).upper()


def sortear_cenarios(inst: dict, K: int, rng: np.random.Generator, n_zones: int = 1,
                     estacao: str = CLIMA_PADRAO, tempo_viagem: bool = False) -> dict:
    """K cenários: dia de temperatura externa, multiplicadores de porta, normais do termostato
    e, se tempo_viagem, multiplicadores lognormais (média 1) do tempo de cada trecho, indexados
    pelo cliente de DESTINO do trecho (CV = thermal_profile.sigma_t_pct da instância).
    Uma instância pode trazer a própria estação de clima ('estacao_clima'; caso de Atenas, Entrada 103)."""
    cl = carregar_clima(inst.get('estacao_clima', estacao))
    n = len(inst['customers'])
    cv = DOOR_CV.get(_nivel(inst), 0.30)
    s2 = np.log(1.0 + cv * cv)
    return {
        'amb': cl['perfis'][rng.integers(0, len(cl['perfis']), size=K)],          # K×24
        'door': rng.lognormal(-s2 / 2, np.sqrt(s2), size=(K, n + 1)),           # K×(n+1)
        'eps': rng.standard_normal(size=(K, n + 1, max(1, n_zones))),           # K×(n+1)×Z
        **({'tt': _lognormal_media1(rng, inst['thermal_profile'].get('sigma_t_pct', 0.0), (K, n + 1))}
           if tempo_viagem else {}),
    }


def _lognormal_media1(rng, cv, shape):
    s2 = np.log(1.0 + cv * cv)
    return rng.lognormal(-s2 / 2, np.sqrt(s2), size=shape)


def cenario_nominal(inst: dict, n_zones: int = 1, estacao: str = CLIMA_PADRAO) -> dict:
    cl = carregar_clima(inst.get('estacao_clima', estacao))   # estação própria da instância (caso de Atenas)
    n = len(inst['customers'])
    return {'amb': cl['nominal'][None, :], 'door': np.ones((1, n + 1)),
            'eps': np.zeros((1, n + 1, max(1, n_zones)))}


def _amb_at(amb: np.ndarray, hora: np.ndarray) -> np.ndarray:
    """Interpola o perfil horário (K×24, circular) nas horas dadas (K×m) → K×m."""
    h0 = np.floor(hora).astype(int) % 24
    h1 = (h0 + 1) % 24
    w = hora - np.floor(hora)
    return (np.take_along_axis(amb, h0, axis=1) * (1 - w)
            + np.take_along_axis(amb, h1, axis=1) * w)


def _k(T_C, Ea, A0):
    return A0 * np.exp(-Ea / (R_GAS * (T_C + 273.15)))


def route_quality(route: List[int], inst: dict, ttime, scen: dict,
                  zone_of_customer: Optional[List[int]] = None,
                  zone_temp: Optional[Dict[int, dict]] = None,
                  per_customer_kinetics: bool = False,
                  t_start_h: float = T_START_PADRAO,
                  tau_p_h: Optional[float] = None, detalhe: bool = False,
                  t0_min: float = 0.0):
    """
    Qualidade residual ponderada por valor da rota, por cenário: shape (K,).

    Baseline (zone_of_customer=None): uma zona, T0/σ/limites e cinética da
    categoria dominante (thermal_profile), como no modelo original.
    B2: zone_of_customer[i] = zona do i-ésimo cliente da rota; zone_temp como
    em nsga2_simheuristic.assign_zones(); cinética da categoria de cada cliente.
    """
    if not route:
        return np.ones(scen['amb'].shape[0])
    cust = inst['customers']
    tp = inst['thermal_profile']
    m = len(route)
    K = scen['amb'].shape[0]

    # Cronograma por cenário (sem 'tt' no cenário: o nominal, idêntico ao decoder).
    # Chegada cedo espera a janela; chegada tarde é atendida mesmo assim
    # (janela flexível) e o atraso fica registrado em `late` (minutos).
    st = np.array([cust[c - 1]['service_time_min'] for c in route]) / 60.0           # h
    if isinstance(ttime, td.MatrizTD):
        # tempos dependentes do horário (Entrada 101): o tempo de cada trecho depende da saída; o relógio
        # da instância é o da matriz (hora de início da operação; 6 h no caso SP, 8 h no de Atenas)
        tl, wait, t_srv, late = td.cronograma(route, inst, ttime, K, scen.get('tt'), t0_min)
        t_start_h = ttime.hora_inicio
    else:
        nodes = [0] + route
        tl0 = np.array([ttime[nodes[j]][nodes[j + 1]] for j in range(m)])               # min
        tl = tl0[None, :] * scen['tt'][:, route] if 'tt' in scen else np.repeat(tl0[None, :], K, 0)
        ready = np.array([cust[c - 1]['ready_time_min'] for c in route])
        due = np.array([cust[c - 1]['due_date_min'] for c in route])
        wait = np.zeros((K, m)); t_srv = np.zeros((K, m)); late = np.zeros((K, m))
        t = np.full(K, float(t0_min))   # saída do depósito (min); 0 = comportamento anterior
        for j in range(m):
            t = t + tl[:, j]
            late[:, j] = np.maximum(0.0, t - due[j])
            w = np.maximum(0.0, ready[j] - t)
            t = t + w
            wait[:, j], t_srv[:, j] = w / 60.0, t / 60.0
            t = t + st[j] * 60.0
    tl = tl / 60.0                                                                   # h
    hora = (t_srv + t_start_h) % 24.0
    Ta = _amb_at(scen['amb'], hora)                                                  # K×m
    dem = np.array([cust[c - 1]['demand_kg'] for c in route])
    d = TAU_UNLOAD_S_PER_KG * dem[None, :] * scen['door'][:, route]                  # K×m (s)
    f = np.minimum(1.0, 0.5 * d * PHI_HEAT_PER_S)                                    # K×m
    values = np.array([cust[c - 1].get('load_value_BRL', 1.0) for c in route])
    tot = values.sum() or 1.0

    if zone_of_customer is None:
        zc = np.zeros(m, dtype=int)
        zt = {0: {'T_set_C': tp['T_set_C'], 'T_min_C': tp.get('T_min_C', tp['T_set_C'] - 3.0),
                  'T_max_C': tp.get('T_max_C', tp['T_set_C'] + 5.0), 'sigma_T_C': tp['sigma_T_C']}}
    else:
        zc = np.asarray(zone_of_customer, dtype=int)
        zt = zone_temp
    if per_customer_kinetics:
        pc = inst['thermal_profile']['per_category']
        kin = [(pc[cust[c - 1]['product_category']]['Ea_J_mol'], pc[cust[c - 1]['product_category']]['A0'])
               for c in route]
    else:
        kin = [(tp['Ea_J_mol'], tp['A0'])] * m

    Q = np.zeros((K, m))
    Tp_max = np.full((K, m), np.nan)   # máx. da temperatura do produto até a entrega (se tau_p_h)
    for z, zinf in zt.items():
        idx_z = np.where(zc == z)[0]
        if len(idx_z) == 0:
            continue
        T0 = zinf['T_set_C']
        aberta = (zc == z)                                                           # porta deste compartimento abre na parada j?
        T_ar = np.where(aberta[None, :], T0 + f * (Ta - T0), T0)                     # K×m
        T_prod = np.where(aberta[None, :], T0 + f * (T_ar - T0), T0)
        # carga deste compartimento ainda a bordo depois do serviço em j
        dem_z = np.where(aberta, dem, 0.0)
        L_after = dem_z[::-1].cumsum()[::-1] - dem_z
        pull = np.zeros((K, m))
        pull[:, 1:] = np.where(aberta[None, :-1],
                               np.minimum(tl[:, 1:], COOL_S_PER_KG * L_after[None, :-1] / 3600.0), 0.0)
        T_pull = np.zeros((K, m))
        T_pull[:, 1:] = 0.5 * (T_ar[:, :-1] + T0)
        eps = scen['eps'][:, route, min(z, scen['eps'].shape[2] - 1)]
        T_st = np.clip(T0 + zinf['sigma_T_C'] * eps, zinf['T_min_C'], zinf['T_max_C'])  # termostato
        # serviço numa parada de OUTRO compartimento: este fica fechado, no set-point
        T_srv = np.where(aberta[None, :], T_prod, T_st)
        for (Ea, A0) in set(kin[i] for i in idx_z):
            pre = _k(T_pull, Ea, A0) * pull + _k(T_st, Ea, A0) * (tl - pull + wait)
            srv = _k(T_srv, Ea, A0) * st[None, :]
            cum = np.cumsum(pre, axis=1) + np.concatenate([np.zeros((K, 1)), np.cumsum(srv, axis=1)[:, :-1]], axis=1)
            Qz = np.exp(-cum)
            for i in idx_z:
                if kin[i] == (Ea, A0):
                    Q[:, i] = Qz[:, i]
        if tau_p_h:
            # Temperatura do produto: modelo concentrado de 1ª ordem,
            # dTp/dt = (T_ar − Tp)/τp, solução exata por trecho de T_ar constante.
            # Parte de T0 (produto pré-resfriado). Com a porta aberta o ar está
            # em T_ar durante todo o serviço (refrigeração inativa na descarga).
            T_air_srv = np.where(aberta[None, :], T_ar, T_st)
            Tp = np.full(K, float(T0)); run = Tp.copy()
            for j in range(m):
                Tp = T_pull[:, j] + (Tp - T_pull[:, j]) * np.exp(-pull[:, j] / tau_p_h)
                run = np.maximum(run, Tp)
                dur = tl[:, j] - pull[:, j] + wait[:, j]
                Tp = T_st[:, j] + (Tp - T_st[:, j]) * np.exp(-dur / tau_p_h)
                run = np.maximum(run, Tp)
                if zc[j] == z:
                    Tp_max[:, j] = run
                Tp = T_air_srv[:, j] + (Tp - T_air_srv[:, j]) * np.exp(-st[j] / tau_p_h)
                run = np.maximum(run, Tp)
    if detalhe:
        return {'Q': Q, 'Tp_max': Tp_max, 'values': values, 'late': late,
                'cat': [cust[c - 1]['product_category'] for c in route]}
    return (Q * values[None, :]).sum(axis=1) / tot
