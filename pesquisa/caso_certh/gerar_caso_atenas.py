#!/usr/bin/env python3
"""
Caso de Atenas com dados operacionais de um operador logístico (16/10/2026, diário Entrada 103; melhoria 1).

Fonte: Vrani, Apostolidis, Kapoutsis & Kosmatopoulos (2025), "Delivering data: A real-world dataset for
last-mile delivery optimization", Data in Brief 61, 111762 (doi:10.1016/j.dib.2025.111762); dados em
Zenodo, doi:10.5281/zenodo.15672291 (v3, CC BY 4.0), baixados em 16/10/2026 para
dados_brutos/zenodo_15672291 (md5 de data.zip conferido com o do Zenodo).
Nove dias de entregas farmacêuticas de um operador logístico (3PL) em Atenas: pedidos com peso, tempo de
serviço e janela (minutos desde as 08:00; definidos pelo tipo de destinatário) e matrizes de distância e de
tempo (Google Distance Matrix) em três modelos de tráfego (otimista, mais provável, pessimista); as
localizações foram omitidas pelos autores.

O que vem dos dados (operacional): clientes, demandas, janelas, tempos de serviço, distâncias, tempo médio
e variabilidade do tempo de cada arco. Seguindo os autores (distribuição Beta), cada arco tem média e
desvio-padrão PERT: com a ≤ m ≤ b os três tempos ordenados, μ = (a + 4m + b)/6 e σ = (b − a)/6; o tempo
nominal é μ e o CV do arco é σ/μ (incerteza própria de cada arco; src/tempo_dependente.py).
O que é suposto (documentado): frota (capacidade 800 kg, como as vans leves do caso SP; tamanho pela regra
EDD × 1,10 do caso SP), custos (os do caso SP), carga refrigerada na faixa de 2 a 8 °C representada pelo perfil
térmico 'dairy' (set-point 4 °C, máximo 8 °C) — a base não informa o produto —, clima de Atenas no mês mais
quente (NOAA ISD), incerteza de porta e termostato no nível médio.
Uso: python -I pesquisa/caso_certh/gerar_caso_atenas.py   →  instances/caso_atenas/CCVRPTW-ST_ATH{d}_{n}_DATA.json
"""
import json, math, os, sys
from collections import Counter

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(AQUI, '..', '..'))
sys.path.insert(0, os.path.join(BASE, 'src')); sys.path.insert(0, os.path.join(BASE, 'pesquisa', 'caso_sp'))
import instance_generator as ig   # noqa: E402
import gerar_caso_sp as g         # noqa: E402   (frota_edd e custos do caso SP)
import pandas as pd               # noqa: E402

DADOS = os.path.join(AQUI, 'dados_brutos', 'zenodo_15672291', 'extraido')
SAIDA = os.path.join(BASE, 'instances', 'caso_atenas')
CAPACIDADE = 800.0
CATEGORIA = 'dairy'
ESTACAO = '16716199999'          # Atenas, Eleftherios Venizelos (LGAV), NOAA ISD
HORA_INICIO = 8.0
HORIZONTE = 600.0                # 08:00 às 18:00 (o maior prazo dos dados é 360 min)


def pert(o, m, p):
    a, mo, b = sorted((o, m, p))
    mu = (a + 4 * mo + b) / 6.0; sd = (b - a) / 6.0
    return mu, (sd / mu if mu > 0 else 0.0)


def main():
    if os.path.isdir(SAIDA) and os.listdir(SAIDA):
        sys.exit(f'{SAIDA} não está vazia: arquive antes de gerar (nada foi gravado).')
    os.makedirs(SAIDA, exist_ok=True)
    pedidos = pd.ExcelFile(os.path.join(DADOS, 'orders', 'orders.xlsx'))
    tp = ig.THERMAL_PROFILES[CATEGORIA]; unc = ig.UNCERTAINTY_LEVELS['medium']
    catalogo = []
    for d in range(1, 10):
        P = os.path.join(DADOS, 'time_and_distance_matrices', f'day_{d}')
        ler = lambda k: pd.read_excel(os.path.join(P, f'{k}_{d}.xlsx'), index_col=0).values.astype(float).tolist()
        o, m, p = ler('time_matrix_optimistic'), ler('time_matrix_mostlikely'), ler('time_matrix_pessimistic')
        dist = ler('distance_matrix')
        df = pedidos.parse(f'Day {d}').sort_values('NODE_ID')
        n = len(df); N = n + 1
        assert len(o) == N and list(df.NODE_ID) == list(range(1, N)), f'dia {d}: nós fora do padrão'
        mu = [[0.0] * N for _ in range(N)]; cv = [[0.0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i != j:
                    mu[i][j], cv[i][j] = pert(o[i][j], m[i][j], p[i][j])
        tw = [(float(r.EAT), float(r.LAT), float(r.SERVICE_TIME)) for r in df.itertuples()]
        dem = [float(r.WEIGHT) for r in df.itertuples()]
        n_veic = max(math.ceil(sum(dem) / CAPACIDADE), math.ceil(g.frota_edd(mu, tw, dem, CAPACIDADE) * 1.10))
        pares = [(mu[i][j], cv[i][j]) for i in range(N) for j in range(N) if i != j]
        cv_ref = sum(a * b for a, b in pares) / sum(a for a, _ in pares)     # média ponderada pelo tempo
        clientes = [{'id': i, 'x': 0.0, 'y': 0.0, 'demand_kg': dem[i - 1], 'volume_m3': float(df.VOLUME.iloc[i - 1]),
                     'ready_time_min': tw[i - 1][0], 'due_date_min': tw[i - 1][1], 'service_time_min': tw[i - 1][2],
                     'product_category': CATEGORIA, 'product_description': 'carga refrigerada (2-8 °C), perfil dairy',
                     'unit_value_BRL_kg': tp['unit_value_BRL_kg'],
                     'load_value_BRL': round(dem[i - 1] * tp['unit_value_BRL_kg'], 2)} for i in range(1, N)]
        nome = f'CCVRPTW-ST_ATH{d}_{n:03d}_DATA'
        inst = {
            'name': nome, 'family': f'ATH{d}', 'size': n, 'uncertainty_level': 'data',
            'depot': {'id': 0, 'x': 0.0, 'y': 0.0, 'ready_time_min': 0.0, 'due_date_min': HORIZONTE},
            'customers': clientes,
            'fleet': {'num_vehicles': n_veic, 'capacity_kg': CAPACIDADE, 'speed_km_h': None,
                      'fixed_cost_BRL': 150.0, 'variable_cost_BRL_km': 2.50},
            'thermal_profile': {
                'dominant_category': CATEGORIA, 'description': tp['description'], 'T_set_C': tp['T_set_C'],
                'T_max_C': tp['T_max_C'], 'T_min_C': tp['T_min_C'], 'Ea_J_mol': tp['Ea_J_mol'], 'A0': tp['A0'],
                'Q_min': tp['Q_min'], 'sigma_T_C': unc['sigma_T'], 'sigma_t_pct': round(cv_ref, 6),
                'uncertainty_label': 'tempo de viagem: CV de cada arco pelos dados; porta e termostato: nível médio',
                'validation': ig.validate_thermal_params({'category': CATEGORIA}), 'references': tp['references'],
                'per_category': {CATEGORIA: {'T_set_C': tp['T_set_C'], 'T_max_C': tp['T_max_C'], 'Ea_J_mol': tp['Ea_J_mol'],
                                             'A0': tp['A0'], 'Q_min': tp['Q_min'], 'sigma_T_C': unc['sigma_T'],
                                             'validation': ig.validate_thermal_params({'category': CATEGORIA})}}},
            'co2_params': ig.CO2_PARAMS, 'horizon_minutes': HORIZONTE, 'estacao_clima': ESTACAO,
            'dist_matrix_km': [[round(x, 4) for x in row] for row in dist],
            'time_matrix_ff_min': [[round(x, 6) for x in row] for row in mu],
            'td': {'modelo': 'estático (perfil plano); tempo de cada arco lognormal com média e CV PERT dos três '
                             'modelos de tráfego do Google', 'hora_inicio': HORA_INICIO, 'fator_hora': [1.0] * 24,
                   'cv_arco': [[round(x, 6) for x in row] for row in cv]},
            'generation_metadata': {
                'caso': 'Atenas, dados operacionais de um operador logístico (CERTH, 2025)', 'dia': d,
                'num_customers': n, 'capacidade_suposta_kg': CAPACIDADE, 'cv_ref_ponderado': cv_ref,
                'prazos': dict(Counter(int(t[1]) for t in tw)), 'servicos': dict(Counter(int(t[2]) for t in tw)),
                'fontes': ['Vrani et al. (2025) Data in Brief 61:111762, doi:10.1016/j.dib.2025.111762',
                           'Zenodo doi:10.5281/zenodo.15672291 (CC BY 4.0)',
                           'NOAA ISD, estação 16716199999 (Atenas), mês mais quente 2011-2020']}}
        json.dump(inst, open(os.path.join(SAIDA, nome + '.json'), 'w', encoding='utf-8'), ensure_ascii=False)
        vol = df.VOLUME.sum()
        catalogo.append(dict(nome=nome, frota=n_veic, demanda_kg=round(sum(dem), 1), volume_m3=round(vol, 2),
                             cv_ref=round(cv_ref, 3), ida_media_min=round(sum(mu[0][1:]) / n, 1),
                             prazos=inst['generation_metadata']['prazos']))
        print(catalogo[-1], flush=True)
    json.dump({'instancias': catalogo}, open(os.path.join(SAIDA, 'catalogo_caso_atenas.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
