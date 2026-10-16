#!/usr/bin/env python3
"""
Caso realista em São Paulo, versão 2 com dados públicos (14/10/2026, diário Entrada 92; pedido do
pesquisador em 13/10/2026: "vamos partir dessas melhorias ... sem a consulta ao Bonfim sobre
parceria"). A versão 1 (clientes do OpenStreetMap, demais parâmetros do gerador sintético) está
arquivada em gerar_caso_sp_v1_osm.py e instances/caso_sp_v1_osm/.

O que vem de dado público e o que continua gerado (desenho fixado ANTES de gerar ou rodar):
  - Depósito: CEAGESP, Portão 4 (nó OSM 10689437067). [real]
  - Clientes: supermercados e hipermercados ATIVOS no cadastro CNPJ da Receita Federal
    (competência 2026-10; CNAE 4711-3/01 e 4711-3/02; município de São Paulo), geocodificados
    pelo endereço do cadastro no nível do número (geocodificar_cnpj.py), a até 15 km do depósito,
    com o CEP do OSM conferindo com o do cadastro (5 dígitos). Amostra estratificada pelo porte, com
    as proporções do cadastro (a geocodificação favorece empresas maiores). Duas amostras
    independentes (SP1 e SP2) para cada tamanho n = 50 e n = 100. [real]
  - Distâncias e tempos: OSRM sobre a malha do OpenStreetMap, tempo de fluxo livre × 1,585
    (congestionamento médio de São Paulo em 2025, TomTom Traffic Index). [real, média do dia]
  - Janelas de recebimento: Oliveira et al. (2016, Transportes 24(2):53-67) e a dissertação que
    os originou (Ramos, 2015, UFMG, Tabela 5.11), 39 supermercados de Belo Horizonte: 66,7% têm
    programação de recebimento; nesses, a entrega cai numa faixa sorteada com as frequências
    observadas (06-08 h 1,9%; 08-11 h 44,2%; 11-14 h 17,3%; 14-16 h 30,8%; 16-19 h 5,8%); nos
    demais, recebe-se das 08 h às 19 h. [frequências reais; sorteio por loja]
  - Tempo de serviço: distribuição do tempo de descarga por loja da mesma pesquisa (Oliveira et al.,
    2016, Tabela 8; 10 a 120 min, média 45,9 min). [frequências reais; sorteio por loja]
  - Mix de produtos refrigerados: aquisição domiciliar per capita no estado de São Paulo, POF
    2017-2018 (IBGE, SIDRA tabela 2393), agrupada nas quatro categorias térmicas (mapa abaixo);
    leite fluido entra só com a parcela pasteurizada refrigerada (o UHT era 78,2% do leite fluido
    em 2011; Pithan e Silva, 2016, IEA). [real, como proporção]
  - Demanda: faixa da categoria (a mesma do gerador sintético, limitada pela capacidade do
    veículo), com a posição na faixa dada pelo PORTE da empresa no cadastro CNPJ: microempresa no
    terço inferior, pequeno porte no do meio, demais no superior. [porte real; volume estimado]
  - Frota: furgões refrigerados leves de 600 a 800 kg de carga útil, como no benchmark. Pelo
    Código de Trânsito Brasileiro (Anexo I), veículo de carga com PBT até 3.500 kg é caminhonete,
    não caminhão; as restrições a caminhões da ZMRC (Decreto 56.920/2016) não se aplicam.
  - Incerteza MEDIUM e HIGH; clima de São Paulo (caso-base); operação a partir das 06:00.
  → 8 instâncias: CCVRPTW-ST_SP{1,2}_{050,100}_{MEDIUM,HIGH}.
Atribuições: dados © colaboradores do OpenStreetMap (ODbL); OSRM; Nominatim; Receita Federal
(dados abertos do CNPJ); IBGE (POF 2017-2018); TomTom Traffic Index.
Dados baixados ficam em dados_brutos/ e são lidos como não confiáveis: rode com python -I.
Uso: python -I pesquisa/caso_sp/gerar_caso_sp.py
"""
import csv, json, math, os, random, sys, time, urllib.request
from collections import Counter

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(AQUI, '..', '..'))
sys.path.insert(0, os.path.join(BASE, 'src'))
import instance_generator as ig   # noqa: E402

DEPOSITO = (-23.5346891, -46.7345446)          # CEAGESP, Portão 4 (OSM node 10689437067)
RAIO_KM = 15.0
CONGESTIONAMENTO = 0.585                         # TomTom Traffic Index, São Paulo, 2025
HORIZONTE = 960.0                                # 06:00 às 22:00
OSRM = 'https://router.project-osrm.org/table/v1/driving/'
SEMENTES = {(1, 50): 2611, (1, 100): 2612, (2, 50): 2621, (2, 100): 2622,
            # amostras 3 e 4 acrescentadas em 15/10/2026 (análise de fraquezas A1: poder estatístico)
            (3, 50): 2631, (3, 100): 2632, (4, 50): 2641, (4, 100): 2642}
SAIDA = os.path.join(BASE, 'instances', 'caso_sp')
PREFIXO = 'SP'
CONGESTIONAMENTO_PICO = 0.882                    # TomTom 2025, São Paulo, pico da manhã (análise A3)
BRUTOS = os.path.join(AQUI, 'dados_brutos')
LOJAS = os.path.join(BRUTOS, 'cnpj', 'supermercados_sp_geocodificados.csv')
CADASTRO = os.path.join(BRUTOS, 'cnpj', 'supermercados_sp_2026-10.csv')   # todas as lojas ativas (proporções de porte)
POF = os.path.join(BRUTOS, 'ibge', 'sidra2393_sp_br_2018.json')

# Recebimento (minutos desde 06:00). Oliveira et al. (2016); Ramos (2015), Tabela 5.11.
P_PROGRAMACAO = 0.667
FAIXAS_PROG = [((0, 120), 1.9), ((120, 300), 44.2), ((300, 480), 17.3), ((480, 600), 30.8), ((600, 780), 5.8)]
SEM_PROG = (120, 780)                            # 08:00 às 19:00
# Tempo de descarga por loja (min): nº de supermercados (de 39). Oliveira et al. (2016), Tabela 8;
# Ramos (2015), Figura 5.9.
DESCARGA = {10: 5, 15: 2, 20: 5, 25: 1, 30: 9, 40: 5, 45: 1, 50: 1, 60: 2, 90: 2, 120: 6}
UHT = 0.782                                      # parcela do UHT no leite fluido (Pithan e Silva, 2016)
TERCO = {'01': 0, '03': 1, '05': 2}              # porte → terço da faixa de demanda

# Mapa POF → categoria térmica. Itens de grupos inteiros, com exclusões explícitas (salgados, secos,
# conservas; congelados vão para 'frozen').
SALGADOS = ('7.3.2 ', '7.3.5 ', '7.5.1 ', '7.5.2 ', '7.5.5 ', '9.1.2 ', '9.1.14 ', '9.1.20 ', '9.2.18 ', '9.3.4 ')
CONGELADOS = ('9.1.7 ', '9.1.10 ', '9.1.17 ', '9.2.15 ', '9.3.1 ', '12.2.8 ', '16.1.1 ', '16.1.5 ')


def mix_pof():
    """Proporções das quatro categorias pela aquisição domiciliar per capita (kg/ano), SP, 2017-2018."""
    d = json.load(open(POF, encoding='utf-8'))
    kg = {}
    for r in d[0]['resultados']:
        nome = list(r['classificacoes'][0]['categoria'].values())[0]
        v = {s['localidade']['id']: s['serie'].get('2018') for s in r['series']}['35']
        kg[nome] = float(v) if v not in (None, '-', '...', 'X') else 0.0
    folhas = {k: v for k, v in kg.items() if k.split()[0].count('.') == 2}   # itens (nível 2), ex.: '7.1.1 Alcatra'

    def soma(prefixos, excl=()):
        return sum(v for k, v in folhas.items() if k.startswith(prefixos) and not k.startswith(excl))
    cat = {
        'meat': soma(('7.', '8.', '9.', '10.1.'), SALGADOS + CONGELADOS),
        'dairy': soma(('11.2.', '11.3.')) + (1 - UHT) * soma(('11.1.3 ', '11.1.4 ')),
        'produce': soma(('2.1.', '3.2.')),          # folhosas e florais; frutas de clima temperado
        'frozen': soma(CONGELADOS),
    }
    tot = sum(cat.values())
    return {c: v / tot for c, v in cat.items()}, cat


def hav(a, b):
    R = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    return 2 * R * math.asin(math.sqrt(math.sin((la2 - la1) / 2) ** 2 +
                                       math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2))


def matriz_osrm(coords, rotulo):
    """Matriz completa (duração s, distância m) em blocos de até 34 pontos, 1,5 s entre requisições;
    respostas brutas salvas em dados_brutos/osrm/ (mesma rotina da versão 1)."""
    n = len(coords); dur = [[None] * n for _ in range(n)]; dist = [[None] * n for _ in range(n)]
    blocos = [list(range(i, min(n, i + 34))) for i in range(0, n, 34)]
    os.makedirs(os.path.join(BRUTOS, 'osrm'), exist_ok=True)
    for bi, A in enumerate(blocos):
        for bj, B in enumerate(blocos):
            idx = A + [j for j in B if j not in A]
            url = (OSRM + ';'.join(f'{coords[k][1]:.7f},{coords[k][0]:.7f}' for k in idx)
                   + '?annotations=duration,distance'
                   + '&sources=' + ';'.join(str(idx.index(i)) for i in A)
                   + '&destinations=' + ';'.join(str(idx.index(j)) for j in B))
            arq = os.path.join(BRUTOS, 'osrm', f'{rotulo}_b{bi}{bj}.json')
            if not os.path.exists(arq):
                req = urllib.request.Request(url, headers={'User-Agent': 'CCVRPTW-ST research (academic)'})
                raw = urllib.request.urlopen(req, timeout=120).read()
                open(arq, 'wb').write(raw); time.sleep(1.5)
            r = json.load(open(arq, encoding='utf-8'))
            assert r['code'] == 'Ok', r.get('code')
            for a, i in enumerate(A):
                for b, j in enumerate(B):
                    dur[i][j] = r['durations'][a][b]; dist[i][j] = r['distances'][a][b]
    assert all(v is not None for row in dur for v in row), 'par sem rota'
    return dur, dist


def frota_edd(tempo, tw, dem, cap):
    """Mesma heurística EDD de instance_generator.estimate_routes_by_tw, com a matriz de tempos."""
    n = len(dem); ordem = sorted(range(n), key=lambda i: tw[i][1])
    carga, t, ult = [], [], []
    for i in ordem:
        cid = i + 1; rt, due, st = tw[i]; ok = False
        for k in range(len(carga)):
            if carga[k] + dem[i] > cap:
                continue
            chegada = t[k] + tempo[ult[k]][cid]
            if chegada > due:
                continue
            carga[k] += dem[i]; t[k] = max(chegada, rt) + st; ult[k] = cid; ok = True; break
        if not ok:
            carga.append(dem[i]); t.append(max(tempo[0][cid], rt) + st); ult.append(cid)
    return max(1, len(carga))


def sorteia(rng, pares):
    x = rng.uniform(0, sum(p for _, p in pares)); acc = 0.0
    for v, p in pares:
        acc += p
        if x <= acc:
            return v
    return pares[-1][0]


def main():
    # Opções acrescentadas em 15/10/2026 (análises de fraquezas A1 e A3; o padrão reproduz a geração de 14/10):
    #   --amostras 3 4 --acrescentar   novas amostras de lojas, sem tocar nas instâncias existentes;
    #   --congestionamento 0.882 --saida instances/caso_sp_pico --prefixo SPP   as MESMAS lojas, janelas,
    #     serviços e demandas das amostras indicadas, planejadas com o congestionamento do pico da manhã
    #     (a frota é reestimada pela mesma regra, com os tempos de pico).
    import argparse
    global CONGESTIONAMENTO, SAIDA, PREFIXO
    ap = argparse.ArgumentParser()
    ap.add_argument('--amostras', nargs='+', type=int, default=[1, 2])
    ap.add_argument('--congestionamento', type=float, default=CONGESTIONAMENTO)
    ap.add_argument('--saida', default=SAIDA)
    ap.add_argument('--prefixo', default=PREFIXO)
    ap.add_argument('--acrescentar', action='store_true', help='grava só as instâncias que ainda não existem')
    opc = ap.parse_args()
    CONGESTIONAMENTO, PREFIXO = opc.congestionamento, opc.prefixo
    SAIDA = opc.saida if os.path.isabs(opc.saida) else os.path.join(BASE, opc.saida)
    mix, kg = mix_pof()
    print('mix POF (SP, kg/ano per capita): ' + ', '.join(f'{c} {kg[c]:.2f} kg ({100 * mix[c]:.1f}%)' for c in mix))
    # Filtros de qualidade fixados DEPOIS da geocodificação e ANTES de gerar (14/10/2026, Entrada 94):
    #  (i) o CEP do resultado no OSM tem de conferir com o do cadastro nos 5 primeiros dígitos
    #      (79% conferem; o resto indica rua homônima ou posição errada);
    # (ii) a geocodificação acerta mais as lojas de empresas maiores (46% das "demais" contra 24% das
    #      microempresas), então a amostra é ESTRATIFICADA pelo porte, com as proporções do cadastro.
    lojas = [l for l in csv.DictReader(open(LOJAS, encoding='utf-8'))
             if float(l['dist_deposito_km']) <= RAIO_KM and l['porte'] in TERCO and l['cep_confere'] == '1']
    lojas.sort(key=lambda l: l['cnpj'])
    cad = Counter(l['porte'] for l in csv.DictReader(open(CADASTRO, encoding='utf-8')) if l['porte'] in TERCO)
    prop = {p: c / sum(cad.values()) for p, c in cad.items()}
    print(f'{len(lojas)} lojas geocodificadas a até {RAIO_KM} km, com porte informado e CEP conferido; '
          f'porte: {dict(Counter(l["porte"] for l in lojas))}; proporções do cadastro: '
          + ', '.join(f'{p} {100 * v:.1f}%' for p, v in sorted(prop.items())))
    if os.path.isdir(SAIDA) and os.listdir(SAIDA) and not opc.acrescentar:
        sys.exit(f'{SAIDA} não está vazia: arquive a versão anterior antes de gerar (nada foi gravado).')
    os.makedirs(SAIDA, exist_ok=True)
    arq_cat = os.path.join(SAIDA, 'catalogo_caso_sp.json')
    catalogo = json.load(open(arq_cat, encoding='utf-8'))['instancias'] if opc.acrescentar and os.path.exists(arq_cat) else []
    for (amostra, n), sem in SEMENTES.items():
        if amostra not in opc.amostras:
            continue
        if all(os.path.exists(os.path.join(SAIDA, f'CCVRPTW-ST_{PREFIXO}{amostra}_{n:03d}_{nv}.json')) for nv in ('MEDIUM', 'HIGH')):
            print(f'{PREFIXO}{amostra}_{n:03d} já existe: não regravada')
            continue
        rng = random.Random(sem)
        # quotas por porte (maiores restos), sorteio dentro de cada estrato
        bruto = {p: n * v for p, v in prop.items()}
        cota = {p: int(math.floor(x)) for p, x in bruto.items()}
        for p in sorted(bruto, key=lambda p: bruto[p] - cota[p], reverse=True)[:n - sum(cota.values())]:
            cota[p] += 1
        escolhidas = []
        for p in sorted(cota):
            escolhidas += rng.sample([l for l in lojas if l['porte'] == p], cota[p])
        rng.shuffle(escolhidas)
        coords = [DEPOSITO] + [(float(l['lat']), float(l['lon'])) for l in escolhidas]
        dur, dist = matriz_osrm(coords, f'v2_sp{amostra}_{n:03d}')
        tempo = [[d / 60.0 * (1 + CONGESTIONAMENTO) for d in row] for row in dur]   # min, com congestionamento
        dkm = [[d / 1000.0 for d in row] for row in dist]
        xy = [((c[1] - DEPOSITO[1]) * 111.32 * math.cos(math.radians(DEPOSITO[0])), (c[0] - DEPOSITO[0]) * 110.57)
              for c in coords]
        rng_v = random.Random(sem * 10)
        tw, prog, aj = [], [], 0
        for i in range(1, n + 1):
            programada = rng_v.random() < P_PROGRAMACAO
            a, b = sorteia(rng_v, FAIXAS_PROG) if programada else SEM_PROG
            if b < tempo[0][i] + 1:                         # garante que a loja seja alcançável (relatado)
                b = math.ceil(tempo[0][i]) + 1; aj += 1
            s = sorteia(rng_v, [(v, c) for v, c in DESCARGA.items()])
            tw.append((float(a), float(b), float(s))); prog.append(programada)
        cats = [sorteia(rng_v, list(mix.items())) for _ in range(n)]
        dem = []
        for l, c in zip(escolhidas, cats):
            lo, hi = ig.DEMAND_RANGES[c]; t = TERCO[l['porte']]; w = (hi - lo) / 3
            dem.append(round(rng_v.uniform(lo + t * w, lo + (t + 1) * w), 1))
        cap = round(rng_v.uniform(600, 800), 0); occ = rng_v.uniform(0.68, 0.88)
        n_cap = max(2, math.ceil(sum(dem) / (cap * occ)))
        n_veic = max(n_cap, math.ceil(frota_edd(tempo, tw, dem, cap) * 1.10))
        dom = Counter(cats).most_common(1)[0][0]; tp = ig.THERMAL_PROFILES[dom]
        clientes = [{'id': i, 'x': round(xy[i][0], 4), 'y': round(xy[i][1], 4),
                     'lat': coords[i][0], 'lon': coords[i][1], 'cnpj': escolhidas[i - 1]['cnpj'],
                     'porte': escolhidas[i - 1]['porte'], 'receiving_scheduled': prog[i - 1],
                     'demand_kg': dem[i - 1], 'ready_time_min': tw[i - 1][0], 'due_date_min': tw[i - 1][1],
                     'service_time_min': tw[i - 1][2], 'product_category': cats[i - 1],
                     'product_description': ig.THERMAL_PROFILES[cats[i - 1]]['description'],
                     'unit_value_BRL_kg': ig.THERMAL_PROFILES[cats[i - 1]]['unit_value_BRL_kg'],
                     'load_value_BRL': round(dem[i - 1] * ig.THERMAL_PROFILES[cats[i - 1]]['unit_value_BRL_kg'], 2)}
                    for i in range(1, n + 1)]
        ajustadas = aj
        for nivel in ('medium', 'high'):
            unc = ig.UNCERTAINTY_LEVELS[nivel]
            nome = f'CCVRPTW-ST_{PREFIXO}{amostra}_{n:03d}_{nivel.upper()}'
            inst = {
                'name': nome, 'family': f'{PREFIXO}{amostra}', 'size': n, 'uncertainty_level': nivel,
                'depot': {'id': 0, 'x': 0.0, 'y': 0.0, 'lat': DEPOSITO[0], 'lon': DEPOSITO[1],
                          'osm': 'node/10689437067', 'ready_time_min': 0.0, 'due_date_min': HORIZONTE},
                'customers': clientes,
                'fleet': {'num_vehicles': n_veic, 'capacity_kg': cap, 'speed_km_h': None,
                          'fixed_cost_BRL': 150.0, 'variable_cost_BRL_km': 2.50},
                'thermal_profile': {
                    'dominant_category': dom, 'description': tp['description'], 'T_set_C': tp['T_set_C'],
                    'T_max_C': tp['T_max_C'], 'T_min_C': tp['T_min_C'], 'Ea_J_mol': tp['Ea_J_mol'], 'A0': tp['A0'],
                    'Q_min': tp['Q_min'], 'sigma_T_C': unc['sigma_T'], 'sigma_t_pct': unc['sigma_t_pct'],
                    'uncertainty_label': unc['label'], 'validation': ig.validate_thermal_params({'category': dom}),
                    'references': tp['references'],
                    'per_category': {c: {'T_set_C': ig.THERMAL_PROFILES[c]['T_set_C'], 'T_max_C': ig.THERMAL_PROFILES[c]['T_max_C'],
                                         'Ea_J_mol': ig.THERMAL_PROFILES[c]['Ea_J_mol'], 'A0': ig.THERMAL_PROFILES[c]['A0'],
                                         'Q_min': ig.THERMAL_PROFILES[c]['Q_min'], 'sigma_T_C': unc['sigma_T'],
                                         'validation': ig.validate_thermal_params({'category': c})}
                                     for c in set(cats)}},
                'co2_params': ig.CO2_PARAMS, 'horizon_minutes': HORIZONTE,
                # matrizes da malha viária real (usadas no lugar das euclidianas)
                'dist_matrix_km': [[round(v, 4) for v in row] for row in dkm],
                'time_matrix_min': [[round(v, 4) for v in row] for row in tempo],
                'generation_metadata': {
                    'caso': 'São Paulo, dados públicos (v2)' + (', planejado com o pico da manhã' if PREFIXO != 'SP' else ''),
                    'seed_amostra': sem, 'amostra': amostra,
                    'num_customers': n, 'raio_km': RAIO_KM, 'congestionamento': CONGESTIONAMENTO,
                    'category_distribution': dict(Counter(cats)), 'mix_pof': {c: round(v, 4) for c, v in mix.items()},
                    'porte_distribution': dict(Counter(l['porte'] for l in escolhidas)),
                    'recebimento_programado': sum(prog), 'janelas_ajustadas_alcance': ajustadas,
                    'fontes': ['Receita Federal, dados abertos do CNPJ, competência 2026-10',
                               'Nominatim/OpenStreetMap (ODbL), geocodificação, 14/10/2026',
                               'OSRM, router.project-osrm.org, perfil car, 14/10/2026',
                               f'TomTom Traffic Index, São Paulo 2025, congestion level {100 * CONGESTIONAMENTO:.1f}% '
                               + ('(average)' if abs(CONGESTIONAMENTO - 0.585) < 1e-9 else '(morning rush hour)'),
                               'Oliveira et al. (2016) Transportes 24(2):53-67, doi:10.14295/transportes.v24i2.1058',
                               'Ramos (2015) dissertação de mestrado, UFMG, Tabela 5.11',
                               'IBGE, POF 2017-2018, SIDRA tabela 2393, São Paulo',
                               'Pithan e Silva (2016) Análises e Indicadores do Agronegócio 11(9), IEA']}}
            json.dump(inst, open(os.path.join(SAIDA, nome + '.json'), 'w', encoding='utf-8'), ensure_ascii=False)
            t_ida = [tempo[0][i] for i in range(1, n + 1)]
            catalogo.append(dict(nome=nome, frota=n_veic, capacidade=cap, demanda=round(sum(dem), 1),
                                 ida_media_min=round(sum(t_ida) / n, 1), ida_max_min=round(max(t_ida), 1),
                                 km_ida_media=round(sum(dkm[0][1:]) / n, 1), servico_medio_min=round(sum(x[2] for x in tw) / n, 1),
                                 programadas=sum(prog), janelas_ajustadas=ajustadas, dominante=dom,
                                 categorias=dict(Counter(cats)), porte=dict(Counter(l['porte'] for l in escolhidas))))
            print(catalogo[-1], flush=True)
    json.dump(dict(mix_pof=mix, kg_pof=kg, instancias=catalogo),
              open(os.path.join(SAIDA, 'catalogo_caso_sp.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
