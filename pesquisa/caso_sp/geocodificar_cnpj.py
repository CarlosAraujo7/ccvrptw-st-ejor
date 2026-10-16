#!/usr/bin/env python3
"""
Geocodifica, pelo endereço do cadastro CNPJ, uma amostra aleatória dos supermercados ativos de
São Paulo (14/10/2026, diário Entrada 92).

Serviço: Nominatim público (OpenStreetMap), conforme a política de uso consultada em 14/10/2026:
uma requisição por segundo no máximo, uma única linha de execução, User-Agent que identifica a
aplicação, resultados guardados em cache local (uma tarefa única, não recorrente).
Busca estruturada: street = "<número> <tipo> <logradouro>", city = São Paulo, country = Brazil.
Só aceita resultado com NÚMERO DA CASA reconhecido e dentro do município de São Paulo: o centro de
uma rua longa pode ficar a quilômetros da loja, então resultados só no nível da rua são descartados.

Ordem: permutação aleatória fixa (semente 2026) das lojas ativas; para quando houver ALVO lojas
geocodificadas a até RAIO_KM do depósito, ou quando a lista acabar. Como a ordem é aleatória, o
conjunto aceito é uma amostra aleatória das lojas geocodificáveis; a taxa de sucesso por porte é
relatada para mostrar eventuais vieses.
Uso: python -I pesquisa/caso_sp/geocodificar_cnpj.py [--alvo 400]
Saída: dados_brutos/cnpj/supermercados_sp_geocodificados.csv; cache em dados_brutos/nominatim/
Dados © colaboradores do OpenStreetMap (ODbL); geocodificação: Nominatim.
"""
import argparse, csv, json, math, os, random, time, urllib.parse, urllib.request
from collections import Counter

AQUI = os.path.dirname(os.path.abspath(__file__))
BRUTOS = os.path.join(AQUI, 'dados_brutos')
ENTRADA = os.path.join(BRUTOS, 'cnpj', 'supermercados_sp_2026-10.csv')
SAIDA = os.path.join(BRUTOS, 'cnpj', 'supermercados_sp_geocodificados.csv')
CACHE = os.path.join(BRUTOS, 'nominatim', 'cache.json')
URL = 'https://nominatim.openstreetmap.org/search?'
UA = 'CCVRPTW-ST academic research, one-off geocoding of supermarket addresses (python urllib)'
DEPOSITO = (-23.5346891, -46.7345446)
RAIO_KM = 15.0


def hav(a, b):
    R = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    return 2 * R * math.asin(math.sqrt(math.sin((la2 - la1) / 2) ** 2 +
                                       math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2))


NOVAS = [0]   # requisições feitas nesta execução (o cache é gravado a cada 25)


def consulta(params, cache):
    chave = urllib.parse.urlencode(sorted(params.items()))
    if chave not in cache:
        req = urllib.request.Request(URL + chave, headers={'User-Agent': UA})
        cache[chave] = json.loads(urllib.request.urlopen(req, timeout=60).read().decode('utf-8'))
        time.sleep(1.1)                                   # ≤ 1 requisição por segundo
        NOVAS[0] += 1
        if NOVAS[0] % 25 == 0:                            # ponto de controle (14/10/2026): nada se perde se cair
            json.dump(cache, open(CACHE, 'w', encoding='utf-8'))
    return cache[chave]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alvo', type=int, default=400)
    a = ap.parse_args()
    lojas = list(csv.DictReader(open(ENTRADA, encoding='utf-8')))
    random.Random(2026).shuffle(lojas)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    cache = json.load(open(CACHE, encoding='utf-8')) if os.path.exists(CACHE) else {}
    aceitas, tentadas, motivo = [], Counter(), Counter()
    try:
        for i, l in enumerate(lojas):
            if sum(1 for x in aceitas if x['dist_deposito_km'] <= RAIO_KM) >= a.alvo:
                break
            tentadas[l['porte']] += 1
            num = l['numero'].strip().upper()
            if not num or not any(ch.isdigit() for ch in num) or num in ('SN', 'S/N', 'S N'):
                motivo['sem número'] += 1; continue
            num = ''.join(ch for ch in num.split()[0] if ch.isdigit())
            rua = f"{num} {l['tipo_logradouro'].strip()} {l['logradouro'].strip()}".strip()
            r = consulta({'street': rua, 'city': 'São Paulo', 'state': 'São Paulo', 'country': 'Brazil',
                          'format': 'jsonv2', 'addressdetails': '1', 'limit': '1'}, cache)
            if not r:
                motivo['sem resultado'] += 1; continue
            r = r[0]; ad = r.get('address', {})
            if not ad.get('house_number'):
                motivo['só nível de rua'] += 1; continue
            if ad.get('city') != 'São Paulo':
                motivo['fora do município'] += 1; continue
            p = (float(r['lat']), float(r['lon']))
            cep_osm = ad.get('postcode', '').replace('-', '')
            aceitas.append(dict(l, lat=p[0], lon=p[1], dist_deposito_km=round(hav(DEPOSITO, p), 3),
                                osm=f"{r.get('osm_type', '')}/{r.get('osm_id', '')}", cep_osm=cep_osm,
                                cep_confere=int(bool(cep_osm) and cep_osm[:5] == l['cep'][:5])))
            if len(aceitas) % 25 == 0:
                print(f'  {i + 1} lojas vistas, {NOVAS[0]} requisições novas, {len(aceitas)} aceitas, '
                      f'{sum(1 for x in aceitas if x["dist_deposito_km"] <= RAIO_KM)} no raio', flush=True)
    finally:
        json.dump(cache, open(CACHE, 'w', encoding='utf-8'))
    with open(SAIDA, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(aceitas[0])); w.writeheader(); w.writerows(aceitas)
    dentro = [x for x in aceitas if x['dist_deposito_km'] <= RAIO_KM]
    ok_porte = Counter(x['porte'] for x in aceitas)
    print(f'\ntentadas {sum(tentadas.values())} | aceitas {len(aceitas)} | a até {RAIO_KM} km: {len(dentro)}')
    print(f'descartes: {dict(motivo)}')
    print('taxa de sucesso por porte: ' + ', '.join(f'{p}: {ok_porte[p]}/{tentadas[p]}' for p in sorted(tentadas)))
    com_cep = [x for x in aceitas if x['cep_osm']]
    print(f'CEP do OSM confere (5 dígitos) em {sum(x["cep_confere"] for x in com_cep)} de {len(com_cep)} com CEP no OSM')
    print(f'gravado: {SAIDA}')


if __name__ == '__main__':
    main()
