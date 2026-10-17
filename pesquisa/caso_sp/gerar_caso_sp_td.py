#!/usr/bin/env python3
"""
Caso São Paulo com tempos de viagem dependentes do horário (16/10/2026, diário Entrada 101).

A partir das 16 instâncias SP (instances/caso_sp) e das respostas brutas do OSRM já baixadas
(dados_brutos/osrm, sem nova consulta), gera duas variantes com as MESMAS lojas, janelas, serviços,
demandas, frota e perfil térmico; só o modelo de tempo de viagem muda:
  SPT  instances/caso_sp_td      tempos dependentes do horário (src/tempo_dependente.py, modelo de
       Ichoua, Gendreau e Potvin, 2003): comprimento de cada arco = tempo do OSRM (fluxo livre, min);
       fator por hora do relógio f(h) = 1 + c(h)/100, c(h) = congestionamento médio dos dias úteis na
       cidade de São Paulo (TomTom Traffic Index 2025, perfil horário extraído por extrair_tomtom.py);
       a operação começa às 06:00 (t = 0).
  SPM  instances/caso_sp_media_operacao   tempos ESTÁTICOS com o fator médio do mesmo perfil nas horas
       de operação (06:00-22:00, o horizonte das instâncias): a melhor constante que um planejador
       estático poderia usar sem conhecer o horário de cada entrega. (O caso SP original usa a média
       anual de todas as horas e dias, 58,5%.)
Conferência: o tempo do OSRM × 1,585 arredondado reproduz a matriz estática das instâncias SP.
Uso: python -I pesquisa/caso_sp/gerar_caso_sp_td.py
"""
import copy, glob, json, os, sys

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(AQUI, '..', '..'))
sys.path.insert(0, AQUI)
import gerar_caso_sp as g   # noqa: E402   (só constantes e matriz_osrm, que lê o cache)

PERFIL = os.path.join(AQUI, 'dados_brutos', 'tomtom', 'perfil_horario_sao_paulo_2025.json')
HORA_INICIO = 6.0
SAIDAS = {'SPT': os.path.join(BASE, 'instances', 'caso_sp_td'),
          'SPM': os.path.join(BASE, 'instances', 'caso_sp_media_operacao')}


def main():
    p = json.load(open(PERFIL, encoding='utf-8'))
    fator = [float(p['fator_tempo_cidade'][str(h)]) for h in range(24)]
    horas_op = range(int(HORA_INICIO), int(HORA_INICIO + g.HORIZONTE / 60))      # 06 a 21 h
    f_op = sum(fator[h] for h in horas_op) / len(horas_op)
    print('fator por hora (dias úteis, cidade):', ' '.join(f'{h:02d}h {fator[h]:.3f}' for h in range(24)))
    print(f'fator médio nas horas de operação (06-22 h): {f_op:.4f}  (estático do caso SP: 1,585)')
    for d in SAIDAS.values():
        if os.path.isdir(d) and os.listdir(d):
            sys.exit(f'{d} não está vazia: arquive antes de gerar (nada foi gravado).')
        os.makedirs(d, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(BASE, 'instances', 'caso_sp', 'CCVRPTW-ST_SP*.json'))):
        inst = json.load(open(f, encoding='utf-8'))
        gm = inst['generation_metadata']; n = gm['num_customers']
        coords = [(inst['depot']['lat'], inst['depot']['lon'])] + [(c['lat'], c['lon']) for c in inst['customers']]
        dur, _ = g.matriz_osrm(coords, f"v2_sp{gm['amostra']}_{n:03d}")
        ff = [[d / 60.0 for d in row] for row in dur]
        erro = max(abs(round(x * (1 + gm['congestionamento']), 4) - y)
                   for rf, rt in zip(ff, inst['time_matrix_min']) for x, y in zip(rf, rt))
        assert erro < 1e-9, f'{inst["name"]}: matriz do OSRM não reproduz a estática (erro {erro})'
        for pref, d in SAIDAS.items():
            novo = copy.deepcopy(inst)
            nome = inst['name'].replace('_SP', '_' + pref, 1)
            novo['name'] = nome; novo['family'] = inst['family'].replace('SP', pref, 1)
            del novo['time_matrix_min']
            meta = novo['generation_metadata']
            meta['derivada_de'] = inst['name']
            if pref == 'SPT':
                novo['time_matrix_ff_min'] = [[round(x, 6) for x in row] for row in ff]
                novo['td'] = {'modelo': 'Ichoua, Gendreau & Potvin (2003): velocidade constante por hora do relógio',
                              'hora_inicio': HORA_INICIO, 'fator_hora': fator,
                              'fonte': p['fonte'] + '; média dos dias úteis, cidade'}
                meta['caso'] = 'São Paulo, dados públicos (v2), tempos dependentes do horário'
                meta['congestionamento'] = None
            else:
                novo['time_matrix_min'] = [[round(x * f_op, 4) for x in row] for row in ff]
                meta['caso'] = 'São Paulo, dados públicos (v2), fator médio das horas de operação'
                meta['congestionamento'] = round(f_op - 1, 6)
            json.dump(novo, open(os.path.join(d, nome + '.json'), 'w', encoding='utf-8'), ensure_ascii=False)
        print(inst['name'], '-> SPT, SPM', flush=True)


if __name__ == '__main__':
    main()
