#!/usr/bin/env python3
"""
Perfis horários de temperatura externa a partir do NOAA ISD (Integrated
Surface Database, global-hourly; Smith, Lott & Vose, 2011, BAMS 92(6):704-708).
01/10/2026 — modelo de temperatura dependente da rota (diário Entrada 70).

Para cada estação: lê os CSVs horários 2011-2020, converte UTC → hora local
(fuso fixo, sem horário de verão), mantém o campo TMP só com código de
qualidade aceito, monta um perfil de 24 h por dia (interpolação linear
circular quando faltam até 6 horas; dias com mais lacunas são descartados) e
grava, para o MÊS MAIS QUENTE (maior média das máximas diárias):
  - todos os perfis diários (cenários empíricos de temperatura externa);
  - o perfil médio (cenário nominal);
  - estatísticas-resumo (Tmax média, amplitude diária média, desvio-padrão da
    média diária — mesma definição de ASHRAE 2017, cap. 14, §3.2 — e a forma
    do perfil normalizado, para comparar com a Tabela 6 do ASHRAE).

Uso: python pesquisa/clima/processar_isd.py <pasta_com_csvs_isd>
Saída: pesquisa/clima/perfis_<estacao>.json e pesquisa/clima/resumo_clima.txt
"""

import csv, glob, json, os, sys
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

ESTACOES = {
    # id ISD: (nome, fuso UTC fixo em horas)
    '83780099999': ('São Paulo (Congonhas)', -3),
    '82398099999': ('Fortaleza (Pinto Martins)', -3),
    '82111099999': ('Manaus (Eduardo Gomes)', -4),
    '82899099999': ('Recife (Guararapes)', -3),
    '83746099999': ('Rio de Janeiro (Galeão)', -3),
    '83971099999': ('Porto Alegre (Salgado Filho)', -3),
    '83377099999': ('Brasília', -3),
    '83840099999': ('Curitiba (Afonso Pena)', -3),
    '83229099999': ('Salvador', -3),
    '06260099999': ('De Bilt (NL)', 1),
    '08221099999': ('Madrid (Barajas)', 1),
    '72259003927': ('Dallas-Fort Worth (EUA)', -6),
    # caso de Atenas (16/10/2026, Entrada 103): UTC+3 é a hora legal de verão (EEST), em vigor no mês mais quente
    '16716199999': ('Atenas (Eleftherios Venizelos)', 3),
}
ASHRAE_T6 = [0.88, 0.92, 0.95, 0.98, 1.00, 0.98, 0.91, 0.74, 0.55, 0.38, 0.23, 0.13,
             0.05, 0.00, 0.00, 0.06, 0.14, 0.24, 0.39, 0.50, 0.59, 0.68, 0.75, 0.82]  # horas 1..24
OUT = os.path.dirname(os.path.abspath(__file__))


def ler(estacao, pasta):
    _, fuso = ESTACOES[estacao]
    obs = defaultdict(lambda: defaultdict(list))   # dia -> hora -> [temps]
    for f in sorted(glob.glob(os.path.join(pasta, f'{estacao}_*.csv'))):
        with open(f, encoding='utf-8', errors='replace') as fh:
            for r in csv.DictReader(fh):
                tmp = r.get('TMP', '')
                if not tmp or ',' not in tmp:
                    continue
                v, q = tmp.split(',')[:2]
                if v.strip() in ('+9999', '9999') or q.strip() not in ('0', '1', '4', '5', '9', 'A', 'C', 'I', 'M', 'P', 'R', 'U'):
                    continue
                t = datetime.fromisoformat(r['DATE']) + timedelta(hours=fuso)
                obs[t.date()][t.hour].append(int(v) / 10.0)
    perfis = {}
    for dia, horas in obs.items():
        h = np.full(24, np.nan)
        for k, vs in horas.items():
            h[k] = np.mean(vs)
        if np.isnan(h).sum() > 6:
            continue
        ok = ~np.isnan(h)
        x = np.arange(24)
        h[~ok] = np.interp(x[~ok], x[ok], h[ok], period=24)
        perfis[dia] = h
    return perfis


def main(pasta, resumo='resumo_clima.txt'):
    linhas = []
    for est, (nome, _) in ESTACOES.items():
        if not glob.glob(os.path.join(pasta, f'{est}_*.csv')):
            continue
        perfis = ler(est, pasta)
        if len(perfis) < 300:
            linhas.append(f'{nome:30s} dados insuficientes ({len(perfis)} dias)')
            continue
        por_mes = defaultdict(list)
        for d, h in perfis.items():
            por_mes[d.month].append((d, h))
        mes = max(por_mes, key=lambda m: np.mean([h.max() for _, h in por_mes[m]]))
        D = sorted(por_mes[mes], key=lambda x: x[0])
        H = np.array([h for _, h in D])
        tmax, tmin, tmed = H.max(1), H.min(1), H.mean(1)
        medio = H.mean(0)
        frac = (medio.max() - medio) / (medio.max() - medio.min())   # forma normalizada (0 = pico)
        ash = np.roll(np.array(ASHRAE_T6), 1)   # índice 0 = hora 0 (= hora 24 da tabela)
        rmse_ash = float(np.sqrt(np.mean((frac - ash) ** 2)))
        json.dump({'estacao': est, 'nome': nome, 'mes': mes, 'anos': '2011-2020',
                   'fonte': 'NOAA ISD global-hourly', 'n_dias': len(D),
                   'perfil_nominal_C': medio.round(2).tolist(),
                   'perfis_diarios_C': H.round(1).tolist(),
                   'datas': [str(d) for d, _ in D]},
                  open(os.path.join(OUT, f'perfis_{est}.json'), 'w', encoding='utf-8'))
        linhas.append(f'{nome:30s} mês {mes:2d}  dias {len(D):4d}  Tmax {tmax.mean():5.1f}  '
                      f'amplitude {np.mean(tmax - tmin):5.1f}  sd(média diária) {tmed.std(ddof=1):4.2f}  '
                      f'pico às {int(medio.argmax()):2d}h  RMSE vs. ASHRAE T6 {rmse_ash:4.2f}')
    txt = ('Resumo climático do mês mais quente, 2011-2020 (NOAA ISD), hora local fixa\n'
           'sd(média diária): ASHRAE 2017 cap. 14 §3.2; RMSE: perfil normalizado vs. Tabela 6 do ASHRAE\n\n'
           + '\n'.join(linhas) + '\n')
    open(os.path.join(OUT, resumo), 'w', encoding='utf-8').write(txt)
    print(txt)


if __name__ == '__main__':
    main(sys.argv[1], *sys.argv[2:3])   # 2º argumento opcional: arquivo de resumo (padrão resumo_clima.txt)
