#!/usr/bin/env python3
"""
Tempos de viagem OBSERVADOS de veículos de carga (17/10/2026, diário Entrada 115).
Dados: DT-CARGO (Balke & Adenaw, 2023, Data in Brief 48:109246; ODbL), 54 caminhões pesados de 4 frotas alemãs,
setembro de 2021 a abril de 2022; tracks.csv do repositório oficial github.com/TUMFTM/dt-cargo, baixado em 17/10/2026
para dados_brutos/dt_cargo (lido como não confiável: rode com python -I).

Viagem = trecho gravado com mais de 1 km e perda de sinal ≤ 10%, cujo destino é um agrupamento identificado (cid ≥ 0)
e cuja origem é o agrupamento onde terminou o trecho anterior do MESMO caminhão (cid ≥ 0, distância até o início do
trecho ≤ 1 km, mesmo dia); origem ≠ destino. Par = (origem, destino); só pares com ≥ 10 viagens.
Multiplicador observado r = ritmo (min/km) / ritmo médio do par. Os agrupamentos têm raio de cerca de 1 km (DBSCAN,
ε = 1.000 m), então a variação é praticamente toda temporal (mesmo trajeto em dias diferentes).
Medidas: CV por par (mediana, P10, P90); forma de z = (ln r − média)/desvio do par (assimetria, curtose, caudas) e
quantis 0,95/0,99 de r contra a lognormal de mesmo CV. Saída: resultados/variabilidade_dtcargo.json e amostras.
Uso: python -I pesquisa/tempos_observados/analisar_dtcargo.py
"""
import json, os
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(AQUI, 'dados_brutos', 'dt_cargo')
OUT = os.path.join(AQUI, 'resultados'); os.makedirs(OUT, exist_ok=True)
MIN_VIAGENS = 10


def main():
    t = pd.read_csv(os.path.join(D, 'tracks.csv'))
    n0 = len(t)
    t['ini'] = pd.to_datetime(t.start_time, utc=True, format='ISO8601'); t['fim'] = pd.to_datetime(t.stop_time, utc=True, format='ISO8601')
    t = t.sort_values(['vehicle_id', 'ini']).reset_index(drop=True)
    t['dur'] = (t.fim - t.ini).dt.total_seconds() / 60
    t['cid_ant'] = t.groupby('vehicle_id').cid.shift(1)
    t['gap_ant'] = t.groupby('vehicle_id').track_gap.shift(1)       # distância do fim do trecho anterior a este início
    t['dia_ant'] = t.groupby('vehicle_id').ini.shift(1).dt.date
    ok = ((t.distance > 1000) & (t.r_signal_loss.fillna(0) <= 0.10) & (t.cid >= 0) & (t.cid_ant >= 0)
          & (t.cid != t.cid_ant) & (t.gap_ant.fillna(1e9) <= 1000) & (t.dur > 1) & (t.ini.dt.date == t.dia_ant))
    v = t[ok].copy()
    v['ritmo'] = v.dur / (v.distance / 1000)
    v['par'] = list(zip(v.cid_ant.astype(int), v.cid.astype(int)))
    v['n_par'] = v.groupby('par').ritmo.transform('size')
    v = v[v.n_par >= MIN_VIAGENS].copy()
    g = v.groupby('par')
    v['r'] = v.ritmo / g.ritmo.transform('mean')
    v['lr'] = np.log(v.r)
    v['z'] = (v.lr - g.lr.transform('mean')) / g.lr.transform('std')
    par = g.agg(n=('r', 'size'), dur_media=('dur', 'mean'), km=('distance', lambda x: x.mean() / 1000))
    par['cv'] = v.groupby('par').r.std(ddof=1)
    z = v.z.to_numpy(); z = z[np.isfinite(z)]
    razoes = {0.95: [], 0.99: []}
    for k, d in v.groupby('par'):
        cv = d.r.std(ddof=1); s = np.sqrt(np.log1p(cv * cv))
        for q, zq in ((0.95, 1.6449), (0.99, 2.3263)):
            razoes[q].append(np.quantile(d.r, q) / np.exp(-s * s / 2 + s * zq))
    faixas = [(0, 15), (15, 30), (30, 60), (60, 600)]
    res = dict(
        fonte='DT-CARGO (Balke & Adenaw 2023), tracks.csv, 54 caminhões pesados, set/2021-abr/2022',
        trechos_brutos=int(n0), viagens_validas_em_pares=int(len(v)), pares=int(len(par)), min_viagens_por_par=MIN_VIAGENS,
        duracao_mediana_min=float(v.dur.median()), distancia_mediana_km=float((v.distance / 1000).median()),
        cv_par=dict(mediana=float(par.cv.median()), p10=float(par.cv.quantile(0.1)), p90=float(par.cv.quantile(0.9)),
                    ponderada_viagens=float(np.average(par.cv, weights=par.n))),
        cv_por_duracao={f'{a}-{b} min': dict(pares=int(((par.dur_media >= a) & (par.dur_media < b)).sum()),
                                             cv_mediana=float(par.cv[(par.dur_media >= a) & (par.dur_media < b)].median())
                                             if ((par.dur_media >= a) & (par.dur_media < b)).any() else None)
                        for a, b in faixas},
        forma_z=dict(assimetria=float(pd.Series(z).skew()), curtose_excesso=float(pd.Series(z).kurt()),
                     p_acima_1645=float(np.mean(z > 1.6449)), p_acima_2326=float(np.mean(z > 2.3263))),
        quantil_emp_sobre_lognormal={str(q): dict(mediana=float(np.median(r)), p10=float(np.percentile(r, 10)),
                                                  p90=float(np.percentile(r, 90))) for q, r in razoes.items()})
    json.dump(res, open(os.path.join(OUT, 'variabilidade_dtcargo.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    np.savez_compressed(os.path.join(OUT, 'amostras_dtcargo.npz'), z=z)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
