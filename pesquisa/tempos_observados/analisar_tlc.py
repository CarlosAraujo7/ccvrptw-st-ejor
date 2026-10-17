#!/usr/bin/env python3
"""
Tempos de viagem OBSERVADOS: variabilidade dos táxis de Nova York (17/10/2026, diário Entrada 113; análise de
fraquezas A2). Dados: NYC TLC Yellow Taxi Trip Records, março e outubro de 2024 (arquivos oficiais da TLC, baixados
em 17/10/2026 para dados_brutos/nyc_tlc; lidos como não confiáveis: rode com python -I).

Filtros: dias úteis (sem o feriado de 14/10/2024), origem ≠ destino, duração de 2 a 120 min, distância de 0,5 a 30
milhas, velocidade de 2 a 60 mph. Ritmo = duração / distância (min/milha), que tira a heterogeneidade de distância
dentro de um par de zonas. Célula = (zona de origem, zona de destino, hora do embarque); só células com ≥ 30 viagens.
Multiplicador observado r = ritmo / ritmo médio da célula (média 1 em cada célula).

Medidas (gravadas em resultados/variabilidade_tlc.json):
  1. CV de r por célula: mediana, P10, P90 (contra 0,10 e 0,20 do artigo), também por faixa de duração média;
  2. forma: z = (ln r − média) / desvio de cada célula; assimetria, curtose e caudas P(z > 1,645), P(z > 2,326)
     (normal: 5% e 1%); quantis 0,95 e 0,99 de r contra a lognormal de mesmo CV da célula;
  3. fator comum: efeito de (dia, hora) = média de ln r das células naquele dia e hora; fração da variância de ln r
     explicada por ele (contra ρ = 0,5 da Seção 6.6);
  4. amostras para reamostragem: z das viagens (resíduos padronizados, sem o efeito do dia) e efeitos de dia
     padronizados, gravados em resultados/amostras_tlc.npz.
Uso: python -I pesquisa/tempos_observados/analisar_tlc.py
"""
import json, os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

AQUI = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(AQUI, 'dados_brutos', 'nyc_tlc')
OUT = os.path.join(AQUI, 'resultados'); os.makedirs(OUT, exist_ok=True)
MESES = ['2024-03', '2024-10']
FERIADOS = {pd.Timestamp('2024-10-14').date()}
MIN_VIAGENS = 30


def carregar():
    cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance', 'PULocationID', 'DOLocationID']
    partes = []
    for m in MESES:
        t = pq.read_table(os.path.join(D, f'yellow_tripdata_{m}.parquet'), columns=cols).to_pandas()
        ini = pd.Timestamp(m + '-01'); fim = ini + pd.offsets.MonthBegin(1)
        t = t[(t.tpep_pickup_datetime >= ini) & (t.tpep_pickup_datetime < fim)]
        partes.append(t)
    t = pd.concat(partes, ignore_index=True)
    n0 = len(t)
    t['dur'] = (t.tpep_dropoff_datetime - t.tpep_pickup_datetime).dt.total_seconds() / 60
    t['dia'] = t.tpep_pickup_datetime.dt.date
    t['hora'] = t.tpep_pickup_datetime.dt.hour
    util = (t.tpep_pickup_datetime.dt.dayofweek < 5) & ~t.dia.isin(FERIADOS)
    vel = t.trip_distance / (t.dur / 60)
    ok = (util & (t.PULocationID != t.DOLocationID) & t.dur.between(2, 120) & t.trip_distance.between(0.5, 30)
          & vel.between(2, 60) & (t.PULocationID < 264) & (t.DOLocationID < 264))
    t = t[ok].copy()
    t['ritmo'] = t.dur / t.trip_distance
    return t, n0


def main():
    t, n0 = carregar()
    g = t.groupby(['PULocationID', 'DOLocationID', 'hora'])
    t['n_cel'] = g.ritmo.transform('size')
    t = t[t.n_cel >= MIN_VIAGENS].copy()
    g = t.groupby(['PULocationID', 'DOLocationID', 'hora'])
    t['r'] = t.ritmo / g.ritmo.transform('mean')
    t['lr'] = np.log(t.r)
    cel = g.agg(n=('r', 'size'), dur_media=('dur', 'mean'))
    cel['cv'] = t.groupby(['PULocationID', 'DOLocationID', 'hora']).r.std(ddof=1)
    # forma: padroniza ln r dentro da célula
    t['z'] = (t.lr - g.lr.transform('mean')) / g.lr.transform('std')
    z = t.z.to_numpy()
    # quantis de r contra a lognormal de mesmo CV (por célula, depois mediana)
    def q_emp_vs_ln(df):
        cv = df.r.std(ddof=1); s = np.sqrt(np.log1p(cv * cv))
        out = {}
        for q, zq in ((0.95, 1.6449), (0.99, 2.3263)):
            out[q] = (np.quantile(df.r, q), np.exp(-s * s / 2 + s * zq))
        return out
    amostra = cel[cel.n >= 200].index[:3000]
    razoes = {0.95: [], 0.99: []}
    for k in amostra:
        d = q_emp_vs_ln(t.loc[g.groups[k]])
        for q in razoes:
            razoes[q].append(d[q][0] / d[q][1])
    # fator comum: efeito de (dia, hora)
    ef = t.groupby(['dia', 'hora']).lr.transform('mean')
    var_total = t.lr.var(); var_dia = ef.var()
    t['z_res'] = (t.lr - ef) / (t.lr - ef).std()
    ef_dh = t.groupby(['dia', 'hora']).lr.mean()
    faixas = [(0, 10), (10, 20), (20, 40), (40, 120)]
    cv_faixa = {f'{a}-{b} min': dict(celulas=int(((cel.dur_media >= a) & (cel.dur_media < b)).sum()),
                                     cv_mediana=float(cel.cv[(cel.dur_media >= a) & (cel.dur_media < b)].median()))
                for a, b in faixas}
    # CV entre dias (limite inferior da variabilidade temporal): variância das médias diárias de r em cada célula,
    # descontada a variância de amostragem (variância dentro do dia / nº de viagens), em células com ≥ 3 viagens em
    # ≥ 10 dias
    cd = t.groupby(['PULocationID', 'DOLocationID', 'hora', 'dia']).r.agg(['mean', 'var', 'size'])
    cd = cd[cd['size'] >= 3]
    gc = cd.groupby(level=[0, 1, 2])
    agg = gc.agg(nd=('mean', 'size'), v_entre=('mean', 'var'))
    agg['v_amostra'] = gc.apply(lambda x: float(np.mean(x['var'] / x['size'])))
    agg = agg[agg.nd >= 10]
    agg['cv_dias'] = np.sqrt(np.clip(agg.v_entre - agg.v_amostra, 0, None))
    cv_dias = dict(celulas=int(len(agg)), mediana=float(agg.cv_dias.median()), p10=float(agg.cv_dias.quantile(0.1)),
                   p90=float(agg.cv_dias.quantile(0.9)))
    res = dict(
        fonte='NYC TLC Yellow Taxi Trip Records, 2024-03 e 2024-10 (dias úteis)', viagens_brutas=int(n0),
        cv_entre_dias=cv_dias,
        viagens_validas_em_celulas=int(len(t)), celulas=int(len(cel)), min_viagens_por_celula=MIN_VIAGENS,
        cv_celula=dict(mediana=float(cel.cv.median()), p10=float(cel.cv.quantile(0.1)), p90=float(cel.cv.quantile(0.9)),
                       ponderada_viagens=float(np.average(cel.cv, weights=cel.n))),
        cv_por_duracao=cv_faixa,
        forma_z=dict(assimetria=float(pd.Series(z).skew()), curtose_excesso=float(pd.Series(z).kurt()),
                     p_acima_1645=float(np.mean(z > 1.6449)), p_acima_2326=float(np.mean(z > 2.3263)),
                     p_abaixo_m1645=float(np.mean(z < -1.6449))),
        quantil_emp_sobre_lognormal={str(q): dict(mediana=float(np.median(v)), p10=float(np.percentile(v, 10)),
                                                  p90=float(np.percentile(v, 90)), celulas=len(v)) for q, v in razoes.items()},
        fator_dia=dict(fracao_variancia=float(var_dia / var_total), n_dia_hora=int(len(ef_dh)),
                       desvio_efeito=float(ef_dh.std())))
    json.dump(res, open(os.path.join(OUT, 'variabilidade_tlc.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    rng = np.random.default_rng(20261017)
    zr = t.z_res.to_numpy(); zr = zr[np.isfinite(zr)]
    ed = ((ef_dh - ef_dh.mean()) / ef_dh.std()).to_numpy()
    np.savez_compressed(os.path.join(OUT, 'amostras_tlc.npz'), z_res=rng.choice(zr, 400000, replace=False), z_dia=ed,
                        z_bruto=rng.choice(z[np.isfinite(z)], 400000, replace=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
