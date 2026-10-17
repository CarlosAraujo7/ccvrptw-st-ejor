#!/usr/bin/env python3
"""
Reavaliação dos planos escolhidos com tempos de viagem de distribuição OBSERVADA (17/10/2026, diário Entrada 113).

Lê as amostras padronizadas de analisar_tlc.py (táxis de Nova York, dias úteis de 2024): resíduos z das viagens
(sem o efeito comum do dia) e efeitos de (dia, hora) padronizados. Para cada instância, 1.000 cenários comuns a todos
os métodos (mesma semente por instância); o multiplicador do trecho que chega ao cliente j é
    M0  lognormal do artigo (CV κ da instância): a avaliação fora da amostra original (_oosC.json);
    M1  forma observada: ln ζ = μ + σ_κ·e, e reamostrado dos resíduos observados, μ tal que E[ζ] = 1;
    M2  M1 + fator do dia observado: ln ζ = μ + σ_κ·(√ρ·e_dia + √(1−ρ)·e), ρ = fração observada (≈ 0,11);
    M3  estresse: como M2, mas com o CV OBSERVADO para a duração nominal do trecho (0,29 abaixo de 10 min, 0,27 de
        10 a 20, 0,25 de 20 a 40, 0,20 acima de 40 min), independentemente do κ com que o plano foi feito.
Planos: o escolhido (certificado mais barato) de DET+C, BUF0.90+C, BUF0.95+C, CAL, SIM+C e ACAL em cada execução
do benchmark v2 (incerteza média e alta), do SP e do SP com horário. Mede: % de planos que cumprem α e P média.
Uso: python pesquisa/tempos_observados/reavaliar_observado.py
"""
import glob, json, math, os, sys
from collections import defaultdict
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(AQUI, '..', '..'))
sys.path.insert(0, os.path.join(BASE, 'src')); sys.path.insert(0, os.path.join(BASE, 'pesquisa'))
sys.path.insert(0, os.path.join(BASE, 'pesquisa', 'caso_sp'))

ALPHA = 0.90; K = 1000
OUTD = os.path.join(AQUI, 'resultados')
ESC = ['C-BUF70C', 'C-BUF80C', 'C-BUF90C', 'C-BUF95C']
METODOS = ['C-DETC', 'C-BUF90C', 'C-BUF95C', 'CAL', 'C-SIMC', 'ACAL']
CONJ = [('benchmark (média e alta)', 'results_c', 'results_c_acal_bench', 'instances/generated', ''),
        ('SP', 'results_c_sp', 'results_c_acal_sp', 'instances/caso_sp', ''),
        ('SP com horário', 'results_c_sp_td', 'results_c_acal_sp_td', 'instances/caso_sp_td', 'td')]
CV_DUR = [(10, 0.29), (20, 0.27), (40, 0.25), (math.inf, 0.20)]


def escolhida(arq):
    if not os.path.exists(arq):
        return None
    S = json.load(open(arq, encoding='utf-8'))
    c = [s for s in S if s['feasible'] and (s.get('p_cert') or 0) >= ALPHA]
    return min(c, key=lambda s: s['f1_cost_BRL']) if c else None


def multiplicadores(sig, e):
    """ζ = exp(μ + sig·e) com E[ζ] = 1 estimado na própria amostra (sig pode ser vetor por coluna)."""
    x = np.exp(sig * e)
    return x / x.mean(axis=0, keepdims=True)


def main():
    import rodar_caso_sp
    import analisar_c as ac
    import ccvrp_c as cc
    A = np.load(os.path.join(OUTD, 'amostras_tlc.npz'))
    rho = json.load(open(os.path.join(OUTD, 'variabilidade_tlc.json'), encoding='utf-8'))['fator_dia']['fracao_variancia']
    rel = open(os.path.join(OUTD, 'relatorio_reavaliacao_observada.txt'), 'w', encoding='utf-8')

    def log(t=''):
        print(t, flush=True); rel.write(t + '\n')
    log(f'Reavaliação com tempos de distribuição observada (táxis de Nova York, 2024); ρ do dia = {rho:.3f}; K = {K}')
    resumo = {}
    rodar_caso_sp.aplicar_matrizes()     # vale para os três tipos de instância (euclidiana, viária, dependente do horário)
    for nome, res, res_acal, idir, var in CONJ:
        ac.INST_DIR = os.path.join(BASE, idir); ac._cache.clear()
        nomes = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(BASE, res, 'C-SIMC', '*_oosC.json'))})
        if 'benchmark' in nome:
            nomes = [n_ for n_ in nomes if not n_.endswith('LOW')]
        P = defaultdict(lambda: defaultdict(list))   # modelo -> método -> [P]
        for nm in nomes:
            inst, T, sc, _ = ac._ctx(nm, 1)
            n = len(inst['customers']); kap = inst['thermal_profile']['sigma_t_pct']
            sk = math.sqrt(math.log1p(kap * kap))
            rng = np.random.default_rng(99991 + sum(map(ord, nm)))
            e = rng.choice(A['z_res'], size=(K, n + 1)); ed = rng.choice(A['z_dia'], size=(K, 1))
            mix = math.sqrt(rho) * ed + math.sqrt(1 - rho) * e
            tt = {'M0': sc['tt'], 'M1': multiplicadores(sk, e), 'M2': multiplicadores(sk, mix)}
            for sd in range(10):
                planos = {a: escolhida(os.path.join(BASE, res, a, f'{nm}__{a}__seed{sd}_solutions.json'))
                          for a in ['C-DETC', 'C-SIMC'] + ESC}
                lad = [planos[a] for a in ESC if planos[a]]
                planos['CAL'] = min(lad, key=lambda s: s['f1_cost_BRL']) if lad else None
                planos['ACAL'] = escolhida(os.path.join(BASE, res_acal, 'C-ACALC-B', f'{nm}__C-ACALC-B__seed{sd}_solutions.json'))
                for a in METODOS:
                    x = planos.get(a)
                    if x is None:
                        continue
                    # M3: CV observado pela duração nominal do trecho de chegada a cada cliente deste plano
                    sig = np.full(n + 1, sk)
                    for r in x['routes']:
                        prev = 0
                        for c in r:
                            tau = T.chegada(prev, c, 0.0) if hasattr(T, 'chegada') else T[prev][c]
                            cv = next(v for lim, v in CV_DUR if tau < lim)
                            sig[c] = math.sqrt(math.log1p(cv * cv)); prev = c
                    tt['M3'] = multiplicadores(sig[None, :], mix)
                    for mod in ('M0', 'M1', 'M2', 'M3'):
                        on = np.ones(K, dtype=bool)
                        for r, d0 in zip(x['routes'], x['partida_min']):
                            on &= cc.pontual_rota(r, inst, T, tt[mod], d0)
                        P[mod][a].append(float(on.mean()))
        log(f'\n=== {nome} ({len(nomes)} instâncias) — % de planos escolhidos que cumprem α (P média) ===')
        log(f'  {"método":10s} ' + ' '.join(f'{m:>16s}' for m in ('M0 lognormal', 'M1 forma obs.', 'M2 + fator dia', 'M3 CV observado')))
        resumo[nome] = {}
        for a in METODOS:
            cel = []
            for mod in ('M0', 'M1', 'M2', 'M3'):
                v = P[mod][a]
                cel.append(f'{100 * np.mean(np.array(v) >= ALPHA):6.1f}% ({np.mean(v):.3f})' if v else '--')
                resumo[nome].setdefault(a, {})[mod] = dict(cumpre=float(100 * np.mean(np.array(v) >= ALPHA)) if v else None,
                                                           P=float(np.mean(v)) if v else None, n=len(v))
            log(f'  {a:10s} ' + ' '.join(f'{c_:>16s}' for c_ in cel))
    json.dump(resumo, open(os.path.join(OUTD, 'resumo_reavaliacao_observada.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    rel.close()


if __name__ == '__main__':
    main()
