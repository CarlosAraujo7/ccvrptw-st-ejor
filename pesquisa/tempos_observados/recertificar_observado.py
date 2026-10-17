#!/usr/bin/env python3
"""
Recertificação com tempos de distribuição OBSERVADA (17/10/2026, diário Entrada 113).

Pergunta: a etapa de certificação, que pode simular qualquer distribuição, absorve o erro de especificação do modelo
de tempos? Para cada execução e método, todas as soluções viáveis da frente final guardada são recertificadas em
500 cenários do modelo observado (M1, M2 ou M3 de reavaliar_observado.py), com uma semente INDEPENDENTE da avaliação;
o plano escolhido passa a ser o mais barato com P ≥ α nessa recertificação, e é avaliado nos mesmos 1.000 cenários de
avaliação de reavaliar_observado.py. Mede: % de execuções com plano recertificado, % de escolhas que cumprem α na
avaliação, e a variação mediana de custo contra a escolha original (certificada com a lognormal).
Uso: python pesquisa/tempos_observados/recertificar_observado.py
"""
import glob, json, math, os, sys
from collections import defaultdict
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(AQUI, '..', '..'))
sys.path.insert(0, os.path.join(BASE, 'src')); sys.path.insert(0, os.path.join(BASE, 'pesquisa'))
sys.path.insert(0, os.path.join(BASE, 'pesquisa', 'caso_sp')); sys.path.insert(0, AQUI)
import reavaliar_observado as ro   # noqa: E402

ALPHA = 0.90; K_EVAL = 1000; K_CERT = 500
OUTD = ro.OUTD


def frente(arq):
    if not os.path.exists(arq):
        return []
    return [s for s in json.load(open(arq, encoding='utf-8')) if s['feasible']]


def main():
    import rodar_caso_sp
    import analisar_c as ac
    import ccvrp_c as cc
    A = np.load(os.path.join(OUTD, 'amostras_tlc.npz'))
    rho = json.load(open(os.path.join(OUTD, 'variabilidade_tlc.json'), encoding='utf-8'))['fator_dia']['fracao_variancia']
    rodar_caso_sp.aplicar_matrizes()
    rel = open(os.path.join(OUTD, 'relatorio_recertificacao_observada.txt'), 'w', encoding='utf-8')

    def log(t=''):
        print(t, flush=True); rel.write(t + '\n')
    log(f'Recertificação com tempos observados ({K_CERT} cenários independentes); avaliação em {K_EVAL} cenários (mesmos de reavaliar_observado.py)')
    resumo = {}
    for nome, res, res_acal, idir, _ in ro.CONJ:
        ac.INST_DIR = os.path.join(BASE, idir); ac._cache.clear()
        nomes = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(BASE, res, 'C-SIMC', '*_oosC.json'))})
        if 'benchmark' in nome:
            nomes = [n_ for n_ in nomes if not n_.endswith('LOW')]
        R = defaultdict(lambda: defaultdict(list))   # modelo -> método -> [(tem, cumpre, dcusto)]
        for nm in nomes:
            inst, T, sc, _ = ac._ctx(nm, 1)
            n = len(inst['customers']); kap = inst['thermal_profile']['sigma_t_pct']
            sk = math.sqrt(math.log1p(kap * kap))
            rng = np.random.default_rng(99991 + sum(map(ord, nm)))          # MESMA semente da avaliação
            e = rng.choice(A['z_res'], size=(K_EVAL, n + 1)); ed = rng.choice(A['z_dia'], size=(K_EVAL, 1))
            mix = math.sqrt(rho) * ed + math.sqrt(1 - rho) * e
            rngc = np.random.default_rng(424242 + sum(map(ord, nm)))       # certificação independente
            ec = rngc.choice(A['z_res'], size=(K_CERT, n + 1)); edc = rngc.choice(A['z_dia'], size=(K_CERT, 1))
            mixc = math.sqrt(rho) * edc + math.sqrt(1 - rho) * ec
            for sd in range(10):
                F = {a: frente(os.path.join(BASE, res, a, f'{nm}__{a}__seed{sd}_solutions.json')) for a in ['C-DETC', 'C-SIMC'] + ro.ESC}
                F['CAL'] = [s for a in ro.ESC for s in F[a]]
                F['ACAL'] = frente(os.path.join(BASE, res_acal, 'C-ACALC-B', f'{nm}__C-ACALC-B__seed{sd}_solutions.json'))
                for a in ro.METODOS:
                    sols = F.get(a) or []
                    orig = [s for s in sols if (s.get('p_cert') or 0) >= ALPHA]
                    if not orig:
                        continue
                    c0 = min(s['f1_cost_BRL'] for s in orig)
                    for mod in ('M1', 'M2', 'M3'):
                        avaliados = []
                        for s in sols:
                            if mod == 'M3':
                                sig = np.full(n + 1, sk)
                                for r in s['routes']:
                                    prev = 0
                                    for c in r:
                                        tau = T.chegada(prev, c, 0.0) if hasattr(T, 'chegada') else T[prev][c]
                                        cv = next(v for lim, v in ro.CV_DUR if tau < lim)
                                        sig[c] = math.sqrt(math.log1p(cv * cv)); prev = c
                                ttc, tte = ro.multiplicadores(sig[None, :], mixc), ro.multiplicadores(sig[None, :], mix)
                            else:
                                base_c, base_e = (ec, e) if mod == 'M1' else (mixc, mix)
                                ttc, tte = ro.multiplicadores(sk, base_c), ro.multiplicadores(sk, base_e)
                            onc = np.ones(K_CERT, dtype=bool)
                            for r, d0 in zip(s['routes'], s['partida_min']):
                                onc &= cc.pontual_rota(r, inst, T, ttc, d0)
                            if onc.mean() >= ALPHA:
                                avaliados.append((s['f1_cost_BRL'], s, tte))
                        if not avaliados:
                            R[mod][a].append((False, False, np.nan)); continue
                        f1, s, tte = min(avaliados, key=lambda x: x[0])
                        on = np.ones(K_EVAL, dtype=bool)
                        for r, d0 in zip(s['routes'], s['partida_min']):
                            on &= cc.pontual_rota(r, inst, T, tte, d0)
                        R[mod][a].append((True, bool(on.mean() >= ALPHA), 100 * (f1 / c0 - 1)))
            print(f'  {nm}', flush=True)
        log(f'\n=== {nome} — recertificação com tempos observados: com plano recertificado / escolha cumpre α / Δcusto mediano ===')
        resumo[nome] = {}
        for a in ro.METODOS:
            cel = []
            for mod in ('M1', 'M2', 'M3'):
                L = R[mod][a]
                tem = 100 * np.mean([x[0] for x in L]) if L else np.nan
                cum = 100 * np.mean([x[1] for x in L if x[0]]) if any(x[0] for x in L) else np.nan
                dc = np.nanmedian([x[2] for x in L if x[0]]) if any(x[0] for x in L) else np.nan
                cel.append(f'{tem:5.1f}% {cum:5.1f}% {dc:+5.2f}%')
                resumo[nome].setdefault(a, {})[mod] = dict(com_plano=float(tem), cumpre=float(cum), dcusto=float(dc), n=len(L))
            log(f'  {a:10s} ' + ' | '.join(f'{m}: {c_}' for m, c_ in zip(('M1', 'M2', 'M3'), cel)))
    json.dump(resumo, open(os.path.join(OUTD, 'resumo_recertificacao_observada.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    rel.close()


if __name__ == '__main__':
    main()
