#!/usr/bin/env python3
"""
Planos escolhidos sob a forma OBSERVADA nos caminhões do DT-CARGO (17/10/2026, diário Entrada 115).
Modelo M1C: ln ζ = μ + σ_κ·e, com e reamostrado dos resíduos padronizados das viagens de caminhão
(resultados/amostras_dtcargo.npz; cauda bem mais pesada que a dos táxis) e σ_κ o da instância; E[ζ] = 1.
Para cada execução e método: (a) a escolha original (certificada com a lognormal) avaliada em 1.000 cenários M1C;
(b) a frente guardada recertificada em 500 cenários M1C independentes, e a nova escolha avaliada nos mesmos 1.000.
Sementes próprias (não alteram os resultados de reavaliar_observado.py e recertificar_observado.py).
Uso: python pesquisa/tempos_observados/reavaliar_caminhoes.py
"""
import glob, json, math, os, sys
from collections import defaultdict
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(AQUI, '..', '..'))
sys.path.insert(0, os.path.join(BASE, 'src')); sys.path.insert(0, os.path.join(BASE, 'pesquisa'))
sys.path.insert(0, os.path.join(BASE, 'pesquisa', 'caso_sp')); sys.path.insert(0, AQUI)
import reavaliar_observado as ro       # noqa: E402
import recertificar_observado as rc    # noqa: E402

ALPHA = 0.90; K_EVAL = 1000; K_CERT = 500


def main():
    import rodar_caso_sp
    import analisar_c as ac
    import ccvrp_c as cc
    zc = np.load(os.path.join(ro.OUTD, 'amostras_dtcargo.npz'))['z']
    rodar_caso_sp.aplicar_matrizes()
    rel = open(os.path.join(ro.OUTD, 'relatorio_caminhoes.txt'), 'w', encoding='utf-8')

    def log(t=''):
        print(t, flush=True); rel.write(t + '\n')
    log('Forma observada nos caminhões (DT-CARGO): escolha original e recertificada, % que cumpre α em 1.000 cenários M1C')
    resumo = {}
    for nome, res, res_acal, idir, _ in ro.CONJ:
        ac.INST_DIR = os.path.join(BASE, idir); ac._cache.clear()
        nomes = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(BASE, res, 'C-SIMC', '*_oosC.json'))})
        if 'benchmark' in nome:
            nomes = [n_ for n_ in nomes if not n_.endswith('LOW')]
        R = defaultdict(lambda: {'orig': [], 'rec': [], 'tem': [], 'dc': []})
        for nm in nomes:
            inst, T, _, _ = ac._ctx(nm, 1)
            n = len(inst['customers']); kap = inst['thermal_profile']['sigma_t_pct']
            sk = math.sqrt(math.log1p(kap * kap))
            tte = ro.multiplicadores(sk, np.random.default_rng(55501 + sum(map(ord, nm))).choice(zc, size=(K_EVAL, n + 1)))
            ttc = ro.multiplicadores(sk, np.random.default_rng(66601 + sum(map(ord, nm))).choice(zc, size=(K_CERT, n + 1)))

            def p(s, tt, K):
                on = np.ones(K, dtype=bool)
                for r, d0 in zip(s['routes'], s['partida_min']):
                    on &= cc.pontual_rota(r, inst, T, tt, d0)
                return float(on.mean())
            for sd in range(10):
                F = {a: rc.frente(os.path.join(BASE, res, a, f'{nm}__{a}__seed{sd}_solutions.json')) for a in ['C-DETC', 'C-SIMC'] + ro.ESC}
                F['CAL'] = [s for a in ro.ESC for s in F[a]]
                F['ACAL'] = rc.frente(os.path.join(BASE, res_acal, 'C-ACALC-B', f'{nm}__C-ACALC-B__seed{sd}_solutions.json'))
                for a in ro.METODOS:
                    sols = F.get(a) or []
                    orig = [s for s in sols if (s.get('p_cert') or 0) >= ALPHA]
                    if not orig:
                        continue
                    s0 = min(orig, key=lambda s: s['f1_cost_BRL'])
                    R[a]['orig'].append(p(s0, tte, K_EVAL) >= ALPHA)
                    cert = [s for s in sols if p(s, ttc, K_CERT) >= ALPHA]
                    R[a]['tem'].append(bool(cert))
                    if cert:
                        s1 = min(cert, key=lambda s: s['f1_cost_BRL'])
                        R[a]['rec'].append(p(s1, tte, K_EVAL) >= ALPHA)
                        R[a]['dc'].append(100 * (s1['f1_cost_BRL'] / s0['f1_cost_BRL'] - 1))
        log(f'\n=== {nome} ===   original | recertificada (com plano; Δcusto mediano)')
        resumo[nome] = {}
        for a in ro.METODOS:
            x = R[a]
            o = 100 * np.mean(x['orig']); r = 100 * np.mean(x['rec']) if x['rec'] else float('nan')
            t = 100 * np.mean(x['tem']); dc = float(np.median(x['dc'])) if x['dc'] else float('nan')
            log(f'  {a:10s} {o:6.1f}% | {r:6.1f}% (com plano {t:5.1f}%; Δcusto {dc:+.2f}%)')
            resumo[nome][a] = dict(original=o, recertificada=r, com_plano=t, dcusto=dc)
    json.dump(resumo, open(os.path.join(ro.OUTD, 'resumo_caminhoes.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    rel.close()


if __name__ == '__main__':
    main()
