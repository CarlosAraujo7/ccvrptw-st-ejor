#!/usr/bin/env python3
"""
Calibração preditiva (PCAL) — etapa 2: escolher o nível do buffer ANTES da busca, com custo de uma busca só
(17/10/2026, diário Entrada 108; análise de fraquezas de 17/10, item A1).

Piloto construtivo, para cada nível candidato L ∈ {nominal, 0,70, 0,80, 0,90, 0,95}:
  1. gera N_PILOTO giant tours (metade EDD perturbado, metade vizinho mais próximo, como a população inicial do motor);
  2. decodifica com o Split e melhora com a busca local (com eliminação de rotas), com os tempos de planejamento de L;
  3. simula cada plano (partidas em 0) em K_PILOTO cenários próprios do piloto, independentes dos de busca,
     certificação e avaliação, e o declara "certificado" se P̂ ≥ α⁺ = α + 1,645·√(α(1−α)/K_PILOTO);
  4. C̃(L) = custo do plano certificado mais barato do nível (∞ se nenhum).
Nível previsto L* = argmin C̃(L) (empate: o nível mais alto, mais seguro; se nenhum certifica: 0,95). Lógica da
Proposição 4: comparar custos certificados entre níveis. A busca completa roda só em L*.

Avaliação offline: a busca em L* é a execução existente desse nível (DET+C para o nominal), e o plano escolhido é a
certificada mais barata dela; orçamento = 1 busca + piloto (tempo medido e relatado).
Desenvolvimento: v1 (results_c_v1_cindom). Teste: v2, SP, SP com horário, Atenas. Os parâmetros N_PILOTO, K_PILOTO e
a regra foram fixados ANTES de olhar o desempenho nos conjuntos de teste (diário Entrada 108).
Uso: python pesquisa/calibracao_preditiva.py --prever  (grava as previsões)  e  --avaliar
"""
import argparse, json, math, os, random, sys, time
from multiprocessing import Pool
import numpy as np

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(BASE, 'src')); sys.path.insert(0, os.path.join(BASE, 'pesquisa'))
sys.path.insert(0, os.path.join(BASE, 'pesquisa', 'caso_sp'))

ALPHA = 0.90
N_PILOTO = int(os.environ.get('PCAL_N', 20))
K_PILOTO = 200
SEM_NOMINAL = os.environ.get('PCAL_SEM_NOMINAL') == '1'
SUF = os.environ.get('PCAL_SUF', '')
ALVO = ALPHA + 1.645 * math.sqrt(ALPHA * (1 - ALPHA) / K_PILOTO)
NIVEIS = [(None, 'C-DETC'), (0.70, 'C-BUF70C'), (0.80, 'C-BUF80C'), (0.90, 'C-BUF90C'), (0.95, 'C-BUF95C')]
CONJ = {'v1': ('results_c_v1_cindom', 'instances/generated'), 'v2': ('results_c', 'instances/generated'),
        'SP': ('results_c_sp', 'instances/caso_sp'), 'SPT': ('results_c_sp_td', 'instances/caso_sp_td'),
        'ATH': ('results_c_atenas', 'instances/caso_atenas')}
OUT = os.path.join(BASE, 'pesquisa', 'resultados_calibracao_preditiva')


def _init():
    import rodar_caso_sp
    rodar_caso_sp.aplicar_matrizes()


def prever(tarefa):
    """Piloto construtivo de uma (instância, semente): C̃(L) por nível e o nível previsto."""
    import nsga2_simheuristic as ns
    import ccvrp_c as cc
    import thermal_route as tr
    inst_path, seed = tarefa
    inst = ns.load_instance(inst_path)
    n = len(inst['customers']); cv = inst['thermal_profile'].get('sigma_t_pct', 0.0)
    t0 = time.process_time()
    tt = tr._lognormal_media1(np.random.default_rng(777 + 31 * seed + sum(map(ord, inst['name']))), cv, (K_PILOTO, n + 1))
    res = {}
    for q, met in NIVEIS:
        if SEM_NOMINAL and q is None:
            continue
        ctx = cc.Contexto(inst, seed, buffer_q=q, decoder='split', p_ls=0.5)
        rng = random.Random(1000 + seed)
        melhor = math.inf; n_cert = 0; custos = []
        for k in range(N_PILOTO):
            ch = ns._edd_perturbed(inst, rng) if k % 2 == 0 else ns._nn_chromosome(inst, ctx.dist, rng)
            R, ok = cc.split_decode(ch, ctx)
            if not ok or not R:
                continue
            R = cc.busca_local(R, ctx, rng)
            custo = sum(cc._custo_rota(r, ctx) for r in R) + max(0, len(R) - inst['fleet']['num_vehicles']) * ns.EXTRA_VEHICLE_COST
            on = np.ones(K_PILOTO, dtype=bool)
            for r in R:
                on &= cc.pontual_rota(r, inst, ctx.ttime, tt, 0.0, ctx.delta_tol)
            p = float(on.mean()); custos.append(custo)
            if p >= ALVO:
                n_cert += 1; melhor = min(melhor, custo)
        res[met] = dict(c=melhor, n_cert=n_cert, c_min=min(custos) if custos else math.inf)
    vals = [(res[m]['c'], -i, m) for i, (_, m) in enumerate(NIVEIS) if m in res and math.isfinite(res[m]['c'])]
    prev = min(vals)[2] if vals else 'C-BUF95C'
    return dict(instance=inst['name'], seed=seed, previsto=prev, piloto=res, cpu_s=time.process_time() - t0)


def rodar_previsoes(conjs, workers):
    import glob
    os.makedirs(OUT, exist_ok=True)
    for c in conjs:
        res_dir, idir = CONJ[c]
        arq = os.path.join(OUT, f'previsoes_{c}{SUF}.json')
        if os.path.exists(arq):
            print(c, 'já previsto'); continue
        nomes = sorted({os.path.basename(f).split('__')[0] for f in glob.glob(os.path.join(BASE, res_dir, 'C-SIMC', '*_oosC.json'))})
        tarefas = [(os.path.join(BASE, idir, nm + '.json'), s) for nm in nomes for s in range(10)]
        env = {'SP': '', 'SPT': 'td', 'ATH': 'atenas'}.get(c, '')
        os.environ['CCVRP_SP_VARIANTE'] = env
        t = time.time()
        with Pool(workers, initializer=_init) as pool:
            P = pool.map(prever, tarefas, chunksize=1)
        json.dump(P, open(arq, 'w', encoding='utf-8'), indent=1)
        print(f'{c}: {len(P)} previsões em {time.time() - t:.0f} s; CPU média do piloto {np.mean([p["cpu_s"] for p in P]):.1f} s', flush=True)


def avaliar(conjs):
    import calibracao_preditiva_base as b
    rel = open(os.path.join(OUT, f'relatorio_calibracao_preditiva{SUF}_{"_".join(conjs)}.txt'), 'w', encoding='utf-8')

    def log(t=''):
        print(t, flush=True); rel.write(t + '\n')
    log(f'CALIBRAÇÃO PREDITIVA (PCAL) — piloto construtivo com {N_PILOTO} planos por nível e {K_PILOTO} cenários; '
        f'alvo α⁺ = {ALVO:.3f}; avaliação offline nos 1.000 cenários comuns')
    nomes_conj = {'v1': 'v1 (desenvolvimento)', 'v2': 'v2', 'SP': 'SP', 'SPT': 'SP com horário', 'ATH': 'Atenas'}
    for c in conjs:
        res_dir, _ = CONJ[c]
        P = {(p['instance'], p['seed']): p for p in json.load(open(os.path.join(OUT, f'previsoes_{c}{SUF}.json'), encoding='utf-8'))}
        R = b.carregar(res_dir)

        def pcal(R_, i, s, sem, P=P):
            return R_[(i, s)].get(P[(i, s)]['previsto']), 1
        regras = {'SIM+C': b.regra_sim(1), 'SIM2 (melhor de 2)': b.regra_sim(2), 'BUF0.90+C': b.regra_degraus(['C-BUF90C']),
                  'BUF0.95+C': b.regra_degraus(['C-BUF95C']), 'CAL (4 degraus)': b.regra_degraus(list(b.DEG)),
                  'PCAL (1 busca)': pcal}
        b.avaliar(nomes_conj[c], R, regras, log=log)
        dist = {}
        for p in P.values():
            dist[p['previsto']] = dist.get(p['previsto'], 0) + 1
        cpu = np.mean([p['cpu_s'] for p in P.values()])
        log(f'  nível previsto: ' + ', '.join(f'{m} {100 * v / len(P):.0f}%' for m, v in sorted(dist.items())) +
            f'; CPU do piloto: média {cpu:.1f} s por instância e semente')
    rel.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prever', action='store_true'); ap.add_argument('--avaliar', action='store_true')
    ap.add_argument('--conj', nargs='+', default=['v1'])
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    if a.prever:
        rodar_previsoes(a.conj, a.workers)
    if a.avaliar:
        avaliar(a.conj)
