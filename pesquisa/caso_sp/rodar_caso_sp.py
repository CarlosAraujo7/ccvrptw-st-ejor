#!/usr/bin/env python3
"""
Experimentos do caso realista em São Paulo (11/10/2026, diário Entrada 89).

As instâncias de instances/caso_sp/ trazem matrizes de distância e tempo da malha viária real
('dist_matrix_km', 'time_matrix_min'). O código da busca (src/) NÃO é alterado: cada processo
deste script substitui nsga2_simheuristic.build_matrices por uma versão que devolve essas
matrizes quando a instância as tem (o motor do Caminho C só usa distâncias e tempos por meio
dessa função), e aponta analisar_c.INST_DIR para instances/caso_sp/.

Desenho (fixado antes de rodar): 8 instâncias × 7 métodos (C-DET, C-DETC, escada C-BUF70C/
80C/90C/95C — que compõe o CAL —, C-SIMC) × 10 sementes = 560 execuções, com a configuração da
execução principal (ALGO_CONFIG do experiment_runner: Split, busca local p_ls = 0,5, eliminação
de rotas, carga homogênea). Avaliação fora da amostra em 1.000 cenários comuns (analisar_c),
refinamento por simulação (refinamento_simulacao) e referência de custo do PyVRP.
Uso (na raiz do projeto, depois da execução principal):
  python pesquisa/caso_sp/rodar_caso_sp.py --rodar --workers 10
  python pesquisa/caso_sp/rodar_caso_sp.py --analisar --workers 10
"""
import argparse, glob, json, os, sys
from multiprocessing import Pool

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(AQUI, '..', '..'))
sys.path.insert(0, os.path.join(BASE, 'src')); sys.path.insert(0, os.path.join(BASE, 'pesquisa'))
INST_DIR = os.path.join(BASE, 'instances', 'caso_sp')
RES = os.path.join(BASE, 'results_c_sp')
OUT = os.path.join(BASE, 'pesquisa', 'resultados_caso_sp')
# Variante planejada com o congestionamento do pico da manhã (15/10/2026, análise de fraquezas A3).
# Escolhida pela variável de ambiente, herdada pelos processos filhos do Pool:
#   CCVRP_SP_VARIANTE=pico  →  instances/caso_sp_pico, results_c_sp_pico, pesquisa/resultados_caso_sp_pico
if os.environ.get('CCVRP_SP_VARIANTE') == 'pico':
    INST_DIR = os.path.join(BASE, 'instances', 'caso_sp_pico')
    RES = os.path.join(BASE, 'results_c_sp_pico')
    OUT = os.path.join(BASE, 'pesquisa', 'resultados_caso_sp_pico')
# Variante com tempos de viagem dependentes do horário (16/10/2026, Entrada 101; src/tempo_dependente.py):
#   CCVRP_SP_VARIANTE=td  →  instances/caso_sp_td, results_c_sp_td, pesquisa/resultados_caso_sp_td
#   (CCVRP_SP_INST_DIR / CCVRP_SP_RES sobrepõem as pastas, para testes)
if os.environ.get('CCVRP_SP_VARIANTE') == 'td':
    INST_DIR = os.path.join(BASE, 'instances', 'caso_sp_td')
    RES = os.path.join(BASE, 'results_c_sp_td')
    OUT = os.path.join(BASE, 'pesquisa', 'resultados_caso_sp_td')
#   CCVRP_SP_VARIANTE=media  →  instances/caso_sp_media_operacao (estático, fator médio das horas de
#   operação do mesmo perfil horário), results_c_sp_media, pesquisa/resultados_caso_sp_media
if os.environ.get('CCVRP_SP_VARIANTE') == 'media':
    INST_DIR = os.path.join(BASE, 'instances', 'caso_sp_media_operacao')
    RES = os.path.join(BASE, 'results_c_sp_media')
    OUT = os.path.join(BASE, 'pesquisa', 'resultados_caso_sp_media')
#   CCVRP_SP_VARIANTE=atenas  →  instances/caso_atenas (dados operacionais de um operador logístico de Atenas,
#   pesquisa/caso_certh; Entrada 103), results_c_atenas, pesquisa/resultados_caso_atenas
if os.environ.get('CCVRP_SP_VARIANTE') == 'atenas':
    INST_DIR = os.path.join(BASE, 'instances', 'caso_atenas')
    RES = os.path.join(BASE, 'results_c_atenas')
    OUT = os.path.join(BASE, 'pesquisa', 'resultados_caso_atenas')
INST_DIR = os.environ.get('CCVRP_SP_INST_DIR', INST_DIR)
RES = os.environ.get('CCVRP_SP_RES', RES)
METODOS = ['C-DET', 'C-DETC', 'C-BUF70C', 'C-BUF80C', 'C-BUF90C', 'C-BUF95C', 'C-SIMC']
SEMENTES = 10


def aplicar_matrizes():
    """Substitui build_matrices neste processo (instâncias com matriz viária) e o diretório de
    instâncias da análise fora da amostra."""
    import nsga2_simheuristic as ns
    if getattr(ns, '_caso_sp', False):
        return
    original = ns.build_matrices

    def bm(inst):
        if 'td' in inst:                 # tempos dependentes do horário: comprimentos em fluxo livre + perfil
            import tempo_dependente as tdm   # (+ CV próprio de cada arco, quando a instância o traz)
            cva = inst['td'].get('cv_arco')
            return inst['dist_matrix_km'], tdm.MatrizTD(inst['time_matrix_ff_min'], inst['td']['fator_hora'],
                                                        inst['td']['hora_inicio'], cv_arco=cva,
                                                        cv_ref=inst['thermal_profile']['sigma_t_pct'] if cva else None)
        if 'time_matrix_min' in inst:
            return inst['dist_matrix_km'], inst['time_matrix_min']
        return original(inst)
    ns.build_matrices = bm; ns._caso_sp = True
    import analisar_c as ac
    ac.INST_DIR = INST_DIR


def _worker(task):
    aplicar_matrizes()
    import experiment_runner as er
    return er._worker(task)


def rodar(workers):
    import experiment_runner as er
    tarefas = []
    for f in sorted(glob.glob(os.path.join(INST_DIR, 'CCVRPTW-ST_*.json'))):
        nome = os.path.basename(f)[:-5]
        for alg in METODOS:
            cfg = er.ALGO_CONFIG[alg]; out_dir = os.path.join(RES, alg)
            for s in range(SEMENTES):
                key = f'{nome}__{alg}__seed{s}'
                if os.path.exists(os.path.join(out_dir, key + '_metrics.json')):
                    continue
                tarefas.append({'inst_path': f, 'algo': alg, 'algo_label': alg, 'seed': s, 'pop_size': 100,
                                'n_gen': 200, 'mc_samples': cfg['mc_samples'], 'runner': cfg['runner'],
                                'out_dir': out_dir, 'result_key': key, 'use_multizone': False, 'n_zones': 1,
                                'dedup': cfg.get('dedup', False), 'c_params': cfg.get('c_params')})
    print(f'{len(tarefas)} execuções a rodar', flush=True)
    with Pool(workers, initializer=aplicar_matrizes) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tarefas), 1):
            print(f"  [{i}/{len(tarefas)}] {r.get('key', '')} {r.get('status', '')} {r.get('elapsed', '')}", flush=True)


def analisar(workers):
    aplicar_matrizes()
    import analisar_c as ac
    arqs = sorted(glob.glob(os.path.join(RES, 'C-*', '*_solutions.json')))
    with Pool(workers, initializer=aplicar_matrizes) as pool:
        pool.map(ac._job, arqs, chunksize=2)
    os.makedirs(OUT, exist_ok=True)
    rel = open(os.path.join(OUT, 'relatorio_caso_sp.txt'), 'w', encoding='utf-8')

    def log(s=''):
        print(s, flush=True); rel.write(s + '\n')
    ac.analisar([RES], OUT, log)
    rel.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rodar', action='store_true')
    ap.add_argument('--analisar', action='store_true')
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    if a.rodar:
        rodar(a.workers)
    if a.analisar:
        analisar(a.workers)
