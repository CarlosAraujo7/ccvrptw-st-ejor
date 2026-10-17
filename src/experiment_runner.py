#!/usr/bin/env python3
"""
Experiment Runner — CCVRPTW-ST
===============================
Executa o benchmark completo com múltiplos algoritmos, instâncias e sementes
usando execução paralela via multiprocessing.

Algoritmos disponíveis:
  NSGA2-MC   — NSGA-II + Monte Carlo embarcado (mc=30)  [proposto]
  NSGA2-DET  — NSGA-II com avaliação determinística     [baseline]
  NSGA2-MC10 — NSGA-II + Monte Carlo leve (mc=10)       [sensibilidade]
  MOEA-RS    — Busca aleatória multiobjetivo + MC        [baseline inferior]
  MOEAD-MC   — MOEA/D + Monte Carlo (mc=30)              [baseline externo, frente 4.1,
                23/08/2026 — ver pesquisa/plano_implementacao_moead.md]

Exemplos de uso:
  # Execução completa (54 instâncias × 3 algos × 5 seeds):
  python experiment_runner.py --workers 8

  # Teste rápido (n=25, 2 seeds, 50 gerações):
  python experiment_runner.py --quick --workers 4

  # Só instâncias C1 e R1, apenas NSGA2-MC:
  python experiment_runner.py --families C1 R1 --algorithms NSGA2-MC --seeds 5

  # Retomar execução (pula runs já concluídas):
  python experiment_runner.py --workers 8 --resume

  # B2 (multi-compartimento térmico, 22/08/2026) -- roda a grade normal E,
  # ADICIONALMENTE, uma variante com use_multizone=True por valor de Z pedido.
  # Não afeta os runs baseline (mesma chave/pasta de sempre) -- resultados B2
  # vão para results_b2/{ALGO}_Z{Z}/, com algorithm label "{ALGO}+B2z{Z}"
  # nos metrics.json, para não misturar com o baseline na análise.
  python experiment_runner.py --families C1 --sizes 25 --zones 2 3 --workers 4
"""

import argparse
import csv
import glob
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.normpath(os.path.join(SRC_DIR, '..'))
INST_DIR    = os.path.join(BASE_DIR, 'instances', 'generated')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
# B2 (22/08/2026): pasta SEPARADA do baseline -- nunca escreve em cima dos
# 1.080 resultados já existentes em results/, mesmo por acidente.
RESULTS_B2_DIR = os.path.join(BASE_DIR, 'results_b2')
# Curvas de convergência (06/09/2026, diário Entrada 52): pasta SEPARADA
# também -- mesmo motivo do B2 acima. --fixed-hv-ref escreve aqui, nunca em
# results/ ou results_b2/, mesmo que o result_key coincida byte a byte com um
# já existente lá (o significado do campo 'hypervolume'/*_hv.csv muda: passa
# a usar ponto de referência FIXO por instância, não mais auto-referenciado).
RESULTS_CONV_DIR = os.path.join(BASE_DIR, 'results_conv')

# Os 5 algoritmos da comparação central (SBPO original + MOEAD-MC, frente
# 4.1) -- MESMO conjunto que pesquisa/reprocessar_hipervolume.py usa pro
# 'hypervolume_corrected' já publicado na Tabela 6 do manuscrito. --fixed-hv-ref
# reusa a mesma fórmula de ponto de referência (max dos pontos de TODAS as
# execuções já existentes destes 5 algoritmos, por instância, ×1.1+1e-9) --
# garante que o último checkpoint de cada nova curva bate com o HV final já
# publicado (mesma seed, mesma régua).
ALGOS_COMPARACAO_CENTRAL = {
    'NSGA2-MC', 'NSGA2-MC10', 'NSGA2-DET', 'MOEA-RS', 'MOEAD-MC',
}

# Parâmetros padrão
DEFAULT_POP_SIZE   = 100
DEFAULT_N_GEN      = 200
DEFAULT_MC_SAMPLES = 30
DEFAULT_SEEDS      = 5
DEFAULT_WORKERS    = 10

# Definição dos algoritmos
ALGO_CONFIG: Dict[str, dict] = {
    'NSGA2-MC': {
        'runner':     'nsga2',
        'mc_samples': 30,
        'label':      'NSGA-II + Monte Carlo (proposto)',
    },
    'NSGA2-DET': {
        'runner':     'nsga2',
        'mc_samples': 0,        # 0 = modo determinístico (T = T_set)
        'label':      'NSGA-II Determinístico (baseline)',
    },
    'NSGA2-MC10': {
        'runner':     'nsga2',
        'mc_samples': 10,
        'label':      'NSGA-II + Monte Carlo leve (sensibilidade)',
    },
    'MOEA-RS': {
        'runner':     'random',
        'mc_samples': 30,
        'label':      'Busca Aleatória Multiobjetivo (baseline inferior)',
    },
    'MOEAD-MC': {
        'runner':     'moead',
        'mc_samples': 30,
        'label':      'MOEA/D + Monte Carlo (baseline externo, frente 4.1)',
    },
    # Eliminação de duplicatas (29/09/2026, pesquisa/plano_reavaliacao_oos.md):
    # a frente final do NSGA2-DET é quase toda de clones (ex.: 99 soluções,
    # ~19 distintas) -- no MC o ruído esconde os clones. Estas variantes
    # testam se a vantagem do estocástico sobrevive a um DET sem esse handicap.
    'NSGA2-DET-UNQ': {
        'runner':     'nsga2',
        'mc_samples': 0,
        'dedup':      True,
        'label':      'NSGA-II Determinístico, sem duplicatas',
    },
    'NSGA2-MC10-UNQ': {
        'runner':     'nsga2',
        'mc_samples': 10,
        'dedup':      True,
        'label':      'NSGA-II + Monte Carlo leve, sem duplicatas',
    },
    # Caminho C (03/10/2026, pesquisa/plano_caminho_C.md; src/ccvrp_c.py):
    # f2 nominal do modelo térmico dependente da rota, f3 sobre a duração da
    # rota, saída do depósito como decisão e restrição de chance de
    # pontualidade P(nenhum atraso > δ) ≥ α. mc_samples aqui = nº de cenários
    # de tempo de viagem da SIM (0 = não avalia a restrição).
    'C-SIM':    {'runner': 'nsga2c', 'mc_samples': 30, 'c_params': {'mc_rel': 30, 'alpha': 0.90},
                 'label': 'Caminho C: simheurística (P(pontual) ≥ 0,90, 30 cenários)'},
    'C-SIM95':  {'runner': 'nsga2c', 'mc_samples': 30, 'c_params': {'mc_rel': 30, 'alpha': 0.95},
                 'label': 'Caminho C: simheurística (P(pontual) ≥ 0,95)'},
    'C-SIM10':  {'runner': 'nsga2c', 'mc_samples': 10, 'c_params': {'mc_rel': 10, 'alpha': 0.90},
                 'label': 'Caminho C: simheurística, ablação 10 cenários'},
    'C-SIM50':  {'runner': 'nsga2c', 'mc_samples': 50, 'c_params': {'mc_rel': 50, 'alpha': 0.90},
                 'label': 'Caminho C: simheurística, ablação 50 cenários'},
    'C-MOEAD':  {'runner': 'moeadc', 'mc_samples': 30, 'c_params': {'mc_rel': 30, 'alpha': 0.90},
                 'label': 'Caminho C: MOEA/D com a mesma avaliação e restrição da SIM'},
    # Certificação por simulação longa (500 cenários novos) ao fim da busca —
    # etapa padrão de simheurística; versões DET/BUF certificadas separam o
    # efeito da busca com simulação do efeito da certificação (Entrada 74).
    'C-SIMC':   {'runner': 'nsga2c', 'mc_samples': 30, 'c_params': {'mc_rel': 30, 'alpha': 0.90, 'mc_cert': 500},
                 'label': 'Caminho C: simheurística + certificação (500 cenários)'},
    'C-SIMB80C': {'runner': 'nsga2c', 'mc_samples': 30,
                  'c_params': {'mc_rel': 30, 'alpha': 0.90, 'buffer_q': 0.80, 'mc_cert': 500},
                  'label': 'Caminho C: híbrido — busca com simulação sobre plano com folga 0,80 + certificação'},
    'C-DETC':   {'runner': 'nsga2c', 'mc_samples': 0, 'c_params': {'mc_cert': 500},
                 'label': 'Caminho C: determinístico + certificação'},
    'C-BUF80C': {'runner': 'nsga2c', 'mc_samples': 0, 'c_params': {'buffer_q': 0.80, 'mc_cert': 500},
                 'label': 'Caminho C: folga 0,80 + certificação'},
    'C-BUF90C': {'runner': 'nsga2c', 'mc_samples': 0, 'c_params': {'buffer_q': 0.90, 'mc_cert': 500},
                 'label': 'Caminho C: folga 0,90 + certificação'},
    'C-DET':    {'runner': 'nsga2c', 'mc_samples': 0, 'c_params': {},
                 'label': 'Caminho C: determinístico (não vê a pontualidade)'},
    'C-BUF80':  {'runner': 'nsga2c', 'mc_samples': 0, 'c_params': {'buffer_q': 0.80},
                 'label': 'Caminho C: folga de segurança, tempos no quantil 0,80'},
    'C-BUF90':  {'runner': 'nsga2c', 'mc_samples': 0, 'c_params': {'buffer_q': 0.90},
                 'label': 'Caminho C: folga de segurança, tempos no quantil 0,90'},
    # Frente 4.3 (28/08/2026) -- ablação de granularidade MC, ver
    # pesquisa/plano_ablacao_4_3.md. mc=0/10/30 já cobertos pelos algoritmos
    # acima (NSGA2-DET/MC10/MC); estes 3 preenchem os níveis intermediários
    # pedidos em proposta_extensao_ejor.md §4.3 ("testar 5/20/50 também").
    'NSGA2-MC5': {
        'runner':     'nsga2',
        'mc_samples': 5,
        'label':      'NSGA-II + Monte Carlo (mc=5, ablação 4.3)',
    },
    'NSGA2-MC20': {
        'runner':     'nsga2',
        'mc_samples': 20,
        'label':      'NSGA-II + Monte Carlo (mc=20, ablação 4.3)',
    },
    'NSGA2-MC50': {
        'runner':     'nsga2',
        'mc_samples': 50,
        'label':      'NSGA-II + Monte Carlo (mc=50, ablação 4.3)',
    },
    # Frente 4.3, componentes 2-4 (29/08/2026, ver pesquisa/plano_ablacao_4_3.md
    # §5) -- mc_samples=10 fixo (nível "suficiente" já estabelecido). O nível
    # de referência de cada componente (cx=0.90/mutação mix/init 20-40-40) já
    # existe em results/NSGA2-MC10/ e não é reproduzido aqui.
    'NSGA2-CX070': {
        'runner': 'nsga2', 'mc_samples': 10, 'crossover_rate': 0.70,
        'label': 'NSGA-II + MC10, crossover=0,70 (ablação 4.3, componente 2)',
    },
    'NSGA2-CX098': {
        'runner': 'nsga2', 'mc_samples': 10, 'crossover_rate': 0.98,
        'label': 'NSGA-II + MC10, crossover=0,98 (ablação 4.3, componente 2)',
    },
    'NSGA2-2OPT': {
        'runner': 'nsga2', 'mc_samples': 10, 'mutation_operator': '2opt',
        'label': 'NSGA-II + MC10, mutação só 2-opt (ablação 4.3, componente 3)',
    },
    'NSGA2-OROPT': {
        'runner': 'nsga2', 'mc_samples': 10, 'mutation_operator': 'oropt',
        'label': 'NSGA-II + MC10, mutação só or-opt (ablação 4.3, componente 3)',
    },
    'NSGA2-INITNN': {
        'runner': 'nsga2', 'mc_samples': 10, 'init_mix': (0.0, 1.0, 0.0),
        'label': 'NSGA-II + MC10, init 100% NN (ablação 4.3, componente 4)',
    },
    'NSGA2-INITRND': {
        'runner': 'nsga2', 'mc_samples': 10, 'init_mix': (0.0, 0.0, 1.0),
        'label': 'NSGA-II + MC10, init 100% aleatório (ablação 4.3, componente 4)',
    },
    'NSGA2-INITEQ': {
        'runner': 'nsga2', 'mc_samples': 10, 'init_mix': (0.34, 0.33, 0.33),
        'label': 'NSGA-II + MC10, init 34/33/33 (ablação 4.3, componente 4)',
    },
    # Sensibilidade B2 (29/08/2026, pendência da Entrada 22/26 sobre
    # GAMMA_ZONE_COLD/ZONE_*_COST_KM_MULTIPLE serem estimativas não
    # verificadas). Só fazem sentido com use_multizone=True -- usar sempre
    # com `--zones 2 3`. Nível de referência = NSGA2-MC10 com os valores
    # padrão do módulo, já existente em results_b2/NSGA2-MC10_Z{2,3}/ (não
    # precisa rodar de novo). A tarefa baseline (não-B2) que build_task_list()
    # sempre gera pra qualquer algo é um duplicado inofensivo do baseline de
    # NSGA2-MC10 nestes 4 -- ignorada na análise (só resultados_b2/ é lido).
    'NSGA2-GAMMAHALF': {
        'runner': 'nsga2', 'mc_samples': 10, 'gamma_zone_cold': 0.00675,
        'label': 'NSGA-II + MC10, GAMMA_ZONE_COLD/2 (sensibilidade B2)',
    },
    'NSGA2-GAMMADOUBLE': {
        'runner': 'nsga2', 'mc_samples': 10, 'gamma_zone_cold': 0.027,
        'label': 'NSGA-II + MC10, GAMMA_ZONE_COLD×2 (sensibilidade B2)',
    },
    'NSGA2-KMHALF': {
        'runner': 'nsga2', 'mc_samples': 10,
        'zone_loading_km_mult': 1.30, 'zone_unloading_km_mult': 1.025,
        'label': 'NSGA-II + MC10, ZONE_*_KM_MULTIPLE/2 (sensibilidade B2)',
    },
    'NSGA2-KMDOUBLE': {
        'runner': 'nsga2', 'mc_samples': 10,
        'zone_loading_km_mult': 5.20, 'zone_unloading_km_mult': 4.10,
        'label': 'NSGA-II + MC10, ZONE_*_KM_MULTIPLE×2 (sensibilidade B2)',
    },
}

# Caminho C, F2 (04/10/2026, diário Entrada 75): variantes adicionais da tese
# refinada ("a pontualidade se garante com folga calibrada por simulação") —
# escada de folgas certificadas (o procedimento calibrado CAL é composto na
# análise a partir dela), MOEA/D certificado, nível de serviço 0,95 e ablação
# do nº de cenários da simheurística certificada.
for _q in (0.70, 0.95):
    ALGO_CONFIG[f'C-BUF{int(_q * 100)}C'] = {
        'runner': 'nsga2c', 'mc_samples': 0, 'c_params': {'buffer_q': _q, 'mc_cert': 500},
        'label': f'Caminho C: folga {_q:.2f} + certificação'}
ALGO_CONFIG['C-MOEADC'] = {'runner': 'moeadc', 'mc_samples': 30,
                           'c_params': {'mc_rel': 30, 'alpha': 0.90, 'mc_cert': 500},
                           'label': 'Caminho C: MOEA/D + certificação'}
ALGO_CONFIG['C-SIMC95'] = {'runner': 'nsga2c', 'mc_samples': 30,
                           'c_params': {'mc_rel': 30, 'alpha': 0.95, 'mc_cert': 500},
                           'label': 'Caminho C: simheurística + certificação, α = 0,95'}
for _q in (0.90, 0.95, 0.99):
    ALGO_CONFIG[f'C-BUF{int(_q * 100)}C95'] = {
        'runner': 'nsga2c', 'mc_samples': 0, 'c_params': {'buffer_q': _q, 'alpha': 0.95, 'mc_cert': 500},
        'label': f'Caminho C: folga {_q:.2f} + certificação, α = 0,95'}
for _m in (10, 50):
    ALGO_CONFIG[f'C-SIMC{_m}'] = {'runner': 'nsga2c', 'mc_samples': _m,
                                  'c_params': {'mc_rel': _m, 'alpha': 0.90, 'mc_cert': 500},
                                  'label': f'Caminho C: simheurística + certificação, {_m} cenários (ablação)'}
# Robustez para revisor (08/10/2026, Entrada 81): simheurística com 100 cenários
# na busca -- testa se mais simulação dentro da busca muda a comparação com as folgas.
ALGO_CONFIG['C-SIMC100'] = {'runner': 'nsga2c', 'mc_samples': 100,
                            'c_params': {'mc_rel': 100, 'alpha': 0.90, 'mc_cert': 500},
                            'label': 'Caminho C: simheurística + certificação, 100 cenários na busca'}
# Caminho 1 (10/10/2026, diário Entrada 87): o experimento principal usa carga HOMOGÊNEA
# (categoria dominante, CINETICA_POR_CLIENTE = False em ccvrp_c). O B2 compara as zonas
# com uma base de compartimento único com carga MISTA (cinética de cada produto, mesmo
# modelo das zonas) — variantes *-MIX, rodadas sem --zones e gravadas junto do B2.
for _b in ('C-DET', 'C-BUF90C', 'C-SIMC'):
    ALGO_CONFIG[f'{_b}-MIX'] = {**ALGO_CONFIG[_b],
                                'c_params': {**ALGO_CONFIG[_b].get('c_params', {}), 'cinetica_por_cliente': True},
                                'label': ALGO_CONFIG[_b].get('label', _b) + ' — compartimento único, carga mista'}
# Sensibilidade do B2 (Entrada 87; mesmo desenho da Entrada 44): metade e dobro de
# GAMMA_ZONE_COLD (0,0135) e, juntos, de ZONE_LOADING/UNLOADING_COST_KM_MULTIPLE
# (2,60 / 2,05), sobre a simheurística certificada, rodadas com --zones 2 3 --sem-base.
for _rot, _cp in (('G05', {'gamma': 0.00675}), ('G2', {'gamma': 0.027}),
                  ('KM05', {'m_L': 1.30, 'm_U': 1.025}), ('KM2', {'m_L': 5.20, 'm_U': 4.10})):
    ALGO_CONFIG[f'C-SIMC-{_rot}'] = {**ALGO_CONFIG['C-SIMC'],
                                     'c_params': {**ALGO_CONFIG['C-SIMC'].get('c_params', {}), **_cp},
                                     'label': f'Caminho C: simheurística + certificação, sensibilidade B2 {_rot}'}
# Calibração dentro de uma busca (17/10/2026, diário Entrada 109; análise de fraquezas A1): o nível do buffer começa
# em 0,90 e, a cada G gerações, o plano mais barato da frente é simulado em M cenários novos; acima de 'hi' o nível
# desce um passo dz em z, abaixo de 'lo' sobe. O arquivo guarda os planos de todos os níveis; certificação final igual.
# A e B são as variantes do desenvolvimento; a escolhida vira C-ACALC (fixada antes do teste).
_ACAL = {'A': {'G': 20, 'M': 200, 'lo': 0.92, 'hi': 0.96, 'dz': 0.30, 'qmin': 'nominal', 'qmax': 0.99},
         'B': {'G': 10, 'M': 200, 'lo': 0.92, 'hi': 0.97, 'dz': 0.25, 'qmin': 'nominal', 'qmax': 0.99}}
for _r, _ba in _ACAL.items():
    ALGO_CONFIG[f'C-ACALC-{_r}'] = {'runner': 'nsga2c', 'mc_samples': 0,
                                   'c_params': {'buffer_q': 0.90, 'mc_cert': 500, 'buffer_adapt': _ba},
                                   'label': f'Caminho C: buffer calibrado dentro da busca ({_r}) + certificação'}
# Todas as variantes do Caminho C passam a usar o decoder Split + busca local
# (p_ls = 0,2). O piloto F3 (Entradas 74, results_c_piloto/) usou o decoder
# first-fit — o c_params gravado em cada _metrics.json registra a diferença.
for _k, _v in ALGO_CONFIG.items():
    if _v['runner'] in ('nsga2c', 'moeadc'):
        _v['c_params'] = {'decoder': 'split', 'p_ls': 0.5, **_v.get('c_params', {})}   # Entrada 87: 0,2 -> 0,5

ALL_FAMILIES = ['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2']
ALL_SIZES    = [25, 50, 100]
ALL_UNCERTS  = ['LOW', 'MEDIUM', 'HIGH']


# ─────────────────────────────────────────────────────────────────────────────
# WORKER FUNCTION (executada em cada processo filho)
# ─────────────────────────────────────────────────────────────────────────────

def _worker(task: dict) -> dict:
    """
    Executa uma única combinação (instância, algoritmo, seed) e salva resultados.
    Retorna dicionário de status para relatório do processo pai.
    """
    inst_path  = task['inst_path']
    algo_name  = task['algo']
    seed       = task['seed']
    pop_size   = task['pop_size']
    n_gen      = task['n_gen']
    out_dir    = task['out_dir']
    mc_samples = task['mc_samples']
    runner     = task['runner']
    # B2 (22/08/2026): use_multizone=False/n_zones=1 são os defaults de
    # run_nsga2()/run_random_search() -- tarefas baseline nem passam essas
    # chaves, e o .get() abaixo reproduz o comportamento de sempre.
    use_multizone = task.get('use_multizone', False)
    n_zones       = task.get('n_zones', 1)
    algo_label    = task.get('algo_label', algo_name)  # rótulo p/ analysis.py
    # Frente 4.3 (29/08/2026, componentes 2-4) -- None quando a tarefa não
    # especifica (todo o baseline + B2 + componente 1): resolvido pro default
    # de run_nsga2() logo abaixo, depois que `ns` é importado (não dá pra
    # referenciar ns.CROSSOVER_RATE aqui em cima, módulo ainda não carregado).
    crossover_rate_task    = task.get('crossover_rate')
    mutation_operator_task = task.get('mutation_operator')
    init_mix_task          = task.get('init_mix')
    # Sensibilidade B2 (29/08/2026) -- mesmo padrão acima, resolvido depois
    # que `ns` é importado (não dá pra referenciar ns.GAMMA_ZONE_COLD aqui).
    gamma_zone_cold_task        = task.get('gamma_zone_cold')
    zone_loading_km_mult_task   = task.get('zone_loading_km_mult')
    zone_unloading_km_mult_task = task.get('zone_unloading_km_mult')
    # Curvas de convergência (06/09/2026, Entrada 52) -- None em toda tarefa
    # que não vem de --fixed-hv-ref, preservando o comportamento de sempre.
    hv_ref_point_task = task.get('hv_ref_point')
    dedup = bool(task.get('dedup'))  # 29/09/2026 -- False = comportamento de sempre
    c_params = task.get('c_params') or {}  # Caminho C (03/10/2026)

    inst_name  = os.path.basename(inst_path).replace('.json', '')
    # result_key vem pronto de build_task_list() (uma única fonte da verdade
    # para o nome do arquivo -- evita a chave de _worker() e a checagem de
    # --resume em build_task_list() divergirem entre si).
    result_key = task.get('result_key', f"{inst_name}__{algo_name}__seed{seed}")

    t0 = time.time()
    try:
        # Importa o módulo localmente no processo filho (evita problemas de pickle)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'nsga2_simheuristic',
            os.path.join(SRC_DIR, 'nsga2_simheuristic.py'),
        )
        ns = importlib.util.module_from_spec(spec)
        sys.modules['nsga2_simheuristic'] = ns
        spec.loader.exec_module(ns)

        with open(inst_path, encoding='utf-8') as f:
            inst = json.load(f)

        # Frente 4.3 (componentes 2-4): resolve os defaults de run_nsga2()
        # explicitamente aqui -- se a tarefa não especificou (None), usa
        # exatamente o mesmo valor que run_nsga2() usaria sozinho, então
        # nenhum run baseline/B2/componente-1 muda de comportamento.
        crossover_rate    = crossover_rate_task    if crossover_rate_task    is not None else ns.CROSSOVER_RATE
        mutation_operator = mutation_operator_task if mutation_operator_task is not None else "mix"
        init_mix          = init_mix_task  # None já é o próprio default de run_nsga2()
        # Sensibilidade B2 (29/08/2026) -- mesma resolução None->default real.
        gamma_zone_cold        = gamma_zone_cold_task        if gamma_zone_cold_task        is not None else ns.GAMMA_ZONE_COLD
        zone_loading_km_mult   = zone_loading_km_mult_task   if zone_loading_km_mult_task   is not None else ns.ZONE_LOADING_COST_KM_MULTIPLE
        zone_unloading_km_mult = zone_unloading_km_mult_task if zone_unloading_km_mult_task is not None else ns.ZONE_UNLOADING_COST_KM_MULTIPLE
        # Curvas de convergência (Entrada 52) -- tupla ou None, repassado como está.
        hv_ref_point = tuple(hv_ref_point_task) if hv_ref_point_task is not None else None

        # Executa o algoritmo
        if runner == 'nsga2':
            pareto, hv_hist = ns.run_nsga2(
                inst,
                pop_size=pop_size,
                n_gen=n_gen,
                mc_samples=mc_samples,
                seed=seed,
                verbose=False,
                use_multizone=use_multizone,
                n_zones=n_zones,
                crossover_rate=crossover_rate,
                mutation_operator=mutation_operator,
                init_mix=init_mix,
                gamma_zone_cold=gamma_zone_cold,
                zone_loading_km_mult=zone_loading_km_mult,
                zone_unloading_km_mult=zone_unloading_km_mult,
                hv_ref_point=hv_ref_point,
                dedup=dedup,
            )
        elif runner == 'nsga2c':
            # Caminho C (03/10/2026) -- módulo separado, não altera o NSGA-II acima.
            if SRC_DIR not in sys.path:
                sys.path.insert(0, SRC_DIR)
            import ccvrp_c
            pareto, hv_hist = ccvrp_c.run_nsga2_c(
                inst, pop_size=pop_size, n_gen=n_gen, seed=seed,
                use_multizone=use_multizone, n_zones=n_zones,
                hv_ref_point=hv_ref_point, **c_params)
        elif runner == 'moeadc':
            if SRC_DIR not in sys.path:
                sys.path.insert(0, SRC_DIR)
            import ccvrp_c
            pareto, hv_hist = ccvrp_c.run_moead_c(
                inst, pop_size=pop_size, n_gen=n_gen, seed=seed,
                use_multizone=use_multizone, n_zones=n_zones,
                hv_ref_point=hv_ref_point, **c_params)
        elif runner == 'moead':
            # Baseline externo (frente 4.1, 23/08/2026) -- ver
            # pesquisa/plano_implementacao_moead.md. pop_size e ignorado
            # dentro de run_moead() (o n real de subproblemas vem da malha
            # de Das-Dennis, MOEAD_H=13 -> 105) -- passado aqui só por
            # uniformidade de assinatura com os outros runners.
            pareto, hv_hist = ns.run_moead(
                inst,
                pop_size=pop_size,
                n_gen=n_gen,
                mc_samples=mc_samples,
                seed=seed,
                verbose=False,
                use_multizone=use_multizone,
                n_zones=n_zones,
                hv_ref_point=hv_ref_point,
            )
        else:  # random search
            pareto, hv_hist = ns.run_random_search(
                inst,
                pop_size=pop_size,
                n_gen=n_gen,
                mc_samples=mc_samples,
                seed=seed,
                verbose=False,
                use_multizone=use_multizone,
                n_zones=n_zones,
                hv_ref_point=hv_ref_point,
            )

        # Métricas
        # NOTA (22/08/2026, diário Entradas 13-15): este HV usa ponto de
        # referência AUTO-REFERENCIADO (max deste front x1.1), calculado
        # isoladamente por worker/processo -- não é comparável entre
        # algoritmos diferentes da mesma instância (cada um teria seu próprio
        # ponto de referência). É só um valor provisório para acompanhar o
        # progresso da execução (usado no log abaixo). Para a métrica final,
        # comparável entre algoritmos, gere os resultados primeiro e rode
        # pesquisa/reprocessar_hipervolume.py depois -- ele calcula um único
        # ponto de referência por instância (compartilhado entre os 4
        # algoritmos e as 5 sementes), reproduzindo a convenção usada nos
        # resultados originais (campo 'ref_point_global', ver diário Entrada 13).
        # Curvas de convergência (Entrada 52): com hv_ref_point setado, o HV
        # final usa a MESMA régua fixa dos checkpoints -- 'hypervolume' aqui
        # bate exatamente com o último ponto de {result_key}_hv.csv, e com o
        # 'hypervolume_corrected' já publicado em results/ pra essa mesma
        # (instância, algoritmo, seed), já que é a mesma fórmula de ponto de
        # referência. Sem hv_ref_point (None): comportamento de sempre,
        # auto-referenciado, idêntico a antes desta mudança existir.
        hv     = ns.compute_hypervolume(pareto, ref_point=hv_ref_point)
        spread = ns.compute_spread(pareto)
        gd     = ns.compute_gd(pareto, pareto)

        metrics = {
            'instance':          inst_name,
            'family':            inst.get('family', ''),
            'size':              inst.get('size', 0),
            'uncertainty':       inst.get('uncertainty_level', ''),
            'sigma_T_C':         inst['thermal_profile']['sigma_T_C'],
            # 'algorithm' usa o rótulo (com sufixo +B2zN quando aplicável) --
            # é essa string que analysis.py agrupa/compara, então baseline e
            # B2 nunca se misturam na mesma média/std por engano (22/08/2026).
            'algorithm':         algo_label,
            'algorithm_base':    algo_name,     # sempre o nome "puro" (p/ referência)
            'use_multizone':     use_multizone,
            'n_zones':           n_zones if use_multizone else None,
            'seed':              seed,
            'pop_size':          pop_size,
            'n_gen':             n_gen,
            'mc_samples':        mc_samples,
            'crossover_rate':    crossover_rate    if runner == 'nsga2' else None,
            'mutation_operator': mutation_operator if runner == 'nsga2' else None,
            'init_mix':          list(init_mix) if (runner == 'nsga2' and init_mix is not None) else None,
            'gamma_zone_cold':        gamma_zone_cold        if (runner == 'nsga2' and use_multizone) else None,
            'zone_loading_km_mult':   zone_loading_km_mult   if (runner == 'nsga2' and use_multizone) else None,
            'zone_unloading_km_mult': zone_unloading_km_mult if (runner == 'nsga2' and use_multizone) else None,
            # Curvas de convergência (Entrada 52) -- registrado pra rastreabilidade;
            # None em qualquer execução fora do modo --fixed-hv-ref.
            'hv_ref_point':      list(hv_ref_point) if hv_ref_point is not None else None,
            'dedup':             dedup,
            'c_params':          c_params or None,
            # Entrada 83: cinética do compartimento único (B2 sempre usa a de cada produto)
            'cinetica_por_cliente': (True if use_multizone else
                                     (c_params.get('cinetica_por_cliente')
                                      if c_params.get('cinetica_por_cliente') is not None
                                      else sys.modules['ccvrp_c'].CINETICA_POR_CLIENTE))
                                    if runner in ('nsga2c', 'moeadc') else None,
            'busca_local_delta': sys.modules['ccvrp_c'].BUSCA_LOCAL_DELTA if runner in ('nsga2c', 'moeadc') else None,
            # Entrada 85: eliminação de rotas na busca local
            'eliminar_rotas':    sys.modules['ccvrp_c'].ELIMINAR_ROTAS if runner in ('nsga2c', 'moeadc') else None,
            # Entrada 109: trajetória do nível do buffer na calibração dentro da busca (só nessas execuções)
            **({'q_hist': sys.modules['ccvrp_c'].ULTIMO_Q_HIST} if c_params.get('buffer_adapt') else {}),
            'n_cumprem_chance':  (sum(1 for x in pareto if getattr(x, 'viol', 0.0) == 0.0)
                                  if c_params.get('mc_rel') else None),
            'elapsed_s':         round(time.time() - t0, 2),
            'pareto_size':       len(pareto),
            'n_feasible':        sum(1 for x in pareto if x.feasible),
            'hypervolume':       round(hv, 8),
            'spread':            round(spread, 6),
            'gd':                round(gd, 6),
            'best_cost_BRL':     round(min(x.f_cost  for x in pareto), 2) if pareto else None,
            'best_quality_pct':  round((1 - min(x.f_qloss for x in pareto)) * 100, 3) if pareto else None,
            'best_co2_kg':       round(min(x.f_co2   for x in pareto), 4) if pareto else None,
        }

        # Salva resultados
        os.makedirs(out_dir, exist_ok=True)
        prefix = os.path.join(out_dir, f"{result_key}")

        # Pareto front CSV
        import csv
        with open(f"{prefix}_pareto.csv", 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['sol_id', 'f1_cost_BRL', 'f2_quality_loss',
                        'f3_co2_kg', 'n_routes', 'feasible'])
            for i, ind in enumerate(sorted(pareto, key=lambda x: x.f_cost)):
                w.writerow([i+1, round(ind.f_cost,2), round(ind.f_qloss,6),
                             round(ind.f_co2,4), len(ind.routes), ind.feasible])

        # Soluções da frente final (29/09/2026, reavaliação fora da amostra --
        # pesquisa/plano_reavaliacao_oos.md). Rotas + cromossomo + bloco de
        # zonas, na MESMA ordem do _pareto.csv (sol_id = índice+1), para que
        # pesquisa/analisar_oos.py possa reavaliar f2 num conjunto comum de
        # cenários, independente dos usados na busca. Só grava; não altera
        # nenhum valor já calculado acima.
        with open(f"{prefix}_solutions.json", 'w', encoding='utf-8') as fh:
            json.dump([
                {'sol_id': i + 1,
                 'routes': ind.routes,
                 'chromosome': ind.chromosome,
                 'zone_block': ind.zone_block,
                 'feasible': ind.feasible,
                 'f1_cost_BRL': ind.f_cost,
                 'f2_quality_loss_insample': ind.f_qloss,
                 'f3_co2_kg': ind.f_co2,
                 # Caminho C (03/10/2026): ausentes nos outros runners
                 **({'p_on_insample': ind.p_on, 'viol': ind.viol,
                     'dep_block': ind.dep_block[:len(ind.routes)],
                     'partida_min': list(ind.partida),
                     'p_cert': ind.p_cert} if hasattr(ind, 'viol') else {})}
                for i, ind in enumerate(sorted(pareto, key=lambda x: x.f_cost))
            ], fh)

        # HV history CSV
        with open(f"{prefix}_hv.csv", 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['generation', 'hypervolume'])
            for gen, hv_val in hv_hist:
                w.writerow([gen, round(hv_val, 8)])

        # Metrics JSON
        with open(f"{prefix}_metrics.json", 'w', encoding='utf-8') as fh:
            json.dump(metrics, fh, indent=2, ensure_ascii=False)

        return {'key': result_key, 'status': 'ok', 'elapsed': metrics['elapsed_s'],
                'hv': hv, 'pareto_size': len(pareto)}

    except Exception as e:
        tb = traceback.format_exc()
        err_path = os.path.join(out_dir, f"{result_key}_ERROR.txt")
        os.makedirs(out_dir, exist_ok=True)
        with open(err_path, 'w') as fh:
            fh.write(tb)
        return {'key': result_key, 'status': 'error', 'error': str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# PONTO DE REFERÊNCIA FIXO POR INSTÂNCIA (curvas de convergência, Entrada 52)
# ─────────────────────────────────────────────────────────────────────────────

def _load_pareto_points(pareto_csv_path: str) -> np.ndarray:
    """Mesma leitura de pesquisa/reprocessar_hipervolume.py:carregar_pontos()
    -- duplicada aqui (não importada) pra não acoplar os dois scripts; a
    fórmula do ponto de referência abaixo é a mesma e é conferida contra os
    valores já publicados no teste de pequena escala antes do lote completo."""
    with open(pareto_csv_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return np.empty((0, 3))
    return np.array([
        [float(r['f1_cost_BRL']), float(r['f2_quality_loss']), float(r['f3_co2_kg'])]
        for r in rows
    ])


def compute_ref_points_global(
    algos:       set,
    results_dir: str = RESULTS_DIR,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Ponto de referência FIXO por instância = max(todos os pontos de Pareto de
    TODAS as execuções já existentes em `results_dir`, restrito a `algos`,
    todas as sementes) × 1.1 + 1e-9 -- mesma fórmula de
    pesquisa/reprocessar_hipervolume.py (usada lá pro campo
    'hypervolume_corrected' já publicado na Tabela 6 do manuscrito).

    Não roda nenhuma otimização nova -- só lê os *_pareto.csv que essas
    execuções (já concluídas) deixaram salvos. Barato (54 instâncias × poucas
    dezenas de pontos cada).
    """
    todos = sorted(glob.glob(os.path.join(results_dir, '**', '*_pareto.csv'), recursive=True))
    arquivos = [
        f for f in todos
        if os.path.basename(os.path.dirname(f)) in algos
    ]
    por_instancia: Dict[str, list] = {}
    for f in arquivos:
        # nome de arquivo: {inst_name}__{algo}__seed{N}_pareto.csv
        base = os.path.basename(f)
        inst_name = base.split('__')[0]
        por_instancia.setdefault(inst_name, []).append(f)

    ref_points: Dict[str, Tuple[float, float, float]] = {}
    for inst_name, arqs in por_instancia.items():
        pontos = [_load_pareto_points(a) for a in arqs]
        pontos = [p for p in pontos if len(p) > 0]
        if not pontos:
            continue
        todos_pontos = np.vstack(pontos)
        ref = todos_pontos.max(axis=0) * 1.1 + 1e-9
        ref_points[inst_name] = tuple(float(x) for x in ref)
    return ref_points


# ─────────────────────────────────────────────────────────────────────────────
# MONTAGEM DO PLANO DE EXPERIMENTOS
# ─────────────────────────────────────────────────────────────────────────────

def build_task_list(
    families:   List[str],
    sizes:      List[int],
    algorithms: List[str],
    n_seeds:    int,
    pop_size:   int,
    n_gen:      int,
    resume:     bool,
    zones:       Optional[List[int]] = None,
    uncertainties: Optional[List[str]] = None,
    results_dir: str = RESULTS_DIR,
    ref_points:  Optional[Dict[str, Tuple[float, float, float]]] = None,
    results_b2_dir: str = RESULTS_B2_DIR,
    sem_base: bool = False,
) -> List[dict]:
    """
    Constrói lista de tarefas (instância × algoritmo × seed).

    zones (B2, 22/08/2026, opcional): para cada combinação acima, além da
    tarefa baseline de sempre (inalterada -- mesma chave, mesma pasta,
    use_multizone=False), adiciona uma tarefa extra por valor de Z em `zones`
    (use_multizone=True, n_zones=Z), com rótulo de algoritmo "{algo}+B2z{Z}"
    e saída em results_b2/{algo}_Z{Z}/ -- nunca no lugar do baseline.
    Default (zones=None ou []): comportamento idêntico a antes de B2 existir.

    results_dir (curvas de convergência, Entrada 52, opcional): base pra
    onde vão as tarefas baseline -- default RESULTS_DIR preserva 100% do
    comportamento de sempre. --fixed-hv-ref passa RESULTS_CONV_DIR aqui, pra
    o próprio check de --resume (existência de _metrics.json) já olhar pro
    lugar certo, em vez de achar os _metrics.json antigos em results/ e pular
    tudo por engano.

    ref_points (idem, opcional): dict {inst_name: (f1,f2,f3)} -- quando dado,
    cada tarefa baseline ganha 'hv_ref_point' com o valor da própria
    instância (None se a instância não tiver entrada, ex. B2 experimental).
    Default None preserva o comportamento de sempre (hv_ref_point=None em
    toda tarefa, auto-referenciado).
    """
    zones = zones or []
    # Frente 4.3 (28/08/2026): ALL_UNCERTS já existia como constante mas
    # nunca tinha sido ligada a um filtro de CLI -- `unc` era extraído do
    # nome do arquivo (linha abaixo) e nunca comparado a nada. Default
    # None/[] = todas (comportamento idêntico a antes, nenhuma chamada
    # existente quebra).
    uncertainties = uncertainties or ALL_UNCERTS
    tasks = []
    inst_files = sorted(glob.glob(os.path.join(INST_DIR, '*.json')))
    inst_files = [f for f in inst_files if 'catalog' not in os.path.basename(f)]

    for fpath in inst_files:
        name  = os.path.basename(fpath).replace('.json', '')
        parts = name.split('_')
        fam   = parts[1]           # C1, R2, etc.
        sz    = int(parts[2])      # 025 → 25
        unc   = parts[3]           # LOW/MEDIUM/HIGH

        if fam not in families:
            continue
        if sz not in sizes:
            continue
        if unc not in uncertainties:
            continue

        for algo in algorithms:
            cfg = ALGO_CONFIG[algo]
            mc  = cfg['mc_samples']
            # Frente 4.3 (componentes 2-4, 29/08/2026): só as entradas novas de
            # ALGO_CONFIG que precisarem definem estas chaves; ausentes em
            # todo o resto (baseline, B2, componente 1) -- cfg.get() devolve
            # None, e _worker() resolve None para o default de run_nsga2().
            cx_rate  = cfg.get('crossover_rate')
            mut_op   = cfg.get('mutation_operator')
            imix     = cfg.get('init_mix')
            # Sensibilidade B2 (29/08/2026) -- mesmo padrão: None quando a
            # entrada de ALGO_CONFIG não define, _worker() resolve pro default
            # real de run_nsga2()/evaluate() (GAMMA_ZONE_COLD/ZONE_*_KM_MULTIPLE).
            gamma_cold = cfg.get('gamma_zone_cold')
            km_load    = cfg.get('zone_loading_km_mult')
            km_unload  = cfg.get('zone_unloading_km_mult')
            dedup      = cfg.get('dedup', False)
            c_params   = cfg.get('c_params')

            for seed in range(n_seeds):
                # --- tarefa baseline (inalterada, exceto results_dir/ref_points
                #     opcionais -- ambos default = comportamento de sempre) ---
                out_dir    = os.path.join(results_dir, algo)
                result_key = f"{name}__{algo}__seed{seed}"
                hv_ref_point = ref_points.get(name) if ref_points else None
                if not sem_base and not (resume and os.path.exists(os.path.join(out_dir, f"{result_key}_metrics.json"))):
                    tasks.append({
                        'inst_path':  fpath,
                        'algo':       algo,
                        'algo_label': algo,
                        'seed':       seed,
                        'pop_size':   pop_size,
                        'n_gen':      n_gen,
                        'mc_samples': mc,
                        'runner':     cfg['runner'],
                        'out_dir':    out_dir,
                        'result_key': result_key,
                        'use_multizone': False,
                        'n_zones':       1,
                        'crossover_rate':    cx_rate,
                        'mutation_operator': mut_op,
                        'init_mix':          imix,
                        'gamma_zone_cold':        gamma_cold,
                        'zone_loading_km_mult':   km_load,
                        'zone_unloading_km_mult': km_unload,
                        'hv_ref_point':           hv_ref_point,
                        'dedup':                  dedup,
                        'c_params':               c_params,
                    })

                # --- tarefas B2 (uma por Z pedido, adicional ao baseline) ---
                for z in zones:
                    algo_label   = f"{algo}+B2z{z}"
                    out_dir_b2   = os.path.join(results_b2_dir, f"{algo}_Z{z}")
                    result_key_b2 = f"{name}__{algo_label}__seed{seed}"
                    if resume and os.path.exists(os.path.join(out_dir_b2, f"{result_key_b2}_metrics.json")):
                        continue
                    tasks.append({
                        'inst_path':  fpath,
                        'algo':       algo,
                        'algo_label': algo_label,
                        'seed':       seed,
                        'pop_size':   pop_size,
                        'n_gen':      n_gen,
                        'mc_samples': mc,
                        'runner':     cfg['runner'],
                        'out_dir':    out_dir_b2,
                        'result_key': result_key_b2,
                        'use_multizone': True,
                        'n_zones':       z,
                        'crossover_rate':    cx_rate,
                        'mutation_operator': mut_op,
                        'init_mix':          imix,
                        'gamma_zone_cold':        gamma_cold,
                        'zone_loading_km_mult':   km_load,
                        'zone_unloading_km_mult': km_unload,
                        'dedup':                  dedup,
                        'c_params':               c_params,
                    })
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark CCVRPTW-ST — execução paralela',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--workers',    type=int,   default=DEFAULT_WORKERS,
                        help=f'Número de processos paralelos (padrão: {DEFAULT_WORKERS})')
    parser.add_argument('--seeds',      type=int,   default=DEFAULT_SEEDS,
                        help=f'Sementes por combinação (padrão: {DEFAULT_SEEDS})')
    parser.add_argument('--pop-size',   type=int,   default=DEFAULT_POP_SIZE,
                        help=f'Tamanho da população (padrão: {DEFAULT_POP_SIZE})')
    parser.add_argument('--n-gen',      type=int,   default=DEFAULT_N_GEN,
                        help=f'Número de gerações (padrão: {DEFAULT_N_GEN})')
    parser.add_argument('--families',   nargs='+',  default=ALL_FAMILIES,
                        choices=ALL_FAMILIES, metavar='FAM',
                        help='Famílias a executar (padrão: todas)')
    parser.add_argument('--sizes',      nargs='+',  type=int, default=ALL_SIZES,
                        choices=ALL_SIZES, metavar='N',
                        help='Tamanhos de instância (padrão: 25 50 100)')
    parser.add_argument('--uncertainty', nargs='+', default=ALL_UNCERTS,
                        choices=ALL_UNCERTS, metavar='UNC',
                        help='Níveis de incerteza térmica a executar (padrão: todos) -- '
                             'novo na frente 4.3 (28/08/2026), ALL_UNCERTS já existia '
                             'mas não estava ligado a nenhum filtro de CLI antes.')
    parser.add_argument('--algorithms', nargs='+',  default=None,
                        choices=list(ALGO_CONFIG.keys()), metavar='ALGO',
                        help='Algoritmos a executar (padrão: todos — ver aviso impresso '
                             'se omitido, ALGO_CONFIG já tem %d entradas, Entrada 43)' % len(ALGO_CONFIG))
    parser.add_argument('--resume',     action='store_true',
                        help='Pula execuções com resultado já salvo')
    parser.add_argument('--sem-base',   action='store_true',
                        help='Entrada 87: com --zones, NÃO roda as tarefas de compartimento único (só as de zonas)')
    parser.add_argument('--quick',      action='store_true',
                        help='Modo rápido: n=25, 2 seeds, 50 gerações, só NSGA2-MC e NSGA2-DET')
    parser.add_argument('--zones',      nargs='+', type=int, default=[],
                        choices=[2, 3], metavar='Z',
                        help='B2 (22/08/2026): além da grade baseline de sempre, roda '
                             'também com use_multizone=True para cada Z pedido (2 e/ou 3). '
                             'Vai para results_b2/, não mexe no baseline. Padrão: nenhum '
                             '(comportamento idêntico a antes de B2 existir).')
    parser.add_argument('--results-dir', type=str, default=None, metavar='DIR',
                        help='Auditoria de revisor (07/09/2026, diário Entrada 57): '
                             'redireciona a saída para um diretório customizado, em vez '
                             'do results/ padrão -- uso pretendido: reexecutar a '
                             'comparação central inteira numa única sessão/ambiente, '
                             'gravando num diretório novo (ex. results_single_env/) sem '
                             'misturar com o results/ heterogêneo já existente. Independente '
                             'de --fixed-hv-ref (que já tinha sua própria lógica de '
                             'redirecionamento para RESULTS_CONV_DIR) -- se ambos forem '
                             'passados, --results-dir tem precedência e um aviso é impresso.')
    parser.add_argument('--results-b2-dir', type=str, default=None, metavar='DIR',
                        help='Reavaliação fora da amostra (29/09/2026): redireciona a '
                             'saída das tarefas B2 (--zones) para um diretório customizado, '
                             f'em vez do {RESULTS_B2_DIR} padrão.')
    parser.add_argument('--fixed-hv-ref', action='store_true',
                        help='Curvas de convergência (06/09/2026, diário Entrada 52): usa '
                             'ponto de referência FIXO por instância (mesma fórmula/conjunto '
                             'de algoritmos de pesquisa/reprocessar_hipervolume.py) no HV de '
                             'checkpoint, em vez do auto-referenciado de sempre -- o '
                             'auto-referenciado produz curva não-monotônica (verificado, ver '
                             'diário). Só vale pros 5 algoritmos da comparação central '
                             f'({sorted(ALGOS_COMPARACAO_CENTRAL)}); grava em {RESULTS_CONV_DIR} '
                             '(nunca em results/ ou results_b2/). Requer que results/ já tenha '
                             'os _pareto.csv desses 5 algoritmos pra essas instâncias (usa pra '
                             'calcular o ponto de referência -- não roda nada novo pra isso).')
    args = parser.parse_args()

    # Modo rápido (define seu próprio --algorithms explícito, não precisa do aviso abaixo)
    if args.quick:
        args.sizes      = [25]
        args.seeds      = 2
        args.n_gen      = 50
        args.algorithms = ['NSGA2-MC', 'NSGA2-DET']
        print("⚡ Modo rápido: n=25, 2 seeds, 50 gerações, NSGA2-MC + NSGA2-DET")

    # --algorithms omitido (e não é --quick): roda TODOS os algoritmos de
    # ALGO_CONFIG (padrão de sempre), mas agora avisa explicitamente quais são
    # -- ALGO_CONFIG cresceu de 4 pra 16 entradas ao longo do projeto (efeito
    # colateral por comando sem --algorithms explícito já aconteceu uma vez,
    # Entrada 33; pendência registrada desde então, fechada em 29/08/2026,
    # Entrada 43).
    if args.algorithms is None:
        args.algorithms = list(ALGO_CONFIG.keys())
        print(f"⚠ --algorithms não informado -- rodando TODOS os {len(args.algorithms)} "
              f"algoritmos de ALGO_CONFIG: {', '.join(args.algorithms)}")
        print("  (inclui as variantes de ablação da frente 4.3 -- passe --algorithms "
              "explicitamente pra evitar rodar mais do que pretende)")

    # Curvas de convergência (Entrada 52): valida, calcula os pontos de
    # referência fixos (a partir do que já existe em results/, sem rodar
    # nada novo) e redireciona a saída pra results_conv/ -- nunca em cima de
    # results/ ou results_b2/, mesmo que o --resume ache um _metrics.json
    # antigo (por isso o results_dir precisa ir pro build_task_list()
    # ANTES do check de --resume, não depois).
    results_dir_effective = RESULTS_DIR
    ref_points = None
    if args.fixed_hv_ref:
        algos_fora = set(args.algorithms) - ALGOS_COMPARACAO_CENTRAL
        if algos_fora:
            print(f"✗ --fixed-hv-ref só vale pros {len(ALGOS_COMPARACAO_CENTRAL)} algoritmos "
                  f"da comparação central: {sorted(ALGOS_COMPARACAO_CENTRAL)}")
            print(f"  Algoritmo(s) fora desse conjunto pedido(s): {sorted(algos_fora)}")
            print("  Passe --algorithms explicitamente com só os 5 centrais.")
            return
        print("Calculando ponto de referência fixo por instância "
              f"(a partir dos _pareto.csv já existentes em {RESULTS_DIR}, "
              f"algoritmos {sorted(ALGOS_COMPARACAO_CENTRAL)})...")
        ref_points = compute_ref_points_global(ALGOS_COMPARACAO_CENTRAL, RESULTS_DIR)
        print(f"  {len(ref_points)} instâncias com ponto de referência calculado.")
        instancias_pedidas = {
            os.path.basename(f).replace('.json', '')
            for f in sorted(glob.glob(os.path.join(INST_DIR, '*.json')))
            if 'catalog' not in os.path.basename(f)
        }
        faltando = instancias_pedidas - set(ref_points.keys())
        if faltando:
            print(f"  ⚠ {len(faltando)} instância(s) SEM ponto de referência "
                  f"(sem _pareto.csv prévio dos 5 algoritmos centrais): {sorted(faltando)}")
            print("    Essas tarefas rodarão com hv_ref_point=None (comportamento antigo, "
                  "curva não-monotônica) -- confira antes de prosseguir.")
        results_dir_effective = RESULTS_CONV_DIR

    if args.results_dir:
        if args.fixed_hv_ref:
            print(f"⚠ --results-dir e --fixed-hv-ref ambos passados -- "
                  f"--results-dir tem precedência, gravando em {args.results_dir} "
                  f"(não em {RESULTS_CONV_DIR}); ref_points já calculados acima continuam válidos.")
        results_dir_effective = args.results_dir
        os.makedirs(results_dir_effective, exist_ok=True)

    tasks = build_task_list(
        families   = args.families,
        sizes      = args.sizes,
        algorithms = args.algorithms,
        n_seeds    = args.seeds,
        pop_size   = args.pop_size,
        n_gen      = args.n_gen,
        resume     = args.resume,
        zones      = args.zones,
        uncertainties = args.uncertainty,
        results_dir = results_dir_effective,
        ref_points  = ref_points,
        results_b2_dir = args.results_b2_dir or RESULTS_B2_DIR,
        sem_base   = args.sem_base,
    )

    n_total = len(tasks)
    if n_total == 0:
        print("Nenhuma tarefa pendente. Use --resume=False para forçar reexecução.")
        return

    workers = min(args.workers, n_total, cpu_count())

    print(f"\n{'='*65}")
    print(f"  BENCHMARK CCVRPTW-ST" + ("  (curvas de convergência, ref. fixo)" if args.fixed_hv_ref else ""))
    print(f"{'='*65}")
    print(f"  Tarefas    : {n_total}")
    print(f"  Algoritmos : {', '.join(args.algorithms)}")
    print(f"  Famílias   : {', '.join(args.families)}")
    print(f"  Tamanhos   : {args.sizes}")
    print(f"  Seeds      : {args.seeds}")
    print(f"  Pop / Gen  : {args.pop_size} / {args.n_gen}")
    print(f"  Workers    : {workers}")
    print(f"  Resultados : {results_dir_effective}")
    if args.zones:
        print(f"  B2 (zonas) : {args.zones}  → {args.results_b2_dir or RESULTS_B2_DIR} (adicional ao baseline acima)")
    print(f"{'='*65}\n")

    t_start = time.time()
    done = 0
    errors = []

    with Pool(processes=workers) as pool:
        for result in pool.imap_unordered(_worker, tasks, chunksize=1):
            done += 1
            elapsed_total = time.time() - t_start
            avg_s = elapsed_total / done
            eta_s = avg_s * (n_total - done)

            if result['status'] == 'ok':
                status_str = f"HV={result['hv']:.3e}  Pareto={result['pareto_size']}"
            else:
                status_str = f"ERRO: {result['error'][:50]}"
                errors.append(result)

            print(f"  [{done:4d}/{n_total}] {result['key'][:55]:<55}"
                  f" | {result.get('elapsed', 0):.1f}s"
                  f" | ETA {eta_s/60:.1f}min"
                  f" | {status_str}", flush=True)

    elapsed_total = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"  Concluído em {elapsed_total/60:.1f} min")
    print(f"  Sucesso: {done - len(errors)}/{done}")
    if errors:
        print(f"  Erros ({len(errors)}): ver arquivos *_ERROR.txt em {results_dir_effective}")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
