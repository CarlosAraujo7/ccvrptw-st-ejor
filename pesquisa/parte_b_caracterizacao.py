#!/usr/bin/env python3
"""
Parte B do Caminho C — onde a incerteza importa (04/10/2026, diário Entrada 75;
pesquisa/plano_caminho_C.md §4).

Avalia frentes JÁ OTIMIZADAS (rotas + saídas do depósito, *_solutions.json do
Caminho C) sob condições físicas e operacionais diferentes, sem otimizar de
novo. Um fator por vez em torno do caso-base, mais o cruzamento clima × horário.

Caso-base: São Paulo (NOAA ISD), saída às 06h, porta 0,8 s/kg (Stellingwerf et
al. 2021), CV da porta por nível (Novaes et al. 2015), inércia térmica do
produto τp = 10 h, σ_T e CV do tempo de viagem da instância.

Para cada condição e instância, sobre as soluções distintas das frentes:
  Q  — a qualidade NOMINAL basta para decidir? Spearman nominal × esperada,
       teto de ruído (duas amostras independentes) e arrependimento por
       orçamento de custo (escolha na amostra 1, avaliação na amostra 2).
  QB — probabilidade de quebra da cadeia de frio: fração esperada do valor
       entregue com temperatura máxima do produto acima do limite legal da
       categoria do SET-POINT do veículo (Portaria CVS 3/2026, Arts. 30 e 109; os
       mesmos valores da CVS 5/2013). Mede se as excursões da operação (porta,
       resfriamento) ultrapassam o limite da classe do veículo; o descasamento de
       categorias no compartimento único é tratado pelo B2. Também o
       arrependimento de escolher sem considerá-la.
  Cinética: a da carga homogênea do experimento principal (cc.CINETICA_POR_CLIENTE = False,
  Entrada 87), igual à busca das frentes avaliadas.
  P  — pontualidade P(nenhum atraso > δ) média das soluções.
Saída: pesquisa/resultados_parte_b/parte_b.csv + resumo no terminal.
Uso: python pesquisa\\parte_b_caracterizacao.py --dirs results_c --algs C-DET C-BUF90C C-SIMC
"""
import argparse, csv, glob, json, os, sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(BASE, 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nsga2_simheuristic as ns   # noqa: E402
import thermal_route as tr        # noqa: E402
import ccvrp_c as cc              # noqa: E402
from teste_corte_risco import arrependimento, LIMITE_BR   # noqa: E402

ESTACOES = {'São Paulo': '83780099999', 'Fortaleza': '82398099999', 'Manaus': '82111099999',
            'Recife': '82899099999', 'Rio de Janeiro': '83746099999', 'Porto Alegre': '83971099999',
            'Curitiba': '83840099999'}
BASE_COND = dict(estacao='São Paulo', t_start=6.0, tau=0.8, cv_porta=None, tau_p=10.0,
                 escala_sigmaT=1.0, cv_t=None, ruido_amb=False)


def condicoes():
    out = [('base', {})]
    for e in ESTACOES:
        if e != 'São Paulo':
            out.append((f'clima={e}', {'estacao': e}))
    out += [('inicio=19h', {'t_start': 19.0})]
    for e in ('Fortaleza', 'Manaus', 'Porto Alegre'):
        out.append((f'clima={e}+19h', {'estacao': e, 't_start': 19.0}))
    out += [(f'porta={t}s/kg', {'tau': t}) for t in (0.4, 1.2)]
    out += [(f'cv_porta={c}', {'cv_porta': c}) for c in (0.05, 0.60)]
    out += [(f'tau_p={t}h', {'tau_p': t}) for t in (0.25, 1.0, 4.0)]
    out += [(f'sigmaT×{k}', {'escala_sigmaT': k}) for k in (0.0, 2.0)]
    out += [(f'cv_t={c}', {'cv_t': c}) for c in (0.05, 0.30)]
    # Entrada 84: ruído do termostato DEPENDENTE DA DECISÃO — viola de propósito a
    # hipótese da Proposição 1 (ver ruido_dependente abaixo)
    out += [('sigmaT~amb', {'ruido_amb': True})]
    return out


def ruido_dependente(sol, inst, ttime, sc, cond):
    """Cópia do cenário com o ruído do termostato de cada cliente multiplicado por
    g = max(0, 1 + 2·(T_amb − T̄)/ΔT), onde T_amb é a temperatura ambiente DO PRÓPRIO
    cenário na hora planejada de início do serviço (tempos nominais, saída escolhida
    pela solução), e T̄ e ΔT são a média e a amplitude do perfil nominal. O ruído vai de
    zero na hora mais fria ao dobro na mais quente e passa a depender da sequência e da
    saída do depósito, isto é, da decisão."""
    if not cond.get('ruido_amb') or not np.any(sc['eps']):
        return sc
    est = ESTACOES[cond['estacao']]
    perfil = tr.cenario_nominal(inst, estacao=est)['amb'][0]
    Tm, dT = perfil.mean(), perfil.max() - perfil.min()
    cust = inst['customers']; eps = sc['eps'].copy(); K = eps.shape[0]
    for r, d0 in zip(sol['routes'], sol['partida_min']):
        t = d0; prev = 0
        for c in r:
            t += ttime[prev][c]
            t = max(t, cust[c - 1]['ready_time_min'])
            hora = np.full((K, 1), cond['t_start'] + t / 60.0)
            Ta = tr._amb_at(sc['amb'], hora)[:, 0]
            g = np.maximum(0.0, 1.0 + 2.0 * (Ta - Tm) / dT)
            eps[:, c, :] *= g[:, None]
            t += cust[c - 1]['service_time_min']; prev = c
    return {**sc, 'eps': eps}


def preparar(inst, cond):
    """Aplica a condição (cópia da instância + parâmetros de módulo)."""
    inst = json.loads(json.dumps(inst))
    tp = inst['thermal_profile']
    tp['sigma_T_C'] = tp['sigma_T_C'] * cond['escala_sigmaT']
    for c in tp.get('per_category', {}).values():
        c['sigma_T_C'] = c['sigma_T_C'] * cond['escala_sigmaT']
    if cond['cv_t'] is not None:
        tp['sigma_t_pct'] = cond['cv_t']
    tr.TAU_UNLOAD_S_PER_KG = cond['tau']
    if cond['cv_porta'] is not None:
        tr.DOOR_CV = {k: cond['cv_porta'] for k in ('LOW', 'MEDIUM', 'HIGH')}
    else:
        tr.DOOR_CV = {'LOW': 0.10, 'MEDIUM': 0.30, 'HIGH': 0.60}
    return inst


def avaliar(sol, inst, ttime, sc, cond, lim):
    cust = inst['customers']; acc = 0.0; tot = 0.0; viol = 0.0; on = None
    est = ESTACOES[cond['estacao']]
    sc = ruido_dependente(sol, inst, ttime, sc, cond)
    for r, d0 in zip(sol['routes'], sol['partida_min']):
        d = tr.route_quality(r, inst, ttime, sc, t_start_h=cond['t_start'], tau_p_h=cond['tau_p'],
                             per_customer_kinetics=cc.CINETICA_POR_CLIENTE, detalhe=True, t0_min=d0)
        v = d['values']
        acc = acc + (d['Q'] * v[None, :]).sum(1); tot += v.sum()
        viol = viol + ((d['Tp_max'] > lim) * v[None, :]).sum(1)
        if 'tt' in sc:
            o = (d['late'] <= cc.DELTA_MIN).all(1)
            on = o if on is None else (on & o)
    return 1 - acc / tot, viol / tot, (on.astype(float) if on is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', nargs='+', required=True)
    ap.add_argument('--algs', nargs='+', default=['C-DET', 'C-BUF90C', 'C-SIMC'])
    ap.add_argument('--tamanho', default='050')
    ap.add_argument('--K', type=int, default=200)
    ap.add_argument('--max-sol', type=int, default=150, help='máx. de soluções distintas por instância')
    ap.add_argument('--out', default=os.path.join(BASE, 'pesquisa', 'resultados_parte_b'))
    a = ap.parse_args()
    dirs = [d if os.path.isabs(d) else os.path.join(BASE, d) for d in a.dirs]
    sols = defaultdict(dict)
    for d in dirs:
        for alg in a.algs:
            for f in glob.glob(os.path.join(d, alg, f'*_{a.tamanho}_*_solutions.json')):
                nome = os.path.basename(f).split('__')[0]
                for s in json.load(open(f, encoding='utf-8')):
                    if s['feasible'] and 'partida_min' in s:
                        sols[nome][json.dumps([s['routes'], [round(x, 3) for x in s['partida_min']]])] = s
    print(f"{len(sols)} instâncias; soluções distintas por instância: "
          f"{int(np.median([len(v) for v in sols.values()])) if sols else 0} (mediana)", flush=True)
    os.makedirs(a.out, exist_ok=True)
    linhas = []
    # mesmo subconjunto de soluções em TODAS as condições (semente por instância)
    selecao = {}
    for nome, S in sols.items():
        S = list(S.values())
        if len(S) > a.max_sol:
            r = np.random.default_rng(sum(map(ord, nome)))
            S = [S[i] for i in sorted(r.choice(len(S), a.max_sol, replace=False))]
        selecao[nome] = S
    for rot, delta in condicoes():
        cond = {**BASE_COND, **delta}
        for nome, S in sorted(selecao.items()):
            inst = preparar(ns.load_instance(os.path.join(BASE, 'instances', 'generated', nome + '.json')), cond)
            _, ttime = ns.build_matrices(inst)
            lim = LIMITE_BR[inst['thermal_profile'].get('dominant_category', 'meat')]
            est = ESTACOES[cond['estacao']]
            nom = tr.cenario_nominal(inst, estacao=est)
            s1 = tr.sortear_cenarios(inst, a.K, np.random.default_rng(31), estacao=est, tempo_viagem=True)
            s2 = tr.sortear_cenarios(inst, a.K, np.random.default_rng(32), estacao=est, tempo_viagem=True)
            # tempo de viagem com gerador próprio: idêntico entre condições (números aleatórios comuns)
            n_c = len(inst['customers']); cvt = inst['thermal_profile'].get('sigma_t_pct', 0.0)
            s1['tt'] = tr._lognormal_media1(np.random.default_rng(41), cvt, (a.K, n_c + 1))
            s2['tt'] = tr._lognormal_media1(np.random.default_rng(42), cvt, (a.K, n_c + 1))
            M = []
            for s in S:
                qn, vn, _ = avaliar(s, inst, ttime, nom, cond, lim)
                q1, v1, _ = avaliar(s, inst, ttime, s1, cond, lim)
                q2, v2, p2 = avaliar(s, inst, ttime, s2, cond, lim)
                M.append((s['f1_cost_BRL'], qn[0], q1.mean(), q2.mean(), vn[0], v1.mean(), v2.mean(), p2.mean()))
            M = np.array(M)
            rho = spearmanr(M[:, 1], M[:, 3]).correlation
            teto = spearmanr(M[:, 2], M[:, 3]).correlation
            arq_abs, arq_rel = arrependimento(M[:, 0], M[:, 1], M[:, 2], M[:, 3])
            qb_abs, _ = arrependimento(M[:, 0], M[:, 4], M[:, 5], M[:, 6])
            linhas.append({'condicao': rot, 'instancia': nome, 'nivel': nome.split('_')[-1], 'n_sol': len(S),
                           'q_esp': M[:, 3].mean(), 'q_nom': M[:, 1].mean(), 'rho_q': rho, 'teto_q': teto,
                           'arrep_q_abs': arq_abs, 'arrep_q_rel': arq_rel,
                           'quebra_esp': M[:, 6].mean(), 'arrep_quebra_abs': qb_abs, 'p_pontual': M[:, 7].mean()})
        sub = [L for L in linhas if L['condicao'] == rot]
        print(f"{rot:24s} q esp {100 * np.mean([L['q_esp'] for L in sub]):6.3f}%  "
              f"ρ {np.nanmedian([L['rho_q'] for L in sub]):.4f} (teto {np.nanmedian([L['teto_q'] for L in sub]):.4f})  "
              f"arrep. q {100 * np.nanmedian([L['arrep_q_abs'] for L in sub]):.4f} p.p.  "
              f"quebra {100 * np.mean([L['quebra_esp'] for L in sub]):6.3f}%  "
              f"arrep. quebra {100 * np.nanmedian([L['arrep_quebra_abs'] for L in sub]):.3f} p.p.  "
              f"P(pontual) {np.mean([L['p_pontual'] for L in sub]):.3f}", flush=True)
    with open(os.path.join(a.out, 'parte_b.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0])); w.writeheader(); w.writerows(linhas)


if __name__ == '__main__':
    main()
