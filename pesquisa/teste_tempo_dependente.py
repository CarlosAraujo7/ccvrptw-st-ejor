#!/usr/bin/env python3
"""
Testes dos tempos dependentes do horário (16/10/2026, diário Entrada 101).

  1. Propriedades do modelo de Ichoua, Gendreau e Potvin: com fator constante f, tempo = f·τ;
     chegada contínua e não decrescente na saída (FIFO); travessia de fronteira horária conferida à mão.
  2. Equivalência: com perfil plano (f ≡ 1) e comprimentos = matriz estática, todas as funções de
     cronograma (Δmax, duração, pontualidade por cenário, qualidade térmica, Split, viabilidade, buffer)
     devolvem o mesmo que o código estático, em rotas e cromossomos sorteados de uma instância SP.
Uso: python pesquisa/teste_tempo_dependente.py
"""
import os, random, sys
import numpy as np

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(BASE, 'src')); sys.path.insert(0, os.path.join(BASE, 'pesquisa', 'caso_sp'))
import rodar_caso_sp          # noqa: E402
rodar_caso_sp.aplicar_matrizes()
import nsga2_simheuristic as ns   # noqa: E402
import ccvrp_c as cc              # noqa: E402
import tempo_dependente as td     # noqa: E402
import thermal_route as tr        # noqa: E402

falhas = []


def confere(nome, ok):
    print(('  ok   ' if ok else '  FALHA ') + nome)
    if not ok:
        falhas.append(nome)


def propriedades():
    print('1. propriedades do modelo')
    ff = [[0.0, 30.0], [30.0, 0.0]]
    M = td.MatrizTD(ff, [1.5] * 24, 6.0)
    confere('fator constante 1,5: tempo = 1,5·τ', all(abs(M.chegada(0, 1, t) - t - 45.0) < 1e-9 for t in np.linspace(0, 900, 37)))
    f = [1.0] * 24; f[7] = 2.0                         # 07:00-08:00 com o dobro do tempo
    M = td.MatrizTD(ff, f, 6.0)
    # sai às 06:50 (t = 50): 10 min de fluxo livre até 07:00, faltam 20 min de fluxo livre a taxa 1/2 → 40 min
    confere('travessia da fronteira das 07:00 (saída 06:50 → chegada 07:40)', abs(M.chegada(0, 1, 50.0) - 100.0) < 1e-9)
    # sai às 07:50 (t = 110): 10 min a taxa 1/2 = 5 min de fluxo livre; faltam 25 min em 08:00 → 08:25
    confere('travessia da fronteira das 08:00 (saída 07:50 → chegada 08:25)', abs(M.chegada(0, 1, 110.0) - 145.0) < 1e-9)
    perfil = td.MatrizTD([[0.0, 37.3], [37.3, 0.0]], list(np.random.default_rng(1).uniform(1.0, 2.1, 24)), 6.0)
    ts = np.linspace(0, 1000, 20001)
    a = perfil.chegada_vet(0, 1, ts)
    confere('FIFO: chegada não decrescente na saída (perfil aleatório, 20.001 saídas)', bool(np.all(np.diff(a) >= -1e-9)))
    confere('escalar e vetorial coincidem', bool(np.allclose(a[::500], [perfil.chegada(0, 1, t) for t in ts[::500]], atol=1e-9)))
    xi = np.random.default_rng(2).lognormal(0, 0.3, 1000)
    b = perfil.chegada_vet(0, 1, np.full(1000, 77.0), xi)
    o = np.argsort(xi)
    confere('chegada crescente em ξ (buffer de quantil = quantil da chegada)', bool(np.all(np.diff(b[o]) >= -1e-9)))
    m = 1.37
    confere('buffer: escalada(m) ≡ ξ = m', abs(perfil.escalada(m).chegada(0, 1, 77.0) - perfil.chegada_vet(0, 1, np.array([77.0]), np.array([m]))[0]) < 1e-9)


def equivalencia():
    print('2. perfil plano ≡ código estático (instância SP1_050_HIGH)')
    inst = ns.load_instance(os.path.join(BASE, 'instances', 'caso_sp', 'CCVRPTW-ST_SP1_050_HIGH.json'))
    D, T = ns.build_matrices(inst)
    M = td.MatrizTD(T, [1.0] * 24, 6.0)
    n = len(inst['customers']); rng = random.Random(7)
    cv = inst['thermal_profile']['sigma_t_pct']
    tt = tr._lognormal_media1(np.random.default_rng(3), cv, (400, n + 1))
    sc = tr.sortear_cenarios(inst, 200, np.random.default_rng(5), n_zones=1, tempo_viagem=True)
    nom = tr.cenario_nominal(inst)
    e_dm = e_du = e_q = e_qs = 0.0; dif_on = 0; n_viav = 0; n_inv_dif = 0
    cust = inst['customers']

    def atrasa(r):
        t = 0.0; prev = 0
        for c in r:
            t += T[prev][c]
            if t > cust[c - 1]['due_date_min']:
                return True
            t = max(t, cust[c - 1]['ready_time_min']) + cust[c - 1]['service_time_min']; prev = c
        return False
    for _ in range(300):
        r = rng.sample(range(1, n + 1), rng.randint(1, 9))
        d1, d2 = cc.delta_max(r, inst, T), cc.delta_max(r, inst, M)
        # Δmax só é usado em rotas sem atraso no plano (Split e refinamento garantem); nas rotas que já
        # atrasam saindo em 0, o estático devolve min(F0, espera) sem significado e o dependente devolve 0
        if atrasa(r):
            n_inv_dif += abs(d1 - d2) > 1e-6
        else:
            n_viav += 1; e_dm = max(e_dm, abs(d1 - d2))
        g = rng.choice([0.0, 0.25, 0.5, 1.0]) * d1
        e_du = max(e_du, abs(cc.duracao(r, inst, T, g) - cc.duracao(r, inst, M, g)))
        dif_on += int(np.sum(cc.pontual_rota(r, inst, T, tt, g) != cc.pontual_rota(r, inst, M, tt, g)))
        e_q = max(e_q, float(np.max(np.abs(tr.route_quality(r, inst, T, nom, t0_min=g) - tr.route_quality(r, inst, M, nom, t0_min=g)))))
        e_qs = max(e_qs, float(np.max(np.abs(tr.route_quality(r, inst, T, sc, t0_min=g) - tr.route_quality(r, inst, M, sc, t0_min=g)))))
    confere(f'Δmax (bisseção) = min(F0, espera) nas {n_viav} rotas sem atraso no plano: erro máx. {e_dm:.2e} min '
            f'(rotas que já atrasam: {n_inv_dif} com Δmax diferente, não usadas pelo motor)', e_dm < 1e-6)
    confere(f'duração: erro máx. {e_du:.2e} min', e_du < 1e-8)
    confere(f'pontualidade em 400 cenários × 300 rotas: {dif_on} divergências', dif_on == 0)
    confere(f'qualidade nominal: erro máx. {e_q:.2e}', e_q < 1e-10)
    confere(f'qualidade em 200 cenários: erro máx. {e_qs:.2e}', e_qs < 1e-10)
    for bq in (None, 0.90):
        cs = cc.Contexto(inst, 0, buffer_q=bq, decoder='split', p_ls=0.5)
        ct = cc.Contexto(inst, 0, buffer_q=bq, decoder='split', p_ls=0.5)
        ct.ttime = M; ct.td = True
        ct.ttime_plan = M.escalada(cc.multiplicador_quantil(cv, bq)) if bq else M
        ct.sozinho_inviavel = {i for i in range(1, n + 1) if ct.ttime_plan.chegada(0, i, 0.0) > ct.due[i]}
        confere(f'buffer {bq}: mesmos clientes inviáveis sozinhos', cs.sozinho_inviavel == ct.sozinho_inviavel)
        iguais = 0; viav = 0
        for _ in range(200):
            ch = list(range(1, n + 1)); rng.shuffle(ch)
            iguais += cc.split_decode(ch, cs) == cc.split_decode(ch, ct)
            r = ch[:rng.randint(1, 8)]
            viav += cc._rota_viavel(r, cs) == cc._rota_viavel(r, ct)
        confere(f'buffer {bq}: Split idêntico em {iguais}/200 cromossomos', iguais == 200)
        confere(f'buffer {bq}: viabilidade idêntica em {viav}/200 rotas', viav == 200)


def cv_por_arco():
    """Incerteza própria de cada arco (caso de Atenas, Entrada 103)."""
    print('3. CV próprio de cada arco')
    rng = np.random.default_rng(11)
    ff = rng.uniform(5, 40, (6, 6)).tolist()
    perfil = [1.0] * 24
    k0 = 0.15
    xi = tr._lognormal_media1(np.random.default_rng(4), k0, 200000)
    sem = td.MatrizTD(ff, perfil, 8.0)
    uni = td.MatrizTD(ff, perfil, 8.0, cv_arco=[[k0] * 6 for _ in range(6)], cv_ref=k0)
    t0 = np.zeros(len(xi))
    confere('CV uniforme = cv_ref reproduz a matriz sem CV por arco (cenários)',
            bool(np.allclose(sem.chegada_vet(1, 2, t0, xi), uni.chegada_vet(1, 2, t0, xi), rtol=0, atol=1e-9)))
    mq = cc.multiplicador_quantil(k0, 0.9)
    confere('CV uniforme: buffer igual', abs(sem.escalada(mq).chegada(1, 2, 3.0) - uni.escalada(mq).chegada(1, 2, 3.0)) < 1e-9)
    confere('plano nominal (m = 1) usa a média', abs(uni.chegada(1, 2, 0.0) - ff[1][2]) < 1e-9)
    cva = rng.uniform(0.05, 0.30, (6, 6)).tolist()
    het = td.MatrizTD(ff, perfil, 8.0, cv_arco=cva, cv_ref=k0)
    v = het.chegada_vet(3, 4, t0, xi) / ff[3][4]
    confere(f'arco com CV {cva[3][4]:.3f}: multiplicador com média {v.mean():.4f} e CV {v.std() / v.mean():.4f}',
            abs(v.mean() - 1) < 0.003 and abs(v.std() / v.mean() - cva[3][4]) < 0.004)
    q = float(np.mean(v <= het.escalada(mq).chegada(3, 4, 0.0) / ff[3][4]))
    confere(f'buffer 0,90 = quantil 0,90 do arco (fração dos cenários abaixo: {q:.4f})', abs(q - 0.9) < 0.003)


if __name__ == '__main__':
    propriedades()
    equivalencia()
    cv_por_arco()
    print('\nTODOS OS TESTES PASSARAM' if not falhas else f'\n{len(falhas)} FALHA(S): {falhas}')
