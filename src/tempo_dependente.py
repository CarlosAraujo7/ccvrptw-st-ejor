#!/usr/bin/env python3
"""
Tempos de viagem dependentes do horário (16/10/2026, diário Entrada 101; melhoria 2 da análise crítica).

Modelo de Ichoua, Gendreau e Potvin (2003, doi:10.1016/S0377-2217(02)00147-9): a velocidade em cada
arco é constante por faixa horária (aqui, por hora do relógio), igual à de fluxo livre dividida pelo
fator de tempo f(h) = 1 + c(h)/100, em que c(h) é o congestionamento da hora (TomTom Traffic Index).
O veículo percorre o "comprimento" do arco, medido em minutos de fluxo livre τ, à taxa 1/f(h) por
minuto. Com G(c) = ∫_0^c ds / f(hora(s)) (relógio c em minutos), a chegada ao sair no relógio c é
    A(c, τ) = G⁻¹(G(c) + τ),
contínua e crescente em c (propriedade FIFO: sair mais tarde nunca faz chegar mais cedo).

Incerteza: o multiplicador lognormal ξ (média 1, CV da instância) do modelo estático multiplica o
comprimento em fluxo livre, A = G⁻¹(G(c) + ξ τ); com perfil constante f, o tempo de viagem é f ξ τ, o
mesmo do modelo estático com a matriz f τ. A chegada é crescente em ξ, então o buffer de quantil q
(τ multiplicado por m_q) dá exatamente o quantil q da chegada a cada arco, dada a saída.

O tempo t do motor (minutos desde o início da operação) corresponde ao relógio c = t + 60·hora_inicio.
Uma instância é dependente do horário quando traz o bloco 'td' (ver pesquisa/caso_sp/gerar_caso_sp_td.py);
o build_matrices do caso SP devolve então uma MatrizTD no lugar da matriz de tempos, e as funções de
ccvrp_c e thermal_route que recebem a matriz de tempos desviam para os cálculos daqui. Instâncias sem
o bloco 'td' seguem o código estático original, sem nenhuma alteração.
"""
import math
from bisect import bisect_right
from typing import List, Optional

import numpy as np

HORAS = 24 * 7          # o perfil diário é repetido por uma semana (rotas cabem com folga)


class MatrizTD:
    """Matriz de tempos dependentes do horário: comprimentos em fluxo livre (min) e perfil horário.

    Opcional (caso de Atenas, 16/10/2026, diário Entrada 103): incerteza PRÓPRIA DE CADA ARCO. cv_arco[i][j] é o
    coeficiente de variação do tempo do arco (i, j) e cv_ref o da instância (thermal_profile.sigma_t_pct), com
    que todos os sorteios de cenários continuam sendo feitos. Cada multiplicador sorteado ξ ~ lognormal(média 1,
    cv_ref) é levado ao mesmo quantil da lognormal do arco: z = (ln ξ + s₀²/2)/s₀ e ξ_ij = exp(−s_ij²/2 + s_ij z),
    com s² = ln(1 + cv²). Os números aleatórios comuns se preservam, e o buffer de quantil q (m multiplicado
    pelo quantil q de cv_ref) vira o quantil q de cada arco. O plano nominal (m = 1) usa a média, ξ_ij = 1.
    Sem cv_arco, o comportamento é o anterior, sem nenhuma alteração."""

    def __init__(self, ff: List[List[float]], fator_hora: List[float], hora_inicio: float, m: float = 1.0,
                 cv_arco: Optional[List[List[float]]] = None, cv_ref: Optional[float] = None):
        assert len(fator_hora) == 24 and min(fator_hora) > 0
        self.ff, self.fator_hora, self.hora_inicio, self.m = ff, list(fator_hora), float(hora_inicio), float(m)
        self.C0 = 60.0 * self.hora_inicio
        f = [self.fator_hora[k % 24] for k in range(HORAS)]
        Gb = [0.0]
        for k in range(HORAS):
            Gb.append(Gb[-1] + 60.0 / f[k])
        self._f, self._Gb = f, Gb
        self._fa, self._Gba = np.array(f), np.array(Gb)
        self._dmax = {}
        self.cv_arco, self.cv_ref = cv_arco, cv_ref
        if cv_arco is None:
            self._tau = [[self.m * x for x in row] for row in ff]
        else:
            assert cv_ref and cv_ref > 0, 'cv_ref (CV dos sorteios) é obrigatório com cv_arco'
            self._s = np.sqrt(np.log1p(np.asarray(cv_arco, dtype=float) ** 2))
            self._s0 = math.sqrt(math.log1p(cv_ref * cv_ref))
            if self.m == 1.0:
                mult = np.ones_like(self._s)
            else:
                z = (math.log(self.m) + self._s0 ** 2 / 2) / self._s0
                mult = np.exp(-self._s ** 2 / 2 + self._s * z)
            self._tau = (np.asarray(ff, dtype=float) * mult).tolist()

    def __len__(self):
        return len(self.ff)

    def escalada(self, m: float) -> 'MatrizTD':
        """Mesma matriz com os comprimentos multiplicados por m (buffer de quantil)."""
        return MatrizTD(self.ff, self.fator_hora, self.hora_inicio, self.m * m, self.cv_arco, self.cv_ref)

    def chegada(self, i: int, j: int, t: float) -> float:
        """Chegada a j (tempo do motor) saindo de i no instante t, sem incerteza."""
        c = t + self.C0
        k = int(c // 60.0)
        g = self._Gb[k] + (c - 60.0 * k) / self._f[k] + self._tau[i][j]
        k = bisect_right(self._Gb, g) - 1
        return 60.0 * k + (g - self._Gb[k]) * self._f[k] - self.C0

    def chegada_vet(self, i: int, j: int, t: np.ndarray, xi: Optional[np.ndarray] = None) -> np.ndarray:
        """Chegada a j por cenário (K,), saindo de i em t (K,), com multiplicadores ξ (K,) ou sem incerteza."""
        c = np.asarray(t, dtype=float) + self.C0
        k = (c // 60.0).astype(int)
        if xi is None:
            soma = self._tau[i][j]
        elif self.cv_arco is None:
            soma = self.m * self.ff[i][j] * xi
        else:                                    # mesmo quantil na lognormal do arco
            s = self._s[i][j]
            soma = self.ff[i][j] * np.exp(-s * s / 2 + s * (np.log(xi) + self._s0 ** 2 / 2) / self._s0)
        g = self._Gba[k] + (c - 60.0 * k) / self._fa[k] + soma
        k = np.searchsorted(self._Gba, g, side='right') - 1
        return 60.0 * k + (g - self._Gba[k]) * self._fa[k] - self.C0


def eh_td(ttime) -> bool:
    return isinstance(ttime, MatrizTD)


# ─────────────────────────────────────────────────────────────────────────────
# Versões dependentes do horário das funções de cronograma de ccvrp_c
# ─────────────────────────────────────────────────────────────────────────────

def delta_max(route: List[int], inst: dict, td: MatrizTD) -> float:
    """Maior adiamento d da saída que não cria atraso no plano nem aumenta a duração da rota.
    No modelo estático, min(F0, espera total); aqui, por bisseção, porque o atraso propagado muda de
    tamanho de arco para arco. O predicado "sair em d não atrasa ninguém e o atraso é absorvido por
    alguma espera" é monótono em d (FIFO), e sem absorção a volta ao depósito fica mais tarde."""
    cust = inst['customers']
    chave = (id(cust), tuple(route))
    if chave in td._dmax:
        return td._dmax[chave]
    rdy = [cust[c - 1]['ready_time_min'] for c in route]
    due = [cust[c - 1]['due_date_min'] for c in route]
    svc = [cust[c - 1]['service_time_min'] for c in route]
    ini0 = []; t = 0.0; prev = 0; atrasa = False   # início do serviço saindo em 0
    for k, c in enumerate(route):
        a = td.chegada(prev, c, t); atrasa |= a > due[k]
        t = max(a, rdy[k]); ini0.append(t); t += svc[k]; prev = c

    def ok(d):
        t = d; prev = 0
        for k, c in enumerate(route):
            a = td.chegada(prev, c, t)
            if a > due[k]:
                return False
            s = max(a, rdy[k])
            if s <= ini0[k]:
                return True                      # atraso absorvido: o resto do cronograma não muda
            t = s + svc[k]; prev = c
        return False
    lo, hi = 0.0, (max(0.0, due[0]) if route else 0.0)
    if not route or atrasa:
        res = 0.0                                # como no estático: F0 < 0 dá Δmax = 0
    elif ok(hi):
        res = hi
    else:
        for _ in range(42):
            mid = 0.5 * (lo + hi)
            if ok(mid):
                lo = mid
            else:
                hi = mid
        res = lo
    if len(td._dmax) > 400000:
        td._dmax.clear()
    td._dmax[chave] = res
    return res


def duracao(route: List[int], inst: dict, td: MatrizTD, t0: float) -> float:
    cust = inst['customers']
    t = t0; prev = 0
    for c in route:
        t = td.chegada(prev, c, t)
        t = max(t, cust[c - 1]['ready_time_min']) + cust[c - 1]['service_time_min']
        prev = c
    return td.chegada(prev, 0, t) - t0


def pontual_rota(route: List[int], inst: dict, td: MatrizTD, tt: np.ndarray, t0: float, delta_tol: float) -> np.ndarray:
    cust = inst['customers']
    K = tt.shape[0]
    t = np.full(K, float(t0)); ok = np.ones(K, dtype=bool); prev = 0
    for c in route:
        t = td.chegada_vet(prev, c, t, tt[:, c])
        ok &= (t - cust[c - 1]['due_date_min']) <= delta_tol
        t = np.maximum(t, cust[c - 1]['ready_time_min']) + cust[c - 1]['service_time_min']
        prev = c
    return ok


def viavel(r: List[int], ctx) -> bool:
    """Capacidade e janelas com os tempos de planejamento (saída em 0), como ccvrp_c._rota_viavel."""
    load = 0.0; t = 0.0; prev = 0; td = ctx.ttime_plan
    for c in r:
        load += ctx.dem[c]
        if load > ctx.cap:
            return False
        t = td.chegada(prev, c, t)
        if t > ctx.due[c]:
            return False
        t = max(t, ctx.rdy[c]) + ctx.svc[c]; prev = c
    return True


def split_decode(chrom: List[int], ctx):
    """Split de ccvrp_c.split_decode com chegadas dependentes do horário (mesma programação dinâmica)."""
    perm = [c for c in chrom if c not in ctx.sozinho_inviavel]
    n = len(perm); td = ctx.ttime_plan; D = ctx.dist
    V = [math.inf] * (n + 1); P = [-1] * (n + 1); V[0] = 0.0
    for i in range(n):
        if V[i] == math.inf:
            continue
        load = 0.0; t = 0.0; d = 0.0; prev = 0
        for j in range(i, n):
            c = perm[j]
            load += ctx.dem[c]
            if load > ctx.cap:
                break
            t = td.chegada(prev, c, t)
            if t > ctx.due[c]:
                break
            t = max(t, ctx.rdy[c]) + ctx.svc[c]
            d += D[prev][c]
            custo = V[i] + ctx.fix + ctx.var * (d + D[c][0])
            if custo < V[j + 1]:
                V[j + 1] = custo; P[j + 1] = i
            prev = c
    rotas = []; j = n
    while j > 0:
        i = P[j]; rotas.append(perm[i:j]); j = i
    rotas.reverse()
    return rotas, not ctx.sozinho_inviavel


def cronograma(route: List[int], inst: dict, td: MatrizTD, K: int, xi: Optional[np.ndarray], t0: float):
    """Para thermal_route.route_quality: tempos de viagem por trecho (K×m, min), espera (K×m, h),
    início do serviço (K×m, h) e atraso (K×m, min). xi: K×(n+1) ou None (cenário nominal)."""
    cust = inst['customers']; m = len(route)
    tl = np.zeros((K, m)); wait = np.zeros((K, m)); t_srv = np.zeros((K, m)); late = np.zeros((K, m))
    t = np.full(K, float(t0)); prev = 0
    for j, c in enumerate(route):
        a = td.chegada_vet(prev, c, t, None if xi is None else xi[:, c])
        tl[:, j] = a - t
        late[:, j] = np.maximum(0.0, a - cust[c - 1]['due_date_min'])
        w = np.maximum(0.0, cust[c - 1]['ready_time_min'] - a)
        t = a + w
        wait[:, j], t_srv[:, j] = w / 60.0, t / 60.0
        t = t + cust[c - 1]['service_time_min']
        prev = c
    return tl, wait, t_srv, late
