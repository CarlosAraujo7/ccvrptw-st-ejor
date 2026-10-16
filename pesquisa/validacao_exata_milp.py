"""
Validacao exata (frente nova, 27/08/2026) — pesquisa/plano_validacao_exata.md

Formulacao MILP (fluxo de veiculos, 2 indices) para o VRPTW deterministico
do CCVRPTW-ST, replicando FIELMENTE as regras de decode()/evaluate() de
src/nsga2_simheuristic.py (nao um VRPTW generico de livro-texto):

  - b_i = inicio de servico no cliente i (= max(chegada, ready_time), igual
    a decode() faz: wait = max(0, rt-arrival); r_time = arrival+wait+st).
  - ready_time_i <= b_i <= due_date_i  (janela dura, igual a decode()).
  - SEM restricao de horario no retorno ao deposito — decode() nao checa
    isso, so soma a distancia de volta no custo (evaluate(): nodes =
    [0]+route+[0]).  Verificado lendo o codigo antes de assumir, nao de
    memoria (LEI II.3).
  - Custo f1 = fixed_cost_BRL * (num. de rotas usadas) + variable_cost_BRL_km
    * distancia total — igual a evaluate(), linhas 534-535/553-554.

Etapa 1 (esta primeira versao): so f1 (custo), sem PWL de f2 ainda —
testar a formulacao de roteamento em si numa instancia pequena antes de
empilhar mais complexidade em cima (mesma disciplina que pegou o bug de
normalizacao do MOEA/D).
"""
import sys
import json
import math
import time as pytime

sys.path.insert(0, "src")
import pulp


def euclidean(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def load_instance_nodes(path):
    with open(path, encoding="utf-8") as f:
        inst = json.load(f)
    depot = inst["depot"]
    customers = inst["customers"]
    nodes = [depot] + customers  # indice 0 = deposito, 1..n = clientes (id bate com posicao)
    n = len(customers)
    speed = inst["fleet"]["speed_km_h"]

    dist = [[0.0] * (n + 1) for _ in range(n + 1)]
    ttime = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:
                d = euclidean(nodes[i]["x"], nodes[i]["y"], nodes[j]["x"], nodes[j]["y"])
                dist[i][j] = d
                ttime[i][j] = (d / speed) * 60.0  # minutos

    return inst, nodes, dist, ttime, n


def solve_exact_cost(inst_path, time_limit_s=600, verbose=True):
    inst, nodes, dist, ttime, n = load_instance_nodes(inst_path)
    fleet = inst["fleet"]
    fixed_cost = fleet["fixed_cost_BRL"]
    var_cost = fleet["variable_cost_BRL_km"]
    capacity = fleet["capacity_kg"]
    num_vehicles = fleet["num_vehicles"]
    horizon = inst["horizon_minutes"]

    V = list(range(n + 1))          # 0..n (0 = deposito)
    C = list(range(1, n + 1))       # clientes

    demand = {i: nodes[i]["demand_kg"] for i in C}
    ready = {i: nodes[i]["ready_time_min"] for i in C}
    due = {i: nodes[i]["due_date_min"] for i in C}
    svc = {i: nodes[i]["service_time_min"] for i in C}

    prob = pulp.LpProblem("ccvrptw_exact_cost", pulp.LpMinimize)

    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
         for i in V for j in V if i != j}
    # tempo de inicio de servico (so p/ clientes; deposito = 0 fixo)
    b = {i: pulp.LpVariable(f"b_{i}", lowBound=ready[i], upBound=due[i]) for i in C}
    # carga acumulada na rota ate o cliente i (MTZ de capacidade)
    load = {i: pulp.LpVariable(f"load_{i}", lowBound=demand[i], upBound=capacity) for i in C}

    # Objetivo: custo fixo por rota usada (= arcos saindo do deposito) + custo variavel por distancia
    prob += (fixed_cost * pulp.lpSum(x[(0, j)] for j in C)
             + var_cost * pulp.lpSum(dist[i][j] * x[(i, j)] for i in V for j in V if i != j))

    # Grau de entrada/saida = 1 para cada cliente
    for i in C:
        prob += pulp.lpSum(x[(i, j)] for j in V if j != i) == 1
        prob += pulp.lpSum(x[(j, i)] for j in V if j != i) == 1

    # Numero de rotas <= frota disponivel; saidas do deposito = entradas no deposito
    prob += pulp.lpSum(x[(0, j)] for j in C) <= num_vehicles
    prob += pulp.lpSum(x[(0, j)] for j in C) == pulp.lpSum(x[(i, 0)] for i in C)

    # Big-M para tempo e capacidade
    M_time = horizon + max(max(row) for row in ttime) + max(svc.values())
    M_load = capacity + max(demand.values())

    for i in C:
        for j in C:
            if i != j:
                # b_j >= b_i + svc_i + ttime_ij - M*(1-x_ij)   (so ativa se x_ij=1)
                prob += b[j] >= b[i] + svc[i] + ttime[i][j] - M_time * (1 - x[(i, j)])
                # load_j >= load_i + demand_j - M*(1-x_ij)  (tambem serve de eliminacao de subciclo)
                prob += load[j] >= load[i] + demand[j] - M_load * (1 - x[(i, j)])

    # Partida do deposito: se x_0j=1, b_j >= ttime[0][j] (chegada minima), respeitando ready_time via lowBound already
    for j in C:
        prob += b[j] >= ready[j] + ttime[0][j] - M_time  # relaxada; a apertada real vem da rota via b_i's
        prob += b[j] >= ttime[0][j] - M_time * (1 - x[(0, j)])

    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=time_limit_s)
    t0 = pytime.time()
    status = prob.solve(solver)
    elapsed = pytime.time() - t0

    result = {
        "instance": inst_path,
        "n": n,
        "status": pulp.LpStatus[status],
        "objective_f1_BRL": pulp.value(prob.objective),
        "elapsed_s": elapsed,
        "routes": [],
    }

    if pulp.LpStatus[status] in ("Optimal", "Not Solved", "Infeasible", "Unbounded", "Undefined"):
        # tenta reconstruir rotas mesmo se nao-otimo (ex.: limite de tempo), se houver solucao
        arcs = [(i, j) for (i, j) in x if x[(i, j)].value() and x[(i, j)].value() > 0.5]
        succ = {i: j for (i, j) in arcs if i == 0 or True}
        # monta rotas seguindo sucessores a partir de cada saida do deposito
        starts = [j for (i, j) in arcs if i == 0]
        visited_global = set()
        for s in starts:
            route = [s]
            cur = s
            visited_global.add(s)
            while True:
                nxt = next((j for (i, j) in arcs if i == cur), None)
                if nxt is None or nxt == 0:
                    break
                route.append(nxt)
                visited_global.add(nxt)
                cur = nxt
            result["routes"].append(route)

    return result


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "instances/validacao_exata/CCVRPTW-ST_valexata_C1_005_LOW.json"
    tlim = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    r = solve_exact_cost(path, time_limit_s=tlim, verbose=True)
    print("\n=== RESULTADO ===")
    print(f"instancia: {r['instance']} (n={r['n']})")
    print(f"status: {r['status']}")
    print(f"f1 (custo exato): R$ {r['objective_f1_BRL']:.2f}")
    print(f"tempo: {r['elapsed_s']:.2f}s")
    print(f"rotas ({len(r['routes'])}):")
    for rt in r["routes"]:
        print(" ", rt)
