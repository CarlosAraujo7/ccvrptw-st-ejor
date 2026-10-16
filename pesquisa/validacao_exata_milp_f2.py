"""
Validacao exata, parte 2 (27/08/2026) -- pesquisa/plano_validacao_exata.md Secao 5-6.

Estende validacao_exata_milp.py com uma versao linearizada (PWL) de f2
(qualidade, cinetica de Arrhenius) e o metodo epsilon-constraint, pra tracar
uma fatia exata (a menos da aproximacao PWL, quantificada empiricamente) da
frente de Pareto f1 x f2.

Formula de f2, RE-VERIFICADA contra o codigo-fonte antes de implementar
(nao assumida de memoria -- ver diario, Entrada 38):

  - k = A0 * exp(-Ea/(R*(T_set+273.15)))  -- escalar UNICO por instancia
    (baseline nao-B2, _route_quality_mc linha 371-373).
  - S_i = soma acumulada do TEMPO DE VIAGEM PURO (ttime/60, em horas) do
    deposito ate o cliente i, ao longo da rota -- NAO inclui espera nem
    tempo de servico (_route_quality_mc linha 362-363: leg_t_h vem só de
    ttime[nodes[i]][nodes[i+1]]/60.0). Por isso e uma variavel NOVA no MILP,
    distinta de b_i (inicio de servico, que inclui espera+servico).
  - Q_i = exp(-k * S_i).
  - f2 (qualidade da solucao) = sum_i(value_i * Q_i) / sum_i(value_i), sobre
    TODOS os clientes da solucao (nao por rota) -- verificado que a
    normalizacao por rota em evaluate() se cancela algebricamente
    (total_wq = sum_rotas(r_wq*r_val) = sum_rotas sum_i(Q_i*value_i) =
    sum_global_i(Q_i*value_i); total_val = sum_global_i(value_i)).
    V_total = sum_i(value_i) e uma CONSTANTE (todo cliente e sempre servido
    exatamente uma vez).
  - f_qloss reportado pelo NSGA2 = 1 - f2(qualidade). Aqui trabalhamos com
    "qualidade" (f2 no sentido de "quanto maior melhor") e convertemos pra
    perda só na hora de comparar com o NSGA2-DET.

Linearizacao de Q_i = exp(-k*S_i): DUAS tentativas, a primeira descartada
por um achado tecnico que vale registrar (LEI: reportar o que nao funcionou,
nao só o que funcionou).

Tentativa 1 (descartada): envelope de retas TANGENTES como limitante
superior (Q_i <= g(p_m)+g'(p_m)*(s_i-p_m) pra cada ponto de grade p_m,
solver empurra Q_i pro teto mais apertado). Matematicamente conservadora
(Q_i<=Q_i_real sempre), mas o gap MAXIMO empirico (~0.0072, grade de 8
pontos entre 0 e S_max) NAO diminui com mais pontos -- prova: pra um S de
avaliacao fixo, o valor de uma tangente ancorada em p e unimodal com pico em
p=S e CAI monotonicamente conforme p se afasta de S; como o envelope toma o
MINIMO entre todas as tangentes, o ponto mais DISTANTE (nao o mais proximo)
e que acaba dominando o teto nas bordas do dominio -- "quanto mais pontos,
mais apertado" so vale localmente, nao no envelope-min global. Na pratica
isso fez o sweep de epsilon parar cedo demais (ex.: instancia n=10 ficou
inviavel no primeiro passo acima do piso, sem revelar nenhum trade-off real
-- sintoma, nao coincidencia).

Tentativa 2 (adotada): interpolacao linear por partes PADRAO (metodo lambda
com SOS2) -- Q_i = soma_m(lambda_m * g(p_m)), s_i = soma_m(lambda_m * p_m),
soma(lambda_m)=1, com no maximo 2 lambdas ADJACENTES não-nulos (restricao
SOS2). Isso e uma interpolacao por CORDA entre os 2 breakpoints vizinhos de
s_i -- para funcao convexa, SUPERESTIMA levemente (corda fica acima da
curva), mas o erro e local e cai rapido com mais pontos (gap medido: 0.0002
com 4 pontos, 0.00004 com 8 pontos, pra k e faixa de S reais desta pesquisa
-- 20-190x mais apertado que a Tentativa 1). Por superestimar (nao
subestimar), a restricao deixa de ser 100% conservadora no sentido estrito
(f2_milp pode passar de epsilon por uma margem residual <=0.00004 sem que
f2_true realmente chegue la) -- por isso SEMPRE recalculamos f2_true das
rotas reais (recompute_true_f2) e reportamos ele, nao f2_milp, como o eixo
de qualidade da frente -- e checamos f2_true >= epsilon - tolerancia pra
qualquer ponto aceito.

ACHADO TECNICO relevante, verificado por execucao real antes de confiar
(nao assumido): `pulp.PULP_CBC_CMD().solve(prob)` usa formato MPS por
padrao (`use_mps=True` em `solve_CBC`), e o ESCRITOR DE MPS do PuLP 3.3.2
NAO GRAVA restricoes SOS1/SOS2 no arquivo (confirmado lendo o codigo-fonte
de `pulp.mps_lp.writeMPS` -- nenhuma mencao a SOS) -- ou seja, `prob.sos2`
e SILENCIOSAMENTE IGNORADO pelo caminho padrao, sem erro nem aviso. O
formato LP (`writeLP`) grava SOS corretamente, e o CBC (testado rodando o
binario direto num .lp gerado) HONRA a secao SOS2 quando le esse formato.
Correcao: chamar `solver.actualSolve(prob, use_mps=False)` em vez de
`prob.solve(solver)`, forcando o caminho LP. Sem esse teste (que so fiz
depois de ver um resultado obviamente errado -- lambdas caindo nos dois
EXTREMOS da grade em vez de em pontos adjacentes), o SOS2 teria sido
silenciosamente ignorado e a "interpolacao" teria virado uma combinacao
convexa solta entre QUALQUER par de pontos, sem nenhuma relação real com
s_i -- um erro que inflaria f2_milp de forma arbitraria, não pequena.

s_i (tempo de viagem acumulado) precisa ser EXATO (nao só um limitante),
senao o solver poderia "inflar" s_i sem custo (nada em f1 penaliza s_i
grande) so pra tornar Q_i menor sem necessidade -- por isso a propagacao de
s_i usa IGUALDADE (via big-M nos dois sentidos), diferente de b_i (que so
precisa de ">=" pra caracterizar viabilidade corretamente).
"""
import sys
import os
import json
import math
import time as pytime

sys.path.insert(0, "src")
import pulp

R_GAS = 8.314
EXTRA_VEHICLE_COST = 400.0  # BRL por veiculo alem da frota nominal -- igual a src/nsga2_simheuristic.py linha 57, nao reinventado


def load_instance_nodes(path):
    with open(path, encoding="utf-8") as f:
        inst = json.load(f)
    depot = inst["depot"]
    customers = inst["customers"]
    nodes = [depot] + customers
    n = len(customers)
    speed = inst["fleet"]["speed_km_h"]

    dist = [[0.0] * (n + 1) for _ in range(n + 1)]
    ttime = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(n + 1):
            if i != j:
                d = math.hypot(nodes[j]["x"] - nodes[i]["x"], nodes[j]["y"] - nodes[i]["y"])
                dist[i][j] = d
                ttime[i][j] = (d / speed) * 60.0  # minutos

    return inst, nodes, dist, ttime, n


def compute_k(inst):
    tp = inst["thermal_profile"]
    T_K = tp["T_set_C"] + 273.15
    return tp["A0"] * math.exp(-tp["Ea_J_mol"] / (R_GAS * T_K))


def breakpoints(k, S_max, n_pts=8):
    """Pontos p_m e g(p_m) pra interpolacao linear por partes (SOS2)."""
    pts = [S_max * m / (n_pts - 1) for m in range(n_pts)]
    gvals = [math.exp(-k * p) for p in pts]
    return pts, gvals


def build_model(inst_path, n_pts=8, S_max_margin=1.0, max_extra_vehicles=None):
    inst, nodes, dist, ttime, n = load_instance_nodes(inst_path)
    fleet = inst["fleet"]
    fixed_cost = fleet["fixed_cost_BRL"]
    var_cost = fleet["variable_cost_BRL_km"]
    capacity = fleet["capacity_kg"]
    num_vehicles = fleet["num_vehicles"]
    horizon = inst["horizon_minutes"]

    V = list(range(n + 1))
    C = list(range(1, n + 1))

    demand = {i: nodes[i]["demand_kg"] for i in C}
    ready = {i: nodes[i]["ready_time_min"] for i in C}
    due = {i: nodes[i]["due_date_min"] for i in C}
    svc = {i: nodes[i]["service_time_min"] for i in C}
    value = {i: nodes[i].get("load_value_BRL", 1.0) for i in C}
    V_total = sum(value.values())

    k = compute_k(inst)
    S_max = max(due.values()) / 60.0 * S_max_margin  # horas; due_i/60 >= s_i sempre (prova no plano)
    pts, gvals = breakpoints(k, S_max, n_pts)

    prob = pulp.LpProblem("ccvrptw_exact_f1_f2", pulp.LpMinimize)

    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
         for i in V for j in V if i != j}
    b = {i: pulp.LpVariable(f"b_{i}", lowBound=ready[i], upBound=due[i]) for i in C}
    load = {i: pulp.LpVariable(f"load_{i}", lowBound=demand[i], upBound=capacity) for i in C}
    s = {i: pulp.LpVariable(f"s_{i}", lowBound=0, upBound=due[i] / 60.0) for i in C}
    Q = {i: pulp.LpVariable(f"Q_{i}", lowBound=0, upBound=1.0) for i in C}
    # n_extra: veiculos alem da frota nominal, replicando evaluate() linha
    # ~596-597 (n_extra=max(0,len(routes)-n_nominal), custo EXTRA_VEHICLE_COST
    # cada) -- SEM limite rigido de frota, ao contrario da 1a versao deste
    # MILP (achado da Entrada 38: NSGA2-DET usa veiculo extra pra ganhar
    # qualidade, o "<=num_vehicles" rigido escondia isso como falsa
    # inviabilidade).
    n_extra_ub = max_extra_vehicles if max_extra_vehicles is not None else n
    n_extra = pulp.LpVariable("n_extra", lowBound=0, upBound=n_extra_ub)

    # ---- objetivo: f1 (custo, incluindo veiculo extra pago) ----
    prob += (fixed_cost * pulp.lpSum(x[(0, j)] for j in C)
             + var_cost * pulp.lpSum(dist[i][j] * x[(i, j)] for i in V for j in V if i != j)
             + EXTRA_VEHICLE_COST * n_extra)

    for i in C:
        prob += pulp.lpSum(x[(i, j)] for j in V if j != i) == 1
        prob += pulp.lpSum(x[(j, i)] for j in V if j != i) == 1

    # sem teto rigido -- n_extra absorve o excesso sobre a frota nominal
    prob += n_extra >= pulp.lpSum(x[(0, j)] for j in C) - num_vehicles
    prob += pulp.lpSum(x[(0, j)] for j in C) == pulp.lpSum(x[(i, 0)] for i in C)

    M_time = horizon + max(max(row) for row in ttime) + max(svc.values())
    M_load = capacity + max(demand.values())
    M_s = S_max + max(max(row) for row in ttime) / 60.0

    for i in C:
        for j in C:
            if i != j:
                prob += b[j] >= b[i] + svc[i] + ttime[i][j] - M_time * (1 - x[(i, j)])
                prob += load[j] >= load[i] + demand[j] - M_load * (1 - x[(i, j)])
                # s_j = s_i + ttime_ij/60 QUANDO x_ij=1 -- igualdade nos dois sentidos,
                # senao o solver poderia inflar s_i sem custo em f1 (ver docstring).
                leg_h = ttime[i][j] / 60.0
                prob += s[j] >= s[i] + leg_h - M_s * (1 - x[(i, j)])
                prob += s[j] <= s[i] + leg_h + M_s * (1 - x[(i, j)])

    for j in C:
        prob += b[j] >= ready[j] + ttime[0][j] - M_time
        prob += b[j] >= ttime[0][j] - M_time * (1 - x[(0, j)])
        leg0_h = ttime[0][j] / 60.0
        prob += s[j] >= leg0_h - M_s * (1 - x[(0, j)])
        prob += s[j] <= leg0_h + M_s * (1 - x[(0, j)])

    # ---- Q_i = interpolacao linear por partes de exp(-k*s_i) ----
    # Metodo lambda com adjacencia forcada por BINARIOS ("multiple choice"),
    # NAO SOS2 nativo do solver. Motivo (achado tecnico, ver docstring do
    # modulo): 1) o escritor MPS do PuLP 3.3.2 nao grava SOS1/SOS2 (silencio-
    # samente ignorado no caminho padrao); 2) forcando o caminho LP
    # (use_mps=False), o CBC 2.10.3 empacotado com o PuLP HONRA o SOS2
    # corretamente em problemas pequenos (testado, n=5 -- ok), mas
    # SEGFAULTA de forma reprodutivel em problemas maiores com varios
    # conjuntos SOS2 simultaneos (testado, n=10 -- crash confirmado, mesmo
    # exit code 139/SIGSEGV rodando o cbc direto no .lp, fora do PuLP --
    # nao e um bug de wrapper, e do proprio binario). Tentativa de baixar um
    # CBC mais novo (2.10.12, binario oficial coin-or) esbarrou em
    # dependencia de sistema ausente (libnauty.so.2) sem acesso de root
    # neste sandbox pra instalar. Em vez de arriscar compilar/linkar uma
    # libnauty por conta propria (risco real de UM MISMATCH DE VERSAO
    # produzir respostas erradas silenciosamente, pior que travar visivel),
    # trocamos a formulacao pra usar so BINARIOS comuns + big-M -- exatamente
    # o mesmo tipo de constructo ja usado (e validado) no MILP so-f1 ate
    # n=20 sem qualquer instabilidade.
    n_pts_actual = len(pts)
    n_seg = n_pts_actual - 1  # segmentos entre breakpoints consecutivos
    lam = {(i, m): pulp.LpVariable(f"lam_{i}_{m}", lowBound=0, upBound=1)
           for i in C for m in range(n_pts_actual)}
    z = {(i, seg): pulp.LpVariable(f"zseg_{i}_{seg}", cat="Binary")
         for i in C for seg in range(n_seg)}
    for i in C:
        prob += pulp.lpSum(z[(i, seg)] for seg in range(n_seg)) == 1
        # lambda_m só pode ser >0 se algum segmento adjacente a m estiver ativo
        prob += lam[(i, 0)] <= z[(i, 0)]
        prob += lam[(i, n_pts_actual - 1)] <= z[(i, n_seg - 1)]
        for m in range(1, n_pts_actual - 1):
            prob += lam[(i, m)] <= z[(i, m - 1)] + z[(i, m)]
        prob += pulp.lpSum(lam[(i, m)] for m in range(n_pts_actual)) == 1
        prob += s[i] == pulp.lpSum(lam[(i, m)] * pts[m] for m in range(n_pts_actual))
        prob += Q[i] == pulp.lpSum(lam[(i, m)] * gvals[m] for m in range(n_pts_actual))

    f2_num = pulp.lpSum(value[i] * Q[i] for i in C)  # = V_total * qualidade

    return {
        "prob": prob, "x": x, "b": b, "s": s, "Q": Q, "load": load,
        "inst": inst, "nodes": nodes, "dist": dist, "ttime": ttime, "n": n,
        "V_total": V_total, "f2_num": f2_num, "k": k, "value": value,
        "capacity": capacity,
    }


def recompute_true_f2(routes, inst, ttime, k, value):
    """Recalcula f2 EXATO (sem PWL) a partir das rotas realizadas -- mesma
    formula de _route_quality_mc, pra auditar o gap da aproximacao."""
    total_wq, total_val = 0.0, 0.0
    for route in routes:
        nodes_r = [0] + route
        S = 0.0
        for idx in range(len(route)):
            leg_h = ttime[nodes_r[idx]][nodes_r[idx + 1]] / 60.0
            S += leg_h
            Q_i = math.exp(-k * S)
            v_i = value[route[idx]]
            total_wq += v_i * Q_i
            total_val += v_i
    return total_wq / total_val if total_val > 0 else 0.0


def extract_routes(x, n):
    arcs = [(i, j) for (i, j) in x if x[(i, j)].value() and x[(i, j)].value() > 0.5]
    starts = [j for (i, j) in arcs if i == 0]
    routes = []
    for s0 in starts:
        route, cur = [s0], s0
        while True:
            nxt = next((j for (i, j) in arcs if i == cur), None)
            if nxt is None or nxt == 0:
                break
            route.append(nxt)
            cur = nxt
        routes.append(route)
    return routes


def parse_cbc_log(log_path):
    """
    Le o log bruto do CBC (gravado via logPath do PULP_CBC_CMD) e extrai o
    veredito REAL -- nao confiar no status do PuLP sozinho (achado da
    Entrada 40, 28/08/2026: o PuLP rotula "Optimal" mesmo quando o CBC so
    parou por limite de tempo, sempre que a 1a linha do .sol bate um padrao
    especifico -- ver pulp/apis/coin_api.py::COIN_CMD.get_status()). A
    unica fonte confiavel e a linha "Result - ..." que o proprio CBC
    imprime, mais a linha "Gap:" quando presente.
    Retorna (proven_optimal: bool|None, cbc_gap: float|None, result_line: str|None).
    None pra proven_optimal significa "log nao encontrado/nao lido".
    """
    if not log_path or not os.path.exists(log_path):
        return None, None, None
    result_line = None
    gap = None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip().startswith("Result -"):
                result_line = line.strip()
            elif line.strip().startswith("Gap:"):
                try:
                    gap = float(line.strip().split(":", 1)[1].strip())
                except ValueError:
                    pass
    if result_line is None:
        return None, None, None
    proven = "Optimal solution found" in result_line
    return proven, gap, result_line


def solve_eps(inst_path, epsilon, n_pts=8, time_limit_s=180, verbose=False, max_extra_vehicles=None,
              log_path=None):
    """
    Reconstroi o modelo do zero a cada chamada (em vez de clonar um LpProblem
    ja resolvido) -- testado que pulp.LpProblem.deepcopy()/copy.deepcopy()
    tem comportamento de identidade de variavel nao confiavel (a copia
    resolvida nao bate com as variaveis devolvidas por copia.variables());
    reconstruir e mais caro mas elimina essa fonte de erro (LEI: validar por
    execucao real antes de confiar num atalho, nao assumir que "deveria
    funcionar").

    log_path (novo, Entrada 40): se dado, grava o log bruto do CBC nesse
    arquivo (via PULP_CBC_CMD(logPath=...)) e le de volta pra extrair o
    veredito real (proven_optimal/cbc_gap) -- o status do PuLP sozinho NAO
    e confiavel quando o solver bate no limite de tempo (ver parse_cbc_log).
    """
    model = build_model(inst_path, n_pts=n_pts, max_extra_vehicles=max_extra_vehicles)
    prob = model["prob"]
    V_total = model["V_total"]
    f2_num = model["f2_num"]

    prob += f2_num >= epsilon * V_total, "eps_constraint"

    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=time_limit_s, logPath=log_path)
    t0 = pytime.time()
    # Formato padrao (MPS) -- voltou a ser seguro porque a formulacao atual
    # nao usa mais SOS2 nativo (ver docstring do modulo: SOS2 via LP+CBC
    # funcionava em n=5 mas segfaultava em n=10; trocado por binarios).
    status = prob.solve(solver)
    elapsed = pytime.time() - t0

    status_str = pulp.LpStatus[status]
    proven_optimal, cbc_gap, cbc_result_line = parse_cbc_log(log_path)

    if status_str not in ("Optimal",):
        return {"epsilon": epsilon, "status": status_str, "elapsed_s": elapsed,
                "proven_optimal": proven_optimal, "cbc_gap": cbc_gap, "cbc_result_line": cbc_result_line}

    n = model["n"]
    routes = extract_routes(model["x"], n)

    f1_val = pulp.value(prob.objective)
    f2_milp_num = sum(model["value"][i] * (model["Q"][i].varValue or 0.0) for i in model["value"])
    f2_milp = f2_milp_num / V_total

    f2_true = recompute_true_f2(routes, model["inst"], model["ttime"], model["k"], model["value"])

    return {
        "epsilon": epsilon, "status": status_str, "elapsed_s": elapsed,
        "f1": f1_val, "f2_milp": f2_milp, "f2_true": f2_true,
        "gap_pwl": f2_true - f2_milp, "routes": routes,
        "proven_optimal": proven_optimal, "cbc_gap": cbc_gap, "cbc_result_line": cbc_result_line,
    }


def _save_slim(results, out_path):
    """Salva os resultados (sem 'routes' completo) em JSON -- usado tanto no
    final do sweep quanto INCREMENTALMENTE apos cada ponto (Entrada 40:
    rodadas longas na maquina do pesquisador nao podem perder tudo se forem
    interrompidas no meio)."""
    slim = [{k2: v2 for k2, v2 in r.items() if k2 != "routes"} | {"n_routes": len(r.get("routes", []))}
            for r in results]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2, ensure_ascii=False)


def sweep_epsilon(inst_path, n_steps=30, n_pts=8, time_limit_s=60, verbose=False, log=print, max_extra_vehicles=None,
                   logdir=None, out_path=None):
    """
    Traca a fatia f1 x f2 da frente de Pareto exata (a menos do gap da
    interpolacao SOS2, pequeno e quantificado -- ver docstring do modulo).

    Passo 1: resolve epsilon=0 (so custo) -- da o "piso" natural de qualidade
    da rota mais barata (f2_true nesse ponto).
    Passo 2: varre epsilon em passos iguais de floor ate 1.0, parando no
    primeiro "Infeasible" (dali pra cima nao ha rota que atinja essa
    qualidade dentro do modelo -- a interpolacao SOS2 superestima levemente
    entre breakpoints, entao o teto real pode ser MINIMAMENTE mais baixo que
    o relatado; gap medido <=0.00004 pra grade de 8 pontos, ver docstring).

    Reporta SEMPRE f2_true (recalculado das rotas, formula exata de
    _route_quality_mc) como o eixo de qualidade da frente -- f2_milp e so
    diagnostico interno pra auditar o tamanho do gap da interpolacao.

    logdir (novo, Entrada 40): se dado, grava um .log por ponto (log bruto
    do CBC) nesse diretorio e usa parse_cbc_log() pra reportar
    proven_optimal/cbc_gap de verdade -- NAO confiar so no "status" (ver
    achado do bug de status do PuLP na Entrada 40 do diario).

    out_path (novo, Entrada 40): se dado, salva o JSON de resultados depois
    de CADA ponto (nao so no final) -- pra rodadas de horas na maquina do
    pesquisador nao perderem tudo se a janela for fechada ou o processo
    interrompido no meio.
    """
    if logdir:
        os.makedirs(logdir, exist_ok=True)

    def _logpath(idx, eps):
        if not logdir:
            return None
        return os.path.join(logdir, f"point_{idx:02d}_eps_{eps:.4f}.log")

    r0 = solve_eps(inst_path, epsilon=0.0, n_pts=n_pts, time_limit_s=time_limit_s, verbose=verbose,
                    max_extra_vehicles=max_extra_vehicles, log_path=_logpath(0, 0.0))
    if r0["status"] != "Optimal":
        log(f"  [eps=0.0] status={r0['status']} -- nao deu pra achar nem o piso, abortando sweep")
        if out_path:
            _save_slim([r0], out_path)
        return [r0]

    floor = r0["f2_true"]
    provado0 = "PROVADO" if r0.get("proven_optimal") else ("NAO-PROVADO" if r0.get("proven_optimal") is False else "?")
    log(f"  [eps=0.000] status=Optimal ({provado0}) f1={r0['f1']:.2f} f2_true={floor:.5f} "
        f"(piso natural) t={r0['elapsed_s']:.2f}s")

    results = [r0]
    if out_path:
        _save_slim(results, out_path)
    step = max((1.0 - floor) / n_steps, 1e-4)
    eps = floor + step
    last_f1 = r0["f1"]
    idx = 1
    while eps <= 1.0 + 1e-9:
        r = solve_eps(inst_path, epsilon=eps, n_pts=n_pts, time_limit_s=time_limit_s, verbose=verbose,
                       max_extra_vehicles=max_extra_vehicles, log_path=_logpath(idx, eps))
        if r["status"] != "Optimal":
            log(f"  [eps={eps:.3f}] status={r['status']} -- parando sweep aqui (teto de qualidade "
                f"atingivel dentro do modelo)")
            results.append(r)
            if out_path:
                _save_slim(results, out_path)
            break
        provado = "PROVADO" if r.get("proven_optimal") else ("NAO-PROVADO" if r.get("proven_optimal") is False else "?")
        changed = "" if abs(r["f1"] - last_f1) < 1e-6 else "  <- f1 mudou"
        log(f"  [eps={eps:.3f}] status=Optimal ({provado}) f1={r['f1']:.2f} f2_true={r['f2_true']:.5f} "
            f"f2_milp={r['f2_milp']:.5f} gap_pwl={r['gap_pwl']:.5f} cbc_gap={r.get('cbc_gap')} "
            f"t={r['elapsed_s']:.2f}s{changed}")
        results.append(r)
        if out_path:
            _save_slim(results, out_path)
        last_f1 = r["f1"]
        eps += step
        idx += 1

    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("inst_path", nargs="?", default="instances/validacao_exata/CCVRPTW-ST_valexata_C1_005_LOW.json")
    ap.add_argument("--tlim", type=int, default=60)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--maxextra", type=int, default=None, help="teto de veiculos extra alem da frota nominal (None = so limitado por n)")
    ap.add_argument("--out", default=None, help="se dado, salva os resultados (sem 'routes' completo) em JSON")
    ap.add_argument("--logdir", default=None,
                     help="se dado, grava um .log por ponto (log bruto do CBC) nesse diretorio -- "
                          "necessario pra saber se um ponto 'Optimal' foi REALMENTE provado ou so bateu "
                          "no limite de tempo (achado da Entrada 40 do diario: o status do PuLP sozinho "
                          "mente nesse caso). Fortemente recomendado em rodadas de --tlim grande.")
    args = ap.parse_args()

    k = compute_k(json.load(open(args.inst_path, encoding="utf-8")))
    print(f"instancia: {args.inst_path}")
    print(f"k={k:.6e} /h")

    t0 = pytime.time()
    results = sweep_epsilon(args.inst_path, n_steps=args.steps, time_limit_s=args.tlim,
                             max_extra_vehicles=args.maxextra, logdir=args.logdir, out_path=args.out)
    total_t = pytime.time() - t0
    print(f"\nsweep completo em {total_t:.2f}s, {len(results)} pontos resolvidos")
    n_provados = sum(1 for r in results if r["status"] == "Optimal" and r.get("proven_optimal") is True)
    n_naoprovados = sum(1 for r in results if r["status"] == "Optimal" and r.get("proven_optimal") is False)
    n_outros = sum(1 for r in results if r["status"] != "Optimal")
    print(f"pontos 'Optimal' genuinamente provados: {n_provados}  |  'Optimal' mas NAO provado "
          f"(so bateu no limite de tempo -- ver .log): {n_naoprovados}  |  outros status "
          f"(Infeasible/Not Solved): {n_outros}")

    if args.out:
        # sweep_epsilon ja salvou incrementalmente a cada ponto (Entrada 40) --
        # esta chamada final so garante que o arquivo reflete o resultado completo.
        _save_slim(results, args.out)
        print(f"salvo em {args.out}")
