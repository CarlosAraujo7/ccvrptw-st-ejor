# CCVRPTW-ST — Cold-chain vehicle routing under temperature and travel-time uncertainty

Code, instances, public input data and result summaries for the paper

> **Cold-chain vehicle routing under temperature and travel-time uncertainty: plan the quality, protect the punctuality**
> (submitted to the *European Journal of Operational Research*).

The paper studies a three-objective cold-chain vehicle routing problem with time windows (cost, value-weighted
product-quality loss under Arrhenius kinetics, CO2 emissions). It uses a route-dependent thermal model, departure
times as decisions, a chance constraint on punctuality and an optional multi-compartment vehicle. It compares four
ways to handle punctuality:
- deterministic planning;
- certification after the search;
- simulation-certified time buffers;
- a simheuristic with simulation inside the search.

It also proposes a simulation-based refinement of the chosen plan.

> Code comments and several file names are in Portuguese. A short glossary is at the end of this file.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Solver and experiment runner |
| `src/ccvrp_c.py` | Problem model, Split decoder, granular local search with route elimination, chance constraint, certification, buffered planning times |
| `src/nsga2_simheuristic.py` | NSGA-II and MOEA/D engines, instance loading, hypervolume |
| `src/thermal_route.py` | Route-dependent thermal model (door openings, pull-downs, Arrhenius kinetics) and scenario sampling |
| `src/instance_generator.py` | Generator of the Solomon-style benchmark instances and thermal profiles |
| `src/experiment_runner.py` | Runs methods × instances × seeds in parallel; method configurations in `ALGO_CONFIG` |
| `instances/generated/` | The 54 benchmark instances: families C1, C2, R1, R2, RC1, RC2; n = 25, 50, 100; LOW, MEDIUM, HIGH uncertainty |
| `instances/validacao_exata/` | The 4 small instances solved exactly (n = 5 to 20) |
| `instances/caso_sp/` | The 16 São Paulo instances built from public data (samples SP1–SP4; n = 50, 100; MEDIUM, HIGH) |
| `instances/caso_sp_pico/` | The SP1/SP2 instances planned with the morning-peak congestion (8 instances) |
| `pesquisa/` | Experiment and analysis scripts (see below) |
| `pesquisa/clima/` | Processed hourly ambient-temperature profiles of seven Brazilian capitals (NOAA ISD, 2011–2020) |
| `pesquisa/caso_sp/` | Pipeline of the São Paulo case and its public input data |
| `pesquisa/resultados_*/` | Summary outputs behind every number of the paper (per-run CSV files and text reports) |

### Instance format

Each instance is a JSON file with the fields `customers`, `depot`, `fleet`, `thermal_profile`, `co2_params` and
`horizon_minutes`. Each customer has demand, time window, service time, thermal category and order value. The São
Paulo instances also carry road-network matrices (`dist_matrix_km`, `time_matrix_min`) and the registry key of
each store.

## Requirements

- Python 3.12, with `pip install -r requirements.txt`: numpy, scipy, matplotlib; PyVRP 0.13.3 for the reference and
  external-solver experiments; PuLP for the exact MILP.
- The experiment drivers are PowerShell scripts (`pesquisa/*.ps1`). On Linux or macOS, run the `python ...` lines they
  contain one by one.

## Reproducing the paper

All commands are run from the repository root. Times refer to a 12-core desktop computer with 10 parallel runs.

1. **Smoke test** (one method, one size, one seed):
   ```
   python src/experiment_runner.py --algorithms C-SIMC --sizes 25 --uncertainty HIGH --seeds 1 --workers 1 --results-dir results_demo
   ```
2. **Main experiment** (8,370 runs plus the supplementary blocks, followed by all analyses and tables), about 33 h:
   ```
   powershell -ExecutionPolicy Bypass -File pesquisa/rodar_execucao_c_v2.ps1
   ```
3. **Simulation-based refinement** on the main experiment (test set):
   ```
   powershell -ExecutionPolicy Bypass -File pesquisa/rodar_pos_execucao.ps1
   ```
4. **São Paulo case** (1,120 + 560 runs), PyVRP pipeline, peak-congestion test and SIM+C with 100 scenarios:
   ```
   powershell -ExecutionPolicy Bypass -File pesquisa/caso_sp/rodar_caso_sp_v2.ps1
   powershell -ExecutionPolicy Bypass -File pesquisa/rodar_ciclo_20261015.ps1
   python pesquisa/caso_sp/simc100_sp.py --rodar --analisar --workers 10
   ```
5. **Run times and PyVRP reference** on an idle computer:
   ```
   powershell -ExecutionPolicy Bypass -File pesquisa/rodar_medicoes_maquina_livre.ps1
   ```

Every run is seeded. The out-of-sample evaluation uses 1,000 scenarios per instance, common to all methods and
drawn from a fixed seed (`pesquisa/analisar_c.py`). The scripts write the LaTeX tables and figures to
`pesquisa/latex_ejor_c/tables` and `pesquisa/latex_ejor_c/figures`.

## The São Paulo case from public data

The scripts in `pesquisa/caso_sp/` rebuild the instances in four steps:

1. `extrair_cnpj.py` keeps the active supermarkets and hypermarkets of the municipality (economic activity codes 4711-3/01 and 4711-3/02), with address and company size class, from the open CNPJ register of the Brazilian Federal Revenue Service (reference month 2026-10). The raw register (about 7 GB) is not included; the extracted file is `dados_brutos/cnpj/supermercados_sp_2026-10.csv`. Owner names, phone numbers and e-mails are not kept.
2. `geocodificar_cnpj.py` geocodes a random subset with Nominatim, under its usage policy (1 request/s, cached). A location is kept only with a matched house number. The cache and the result are included, so this step needs no new queries.
3. `gerar_caso_sp.py` samples the stores, stratified by size class, and builds the instances. Road matrices come from OSRM, using the included raw responses, scaled by the TomTom congestion level. Receiving windows and unloading times come from a field survey of 39 supermarkets (Oliveira et al., 2016, *Transportes* 24(2)). The product mix comes from IBGE household-acquisition data (POF 2017–2018, SIDRA table 2393, included).
4. `rodar_caso_sp.py` runs the methods; `tabela_caso_sp.py`, `estresse_pico.py`, `comparar_pico.py` and `simc100_sp.py` produce the reported analyses.

## Raw results

The raw outputs of every run (final fronts, certified probabilities, out-of-sample evaluations; about 550 MB) are
archived separately: **[Zenodo DOI to be added]**. The summaries in `pesquisa/resultados_*/` are enough to check
every table of the paper.

## Licenses

- **Code:** MIT License (`LICENSE`).
- **Data:** see `DATA_LICENSE.md`.
  - The benchmark instances and results are under CC BY 4.0.
  - The São Paulo instances and their input data contain data © OpenStreetMap contributors and are distributed under the Open Database License (ODbL 1.0).

## Citation

See `CITATION.cff`. The bibliographic data will be updated on publication.

## Glossary of Portuguese names

| Name | Meaning |
|---|---|
| pesquisa | research (scripts) |
| resultados | results |
| caso_sp | São Paulo case |
| pico | peak |
| rodar | run |
| analisar | analyze |
| refinamento | refinement |
| escolha do decisor | decision-maker's choice |
| conformidade | compliance |
| clima | climate |
| dados_brutos | raw data |
| validação exata | exact validation |
| relatório | report |
