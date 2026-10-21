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
| `src/ccvrp_c.py` (`buffer_adapt`) | Calibration of the buffer level within a single search (ACAL); off by default |
| `src/tempo_dependente.py` | Time-dependent (FIFO, Ichoua–Gendreau–Potvin) travel times and arc-specific travel-time uncertainty; used only by instances with a `td` block |
| `instances/generated/` | The 54 benchmark instances: families C1, C2, R1, R2, RC1, RC2; n = 25, 50, 100; LOW, MEDIUM, HIGH uncertainty |
| `instances/validacao_exata/` | The 4 small instances solved exactly (n = 5 to 20) |
| `instances/caso_sp/` | The 16 São Paulo instances built from public data (samples SP1–SP4; n = 50, 100; MEDIUM, HIGH) |
| `instances/caso_sp_pico/` | The SP1/SP2 instances planned with the morning-peak congestion (8 instances) |
| `instances/caso_sp_td/` | The 16 São Paulo instances with time-dependent travel times (hourly weekday congestion profile) |
| `instances/caso_sp_media_operacao/` | The 16 São Paulo instances with the mean congestion of the operating hours (static) |
| `instances/escala/` | The 24 scale instances (C1-RC2; n = 200 and 400; MEDIUM and HIGH), same generator as the benchmark |
| `instances/caso_atenas/` | The 9 Athens instances built from the operational data of a pharmaceutical distributor |
| `pesquisa/` | Experiment and analysis scripts (see below) |
| `pesquisa/clima/` | Processed hourly ambient-temperature profiles of seven Brazilian capitals (NOAA ISD, 2011–2020) |
| `pesquisa/caso_sp/` | Pipeline of the São Paulo case and its public input data |
| `pesquisa/caso_certh/` | Pipeline of the Athens case (operational data; the raw data are downloaded from Zenodo) |
| `pesquisa/validar_teoria_buffer.py` | Empirical check of the properties of buffered plans (Lemma 1, Propositions 2–4 of the paper) |
| `pesquisa/teste_tempo_dependente.py` | Tests of the time-dependent and arc-specific travel-time model |
| `pesquisa/resultados_*/` | Summary outputs behind every number of the paper (per-run CSV files and text reports) |

### Instance format

Each instance is a JSON file with the fields `customers`, `depot`, `fleet`, `thermal_profile`, `co2_params` and
`horizon_minutes`. Each customer has demand, time window, service time, thermal category and order value. The São
Paulo instances also carry road-network matrices (`dist_matrix_km`, `time_matrix_min`) and the registry key of
each store.

Instances with a `td` block use `time_matrix_ff_min` (free-flow minutes) with the hourly factors
`td.fator_hora` from the clock hour `td.hora_inicio`; an optional `td.cv_arco` gives the coefficient of variation of
each arc, and `estacao_clima` selects the climate station. All other instances follow the static code path unchanged.

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
6. **Time-dependent travel times** (São Paulo, 2 × 1,120 runs, then the comparison of planning models):
   ```
   python pesquisa/teste_tempo_dependente.py
   powershell -ExecutionPolicy Bypass -File pesquisa/caso_sp/rodar_caso_sp_td.ps1
   python pesquisa/caso_sp/comparar_td.py
   ```
7. **Athens case** (630 runs; download the data first, see below):
   ```
   powershell -ExecutionPolicy Bypass -File pesquisa/caso_certh/rodar_caso_atenas.ps1
   ```
8. **Properties of buffered plans** (Section 4.3 of the paper; Table S9):
   ```
   python pesquisa/validar_teoria_buffer.py
   ```
9. **Calibration within a single search** (ACAL, Section 6.12; Table S19; 114 development and 950 test runs):
   ```
   python pesquisa/rodar_acal.py --fase dev --workers 10
   powershell -ExecutionPolicy Bypass -File pesquisa/rodar_acal_teste.ps1
   python pesquisa/resumo_acal.py
   ```
   `pesquisa/calibracao_preditiva*.py` document a rejected alternative (predicting the buffer level before the
   search with constructive pilots), which did not beat a fixed buffer on the development set.

Every run is seeded. The out-of-sample evaluation uses 1,000 scenarios per instance, common to all methods and
drawn from a fixed seed (`pesquisa/analisar_c.py`). The scripts write the LaTeX tables and figures to
`pesquisa/latex_ejor_c/tables` and `pesquisa/latex_ejor_c/figures`.

## The São Paulo case from public data

The scripts in `pesquisa/caso_sp/` rebuild the instances in four steps:

1. `extrair_cnpj.py` keeps the active supermarkets and hypermarkets of the municipality (economic activity codes 4711-3/01 and 4711-3/02), with address and company size class, from the open CNPJ register of the Brazilian Federal Revenue Service (reference month 2026-10). The raw register (about 7 GB) is not included; the extracted file is `dados_brutos/cnpj/supermercados_sp_2026-10.csv`. Owner names, phone numbers and e-mails are not kept.
2. `geocodificar_cnpj.py` geocodes a random subset with Nominatim, under its usage policy (1 request/s, cached). A location is kept only with a matched house number. The cache and the result are included, so this step needs no new queries.
3. `gerar_caso_sp.py` samples the stores, stratified by size class, and builds the instances. Road matrices come from OSRM, using the included raw responses, scaled by the TomTom congestion level. Receiving windows and unloading times come from a field survey of 39 supermarkets (Oliveira et al., 2016, *Transportes* 24(2)). The product mix comes from IBGE household-acquisition data (POF 2017–2018, SIDRA table 2393, included).
4. `rodar_caso_sp.py` runs the methods; `tabela_caso_sp.py`, `estresse_pico.py`, `comparar_pico.py` and `simc100_sp.py` produce the reported analyses.

## Time-dependent travel times in São Paulo

`pesquisa/caso_sp/extrair_tomtom.py` reads the hourly weekday congestion profile of São Paulo from the public page of
the TomTom Traffic Index 2025 (to be saved by the user; the page is not redistributed). `gerar_caso_sp_td.py` builds the
time-dependent and operating-mean versions of the 16 instances from the OSRM responses. Following Ichoua, Gendreau
and Potvin (2003), the speed of every arc is the free-flow speed divided by `1 + c(h)/100` in clock hour `h`.

## The Athens case from operational data

The data are those of Vrani et al. (2025), *Data in Brief* 61:111762, doi:10.1016/j.dib.2025.111762, published on
Zenodo (doi:10.5281/zenodo.15672291, CC BY 4.0). Download `data.zip` to `pesquisa/caso_certh/dados_brutos/zenodo_15672291/`,
extract it to `extraido/` and run `python -I pesquisa/caso_certh/gerar_caso_atenas.py`. Each arc gets the PERT mean of
its optimistic, most likely and pessimistic travel times as nominal time and the PERT coefficient of variation in its
own lognormal model. Fleet, product and costs are assumptions (see the paper). The climate of Athens comes from NOAA ISD
station 16716199999 (`python pesquisa/clima/processar_isd.py <folder> resumo_clima_atenas.txt`).

## Observed travel times

`pesquisa/tempos_observados/` checks the travel-time model against the public trip records of the New York City Taxi and
Limousine Commission (yellow taxis, March and October 2024; Section 6.13 of the paper). Download
`yellow_tripdata_2024-03.parquet`, `yellow_tripdata_2024-10.parquet` and `taxi_zone_lookup.csv` from the TLC trip record
page (https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) to `pesquisa/tempos_observados/dados_brutos/nyc_tlc/`,
install `pyarrow`, and run `python -I pesquisa/tempos_observados/analisar_tlc.py`, then `reavaliar_observado.py`,
`recertificar_observado.py` and `tabela_observado.py`. The standardized samples used for resampling are included in
`pesquisa/tempos_observados/resultados/amostras_tlc.npz`.

The truck trips come from the DT-CARGO data (Balke & Adenaw, 2023): download `input/public/tracks.csv` from
https://github.com/TUMFTM/dt-cargo to `pesquisa/tempos_observados/dados_brutos/dt_cargo/` and run
`python -I pesquisa/tempos_observados/analisar_dtcargo.py` and `reavaliar_caminhoes.py`.

## Larger instances

`pesquisa/escala/` runs the scale test of Section 6.14 (24 instances with 200 and 400 customers in `instances/escala/`,
ten seeds): `rodar_escala.ps1` runs seeds 0-2 and `pesquisa/rodar_ciclo_b3b4.ps1` (or `rodar_ciclo_b3b4.cmd`) seeds 3-9:
SIM+C, the four buffer rungs of CAL and ACAL with our engine (1,440 runs), the PyVRP pipeline (720 solves, 4 and 8 min
each) and the refinement under both certification rules; `python pesquisa/escala/analisar_escala.py` writes
`pesquisa/resultados_escala/relatorio_escala.txt`, `resumo_escala.json` and Table 7 of the paper (the `*_3sementes*`
files keep the first analysis with seeds 0-2). The design and the hypotheses were fixed before the runs.

## Certification against a lower confidence bound

`python pesquisa/rodar_regra_lcb.py` repeats the refinement in all five sets with the plan chosen by the stricter rule
(cheapest plan with certified punctuality at least α + 1.645·√(α(1−α)/500); option `--regra lcb` of
`refinamento_simulacao.py`, whose default path is unchanged) and writes `refinamento_<start>_lcb.json` next to the
point-rule results. `python pesquisa/regra_lcb.py` compares both rules (Table S18 of the supplementary material;
`pesquisa/resultados_regra_lcb/`), and `python pesquisa/figura_sintese.py` draws Figure 1 of the paper.

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
| tempo_dependente | time-dependent |
| caso_atenas, caso_certh | Athens case (CERTH data) |
| media_operacao | mean over the operating hours |
| refinamento | refinement |
| escolha do decisor | decision-maker's choice |
| conformidade | compliance |
| clima | climate |
| dados_brutos | raw data |
| validação exata | exact validation |
| relatório | report |
