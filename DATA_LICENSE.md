# Data licenses

| Data | License | Attribution |
|---|---|---|
| Benchmark instances (`instances/generated`, `instances/validacao_exata`) and result summaries (`pesquisa/resultados_*`) | CC BY 4.0 | The authors of the paper |
| Climate profiles (`pesquisa/clima/perfis_*.json`) | CC BY 4.0, derived from public-domain data | NOAA Integrated Surface Database (ISD), stations 83780099999 (São Paulo), 83746099999 (Rio de Janeiro), 82398099999 (Fortaleza), 82899099999 (Recife), 82111099999 (Manaus), 83971099999 (Porto Alegre), 83840099999 (Curitiba), 16716199999 (Athens), 2011–2020 |
| São Paulo instances (`instances/caso_sp`, `instances/caso_sp_pico`), OSRM responses and Nominatim cache (`pesquisa/caso_sp/dados_brutos/osrm`, `.../nominatim`) | Open Database License (ODbL 1.0) | © OpenStreetMap contributors; routes computed with OSRM; geocoding by Nominatim |
| Athens instances (`instances/caso_atenas`) | CC BY 4.0 | Derived from Vrani, Apostolidis, Kapoutsis & Kosmatopoulos (2025), *Data in Brief* 61:111762, data on Zenodo, doi:10.5281/zenodo.15672291 (CC BY 4.0); fleet, product and costs are assumptions of this study |
| Time-dependent São Paulo instances (`instances/caso_sp_td`, `instances/caso_sp_media_operacao`) | Open Database License (ODbL 1.0) | As the São Paulo instances; hourly congestion factors quoted from the TomTom Traffic Index 2025 |
| Observed travel-time samples and summaries (`pesquisa/tempos_observados/resultados`) | Derived from public data | New York City Taxi and Limousine Commission, TLC Trip Record Data (yellow taxis, 2024); the raw records are not redistributed |
| Store register extract (`pesquisa/caso_sp/dados_brutos/cnpj/*.csv`) | Open government data | Receita Federal do Brasil, open CNPJ data, reference month 2026-10 |
| Household food acquisition (`pesquisa/caso_sp/dados_brutos/ibge`) | Open government data | IBGE, Pesquisa de Orçamentos Familiares 2017–2018, SIDRA table 2393 |

The congestion levels (58.5% average and 88.2% in the morning rush hour, São Paulo, 2025) and the 24 hourly
weekday congestion levels used by the time-dependent instances are quoted from the public page of the TomTom Traffic
Index; the page itself is not redistributed. The receiving windows and unloading times are frequencies published by Oliveira et al.
(2016), *Transportes* 24(2):53–67, doi:10.14295/transportes.v24i2.1058.
