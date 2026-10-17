#!/usr/bin/env python3
"""
Copia para repositorio_github/ccvrptw-st os arquivos do projeto acrescentados ou alterados nas melhorias de
16-17/10/2026 (tempos dependentes do horário, caso de Atenas, teoria do buffer, degrau nominal; diário
Entradas 101-107). Lista explícita: nada de dados brutos (TomTom, CERTH, NOAA), backups ou logs.
Uso: python pesquisa/sincronizar_repositorio.py
"""
import os, shutil

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
REPO = os.path.join(BASE, 'repositorio_github', 'ccvrptw-st')
ARQUIVOS = [
    'src/ccvrp_c.py', 'src/thermal_route.py', 'src/tempo_dependente.py',
    'pesquisa/teste_tempo_dependente.py', 'pesquisa/validar_teoria_buffer.py', 'pesquisa/analisar_degrau_nominal.py',
    'pesquisa/pipeline_pyvrp_buffer.py', 'pesquisa/sincronizar_repositorio.py',
    'pesquisa/caso_sp/rodar_caso_sp.py', 'pesquisa/caso_sp/tabela_caso_sp.py', 'pesquisa/caso_sp/gerar_caso_sp_td.py',
    'pesquisa/caso_sp/comparar_td.py', 'pesquisa/caso_sp/extrair_tomtom.py', 'pesquisa/caso_sp/rodar_caso_sp_td.ps1',
    'pesquisa/caso_certh/gerar_caso_atenas.py', 'pesquisa/caso_certh/rodar_caso_atenas.ps1',
    'pesquisa/clima/processar_isd.py', 'pesquisa/clima/perfis_16716199999.json', 'pesquisa/clima/resumo_clima_atenas.txt',
    # calibração (17/10/2026, Entradas 108-110)
    'src/experiment_runner.py', 'pesquisa/rodar_acal.py', 'pesquisa/avaliar_acal.py', 'pesquisa/resumo_acal.py',
    'pesquisa/rodar_acal_teste.ps1', 'pesquisa/calibracao_preditiva_base.py', 'pesquisa/calibracao_preditiva.py',
    'pesquisa/calibracao_preditiva_dev.py', 'pesquisa/refinamento_simulacao.py',
]
PASTAS = ['instances/caso_sp_td', 'instances/caso_sp_media_operacao', 'instances/caso_atenas',
          'pesquisa/resultados_caso_sp_td', 'pesquisa/resultados_caso_sp_media', 'pesquisa/resultados_caso_atenas',
          'pesquisa/resultados_teoria_buffer', 'pesquisa/resultados_degrau_nominal', 'pesquisa/resultados_acal',
          'pesquisa/resultados_calibracao_preditiva']
IGNORAR = ('.bak', '__pycache__', 'log_')


def main():
    n = 0
    for a in ARQUIVOS:
        o = os.path.join(BASE, a); d = os.path.join(REPO, a)
        os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy2(o, d); n += 1
    for p in PASTAS:
        for raiz, _, fs in os.walk(os.path.join(BASE, p)):
            if any(x in raiz for x in IGNORAR):
                continue
            for f in fs:
                if any(x in f for x in IGNORAR):
                    continue
                o = os.path.join(raiz, f); d = os.path.join(REPO, os.path.relpath(o, BASE))
                os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy2(o, d); n += 1
    print(n, 'arquivos copiados para', REPO)


if __name__ == '__main__':
    main()
