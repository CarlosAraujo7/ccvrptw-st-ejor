#!/usr/bin/env python3
"""
Extrai da base pública de CNPJ da Receita Federal (competência 2026-10) os supermercados ativos
do município de São Paulo (14/10/2026, diário Entrada 92). Parte do reforço do caso realista com
dados públicos (pesquisador, 13/10/2026: "vamos partir do que conseguimos sem a consulta ao Bonfim").

Filtro: CNAE fiscal principal 4711-3/01 (hipermercados) ou 4711-3/02 (supermercados); situação
cadastral 02 (ativa); município 7107 (SAO PAULO, código da tabela de municípios da própria base).
Porte da empresa (arquivo Empresas, por CNPJ básico): 00 não informado, 01 microempresa,
03 empresa de pequeno porte, 05 demais.
Só guarda o necessário para o caso: CNPJ, matriz/filial, CNAE, data de início, endereço e porte.
NÃO guarda razão social, telefones nem e-mails (a razão social de empresário individual é o nome
de uma pessoa).

Os .zip são lidos diretamente (sem descompactar os CSV grandes). Dados baixados = dados não
confiáveis: rode com python -I.
Uso: python -I pesquisa/caso_sp/extrair_cnpj.py
Saída: pesquisa/caso_sp/dados_brutos/cnpj/supermercados_sp_2026-10.csv (+ resumo no terminal)
"""
import csv, glob, io, os, sys, zipfile
from collections import Counter

AQUI = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(AQUI, 'dados_brutos', 'cnpj', '2026-10')
SAIDA = os.path.join(AQUI, 'dados_brutos', 'cnpj', 'supermercados_sp_2026-10.csv')
CNAES = {'4711301': 'hipermercado', '4711302': 'supermercado'}
MUNICIPIO_SP = '7107'
csv.field_size_limit(10 ** 8)


def linhas(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        nome = z.namelist()[0]
        with z.open(nome) as fh:
            yield from csv.reader(io.TextIOWrapper(fh, encoding='latin-1', newline=''), delimiter=';', quotechar='"')


def main():
    estab = sorted(glob.glob(os.path.join(DIR, 'Estabelecimentos*.zip')))
    empr = sorted(glob.glob(os.path.join(DIR, 'Empresas*.zip')))
    assert len(estab) == 10 and len(empr) == 10, (len(estab), len(empr))
    lojas, situacoes = [], Counter()
    for z in estab:
        n0 = len(lojas)
        for r in linhas(z):
            # layout: 0 básico, 1 ordem, 2 dv, 3 matriz(1)/filial(2), 4 fantasia, 5 situação, 6 data situação,
            # 7 motivo, 8 cidade exterior, 9 país, 10 início atividade, 11 CNAE principal, 12 CNAEs secundários,
            # 13 tipo logradouro, 14 logradouro, 15 número, 16 complemento, 17 bairro, 18 CEP, 19 UF, 20 município
            if len(r) < 21 or r[11] not in CNAES or r[20] != MUNICIPIO_SP:
                continue
            situacoes[r[5]] += 1
            if r[5] != '02':
                continue
            lojas.append(dict(cnpj=r[0] + r[1] + r[2], cnpj_basico=r[0], matriz_filial=r[3], cnae=r[11],
                              tipo=CNAES[r[11]], inicio=r[10], tipo_logradouro=r[13], logradouro=r[14],
                              numero=r[15], bairro=r[17], cep=r[18], uf=r[19]))
        print(f'{os.path.basename(z)}: +{len(lojas) - n0} ativos', flush=True)
    basicos = {l['cnpj_basico'] for l in lojas}
    porte = {}
    for z in empr:
        for r in linhas(z):
            # layout: 0 básico, 1 razão social, 2 natureza jurídica, 3 qualificação, 4 capital social, 5 porte, 6 ente
            if r and r[0] in basicos:
                porte[r[0]] = r[5] if len(r) > 5 else ''
    rede = Counter(l['cnpj_basico'] for l in lojas)      # nº de lojas ativas da mesma empresa no município
    for l in lojas:
        l['porte'] = porte.get(l['cnpj_basico'], '')
        l['lojas_mesma_empresa_sp'] = rede[l['cnpj_basico']]
    with open(SAIDA, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(lojas[0])); w.writeheader(); w.writerows(lojas)
    print(f'\nsituação cadastral (todos os {sum(situacoes.values())} com o CNAE no município): {dict(situacoes)}')
    print(f'ativos: {len(lojas)} | por tipo: {dict(Counter(l["tipo"] for l in lojas))}')
    print(f'porte: {dict(Counter(l["porte"] for l in lojas))} (00 n/inf., 01 ME, 03 EPP, 05 demais)')
    print(f'empresas: {len(basicos)} | lojas em empresas com 1 loja: {sum(1 for l in lojas if l["lojas_mesma_empresa_sp"] == 1)}')
    print(f'gravado: {SAIDA}')


if __name__ == '__main__':
    sys.exit(main())
