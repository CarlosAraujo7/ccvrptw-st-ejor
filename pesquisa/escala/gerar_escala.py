#!/usr/bin/env python3
"""
Instâncias grandes para o teste de escala (17/10/2026, diário Entrada 114; análise de fraquezas A3).
Mesmo gerador e mesma construção das 54 instâncias do benchmark (src/instance_generator.generate_instance; seis
famílias no estilo de Solomon: C1, C2, R1, R2, RC1, RC2), com n = 200 e 400 clientes e incerteza MEDIUM e HIGH:
24 instâncias, com sementes novas (7000 + ...), em instances/escala/.
Uso: python pesquisa/escala/gerar_escala.py
"""
import json, os, sys
from dataclasses import asdict

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(BASE, 'src'))
import instance_generator as ig   # noqa: E402

SAIDA = os.path.join(BASE, 'instances', 'escala')


def main():
    os.makedirs(SAIDA, exist_ok=True)
    k = 0
    for fi, fam in enumerate(('C', 'R', 'RC')):
        for var in (1, 2):
            for n in (200, 400):
                for ui, unc in enumerate(('medium', 'high')):
                    seed = 7000 + 100 * fi + 10 * var + (n // 100) + ui
                    inst = ig.generate_instance(fam, n, unc, seed, variant=var)
                    arq = os.path.join(SAIDA, f'{inst.name}.json')
                    if os.path.exists(arq):
                        continue
                    json.dump(asdict(inst), open(arq, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
                    k += 1
                    print(f'{inst.name}: frota {inst.fleet["num_vehicles"]}, capacidade {inst.fleet["capacity_kg"]} kg, '
                          f'demanda {sum(c["demand_kg"] for c in inst.customers):.0f} kg', flush=True)
    print(k, 'instâncias gravadas em', SAIDA)


if __name__ == '__main__':
    main()
