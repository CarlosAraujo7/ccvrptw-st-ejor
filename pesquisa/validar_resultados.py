#!/usr/bin/env python3
"""
Valida as execuções gravadas antes de uma retomada (13/10/2026, diário Entrada 92).
Uma execução só conta como concluída se _metrics.json (gravado por último, e consultado
pelo --resume) e _solutions.json forem JSON válidos e o motor for o atual
(busca_local_delta e eliminar_rotas = true). As inválidas vão para uma pasta de
quarentena (nada é apagado) e serão refeitas pela retomada.
Uso: python pesquisa/validar_resultados.py [pastas...]   (padrão: só as pastas de saída da execução v2)
"""
import glob, json, os, shutil, sys, time

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def main():
    # SÓ as pastas de saída da execução v2 (lista explícita). Um glob amplo pegou por engano
    # results_c_piloto (resultados antigos de piloto, sem a marca do motor novo) em 13/10/2026;
    # os 2.680 arquivos foram devolvidos intactos (diário Entrada 92).
    ATUAIS = ('results_c', 'results_c_b2', 'results_c_b2_sens', 'results_c_extra', 'results_c_cor', 'results_c_sp')
    pastas = sys.argv[1:] or [os.path.join(BASE, d) for d in ATUAIS if os.path.isdir(os.path.join(BASE, d))]
    q = os.path.join(BASE, 'results_quarentena_' + time.strftime('%Y%m%d_%H%M'))
    ok = ruins = 0
    for pasta in pastas:
        for m in glob.glob(os.path.join(pasta, '*', '*_metrics.json')):
            pre = m[:-len('_metrics.json')]
            try:
                mm = json.load(open(m, encoding='utf-8')); json.load(open(pre + '_solutions.json', encoding='utf-8'))
                assert mm.get('busca_local_delta') is True and mm.get('eliminar_rotas') is True
                ok += 1
            except Exception as e:
                ruins += 1
                destino = os.path.join(q, os.path.basename(pasta), os.path.basename(os.path.dirname(pre)))
                os.makedirs(destino, exist_ok=True)
                for f in glob.glob(pre + '_*'):
                    shutil.move(f, os.path.join(destino, os.path.basename(f)))
                print(f'quarentena: {os.path.basename(pre)} ({type(e).__name__})')
        parciais = [f for f in glob.glob(os.path.join(pasta, '*', '*_solutions.json'))
                    if not os.path.exists(f.replace('_solutions.json', '_metrics.json'))]
        print(f'{os.path.basename(pasta)}: parciais sem métricas (serão refeitas) = {len(parciais)}')
    print(f'válidas: {ok} | em quarentena: {ruins}')


if __name__ == '__main__':
    main()
