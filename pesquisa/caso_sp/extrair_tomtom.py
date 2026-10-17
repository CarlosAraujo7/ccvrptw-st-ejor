#!/usr/bin/env python3
"""
Perfil horário de congestionamento de São Paulo (TomTom Traffic Index 2025), 16/10/2026, diário Entrada 101.

Lê a página pública da cidade (dados_brutos/tomtom/pagina_sao_paulo.html, baixada em 16/10/2026) e extrai o
bloco "weekHourData" do componente "Driving patterns": para cada dia da semana e hora, o nível de
congestionamento c (%, tempo extra de viagem sobre o fluxo livre), a velocidade média v e a velocidade de
fluxo livre fv (km/h), para a cidade ("center") e para a região metropolitana ("metro").
Gera o fator de tempo de viagem por hora, f(h) = 1 + c(h)/100, como a média dos dias úteis (segunda a sexta)
da cidade, que é o perfil usado nos tempos dependentes do horário do caso SP.
Dados lidos como não confiáveis: rode com python -I.
Uso: python -I pesquisa/caso_sp/extrair_tomtom.py
Saída: dados_brutos/tomtom/perfil_horario_sao_paulo_2025.json
"""
import html, json, os, re

AQUI = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(AQUI, 'dados_brutos', 'tomtom')


def dec(v):
    """Desserializa o formato de props das ilhas Astro ([0, valor] ou [1, lista])."""
    if isinstance(v, list) and len(v) == 2 and isinstance(v[0], int):
        t, x = v
        if t == 0:
            return {k: dec(w) for k, w in x.items()} if isinstance(x, dict) else x
        if t == 1:
            return [dec(w) for w in x]
        return x
    if isinstance(v, dict):
        return {k: dec(w) for k, w in v.items()}
    return v


def main():
    s = open(os.path.join(D, 'pagina_sao_paulo.html'), encoding='utf-8').read()
    m = re.search(r'<astro-island[^>]*component-url="[^"]*DrivingPatterns[^"]*"[^>]*props="([^"]*)"', s)
    assert m, 'componente DrivingPatterns não encontrado'
    props = json.loads(html.unescape(m.group(1)))
    W = dec(props['weekHourData'])
    dias_uteis = ('MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY')
    perfil = {}
    for area in ('center', 'metro'):
        por_hora = {h: [] for h in range(24)}
        for r in W[area]:
            dia, h = r['time'].split('-')
            if dia in dias_uteis:
                por_hora[int(h)].append(r['c'])
        perfil[area] = {str(h): round(sum(v) / len(v), 2) for h, v in por_hora.items()}
    fator = {h: round(1 + c / 100, 4) for h, c in perfil['center'].items()}
    media = sum(float(perfil['center'][str(h)]) for h in range(24)) / 24
    out = dict(fonte='TomTom Traffic Index 2025, São Paulo, Driving patterns (página pública, baixada em 16/10/2026)',
               definicao='c = congestionamento (%): tempo extra de viagem sobre o fluxo livre; fator = 1 + c/100',
               congestionamento_dias_uteis_cidade=perfil['center'], congestionamento_dias_uteis_metro=perfil['metro'],
               fator_tempo_cidade=fator, media_24h_dias_uteis_cidade=round(media, 2), bruto=W)
    json.dump(out, open(os.path.join(D, 'perfil_horario_sao_paulo_2025.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('congestionamento médio dos dias úteis (cidade), por hora:')
    print('  ' + '  '.join(f'{h:02d}h {perfil["center"][str(h)]:5.1f}' for h in range(24)))
    print(f'média 24 h dos dias úteis: {media:.1f}%  (média anual publicada, todos os dias: 58,5%)')


if __name__ == '__main__':
    main()
