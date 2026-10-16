#!/usr/bin/env python3
"""
Figuras do manuscrito do Caminho C (05/10/2026, diário Entrada 80), geradas dos
resultados. Paleta categórica de referência (slots 1-3, validada para todos os
pares e para daltonismo); o slot 3 fica abaixo de 3:1 de contraste, então os
valores vão escritos nas barras e a tabela correspondente existe no texto.
Saída: pesquisa/latex_ejor_c/figures/*.pdf
"""
import csv, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT = os.path.join(BASE, 'pesquisa', 'latex_ejor_c', 'figures')
os.makedirs(OUT, exist_ok=True)
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6, 'figure.dpi': 200})

R = list(csv.DictReader(open(os.path.join(BASE, 'pesquisa', 'resultados_c', 'por_execucao.csv'), encoding='utf-8')))
for r in R:
    r['cm'] = float(r['custo_min_cumpre']) if r['custo_min_cumpre'] not in ('', 'nan') else np.nan

# ── Figura 1: % de execuções com plano que cumpre α=0,90, por método e nível ──
metodos = [('C-DET', 'DET'), ('C-DETC', 'DET+C'), ('C-BUF90', 'BUF$_{0.90}$'), ('C-BUF95C', 'BUF$_{0.95}$+C'),
           ('C-CAL-BEST', 'CAL'), ('C-SIMC', 'SIM+C')]
niveis = [('LOW', 'Low', S1), ('MEDIUM', 'Medium', S2), ('HIGH', 'High', S3)]
fig, ax = plt.subplots(figsize=(6.6, 2.6))
w = 0.26
for j, (u, rot, cor) in enumerate(niveis):
    vals = []
    for a, _ in metodos:
        rr = [r for r in R if r['algorithm'] == a and r['uncertainty'] == u]
        vals.append(100 * np.mean([np.isfinite(r['cm']) for r in rr]))
    x = np.arange(len(metodos)) + (j - 1) * (w + 0.02)
    ax.bar(x, vals, width=w, color=cor, label=rot, zorder=3, edgecolor=SURF, linewidth=0.8)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 1.2, f'{v:.0f}', ha='center', va='bottom', fontsize=6.5, color=INK2)
ax.set_xticks(np.arange(len(metodos)))
ax.set_xticklabels([m for _, m in metodos], color=INK)
ax.set_ylabel('Runs with a compliant plan (%)')
ax.set_ylim(0, 112)
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(axis='y', color=GRID, linewidth=0.6, zorder=0)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.legend(title='Travel-time uncertainty', ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.22),
          fontsize=7.5, title_fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_conformidade.pdf'), bbox_inches='tight')

# ── Figura 2: custo extra por instância, relativo a SIM+C ───────────────────
def por_inst(a):
    d = defaultdict(list)
    for r in R:
        if r['algorithm'] == a and np.isfinite(r['cm']):
            d[r['instance']].append(r['cm'])
    return {k: np.mean(v) for k, v in d.items()}


sim = por_inst('C-SIMC')
fig, ax = plt.subplots(figsize=(6.6, 2.5))
for a, rot, cor, mk in (('C-DETC', 'DET+C (certify a deterministic plan)', S2, 'o'),
                        ('C-BUF95C', 'BUF$_{0.95}$+C (buffered plan, certified)', S1, 's')):
    x = por_inst(a)
    d = np.sort([100 * (x[k] / sim[k] - 1) for k in x if k in sim])[::-1]
    ax.plot(np.arange(1, len(d) + 1), d, marker=mk, markersize=3.6, linewidth=1.0, color=cor, label=rot, zorder=3,
            markeredgecolor=SURF, markeredgewidth=0.5)
    off = (6, -2) if a == 'C-DETC' else (10, 9)
    ax.annotate(f'max {d[0]:+.0f}%', (1, d[0]), xytext=off, textcoords='offset points', fontsize=6.5, color=INK2,
                arrowprops=None if a == 'C-DETC' else dict(arrowstyle='-', color=INK2, linewidth=0.5))
ax.axhline(0, color=INK2, linewidth=0.6, zorder=2)
ax.set_xlabel('Instances, sorted by cost difference')
ax.set_ylabel('Cost vs SIM+C (%)')
ax.grid(axis='y', color=GRID, linewidth=0.6, zorder=0)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=7.5, loc='upper right')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_cauda_custo.pdf'), bbox_inches='tight')
print('figuras:', sorted(os.listdir(OUT)))
