#!/usr/bin/env python3
"""
Figura-síntese do artigo (18/10/2026, diário Entrada 117, ciclo B2): nos cinco conjuntos, diferença de custo e de perda
de qualidade de cada método contra o SIM+C (Hodges-Lehmann por instância) sob a regra pontual (cheio) e a regra LCB
(vazado), e o cumprimento dos planos escolhidos. Lê pesquisa/resultados_regra_lcb/resumo_regra_lcb.json (regra_lcb.py).
Uso: python pesquisa/figura_sintese.py
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT = os.path.join(BASE, 'pesquisa', 'latex_ejor_c', 'figures')
S1, S2, S3, S4 = '#2a78d6', '#1baf7a', '#eb6834', '#8a8984'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6, 'figure.dpi': 200})
R = json.load(open(os.path.join(BASE, 'pesquisa', 'resultados_regra_lcb', 'resumo_regra_lcb.json'), encoding='utf-8'))
CONJ = [('Benchmark', '(a) Benchmark'), ('São Paulo', '(b) São Paulo'),
        ('São Paulo, time-dependent', '(c) São Paulo, time-dependent'), ('Athens', '(d) Athens'),
        ('Scale', '(e) 200 and 400 customers')]
MET = [('BUF0.95+C', 'BUF$_{0.95}$+C', S4, 's'), ('CAL+R', 'CAL+R', S1, 'o'), ('ACAL+R', 'ACAL+R', S2, 'D'),
       ('PyVRP, CAL+R', 'PyVRP, CAL+R', S3, '^')]


def hl(conj, regra, m, medida):
    for r in R[conj]['comp'].get(f'{regra}|{m}', []):
        if r['medida'] == medida:
            return r['hl']
    return None


fig, axs = plt.subplots(2, 3, figsize=(6.7, 4.3))
for ax, (c, tit) in zip(axs.flat, CONJ):
    ax.axhline(0, color=INK2, lw=0.5, ls=(0, (3, 2)), zorder=1); ax.axvline(0, color=INK2, lw=0.5, ls=(0, (3, 2)), zorder=1)
    xs, ys = [0.0], [0.0]
    for m, rot, cor, mk in MET:
        p = [(hl(c, g, m, 'custo (%)'), hl(c, g, m, 'perda (p.p.)')) for g in ('pontual', 'lcb')]
        if p[0][0] is None:
            continue
        ax.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], color=cor, lw=0.7, zorder=2)
        ax.scatter(*p[0], s=26, marker=mk, color=cor, edgecolor=cor, zorder=3)
        ax.scatter(*p[1], s=26, marker=mk, facecolor=SURF, edgecolor=cor, linewidth=1.0, zorder=3)
        xs += [q[0] for q in p]; ys += [q[1] for q in p]
    ax.scatter(0, 0, s=30, marker='X', color=INK, zorder=4)
    dx = max(0.3, 0.12 * (max(xs) - min(xs))); dy = max(0.1, 0.12 * (max(ys) - min(ys)))
    ax.set_xlim(min(xs) - dx, max(xs) + dx); ax.set_ylim(min(ys) - dy, max(ys) + dy)
    ax.fill_between([min(xs) - dx, 0], min(ys) - dy, 0, color=S2, alpha=0.06, zorder=0, lw=0)
    ax.set_title(tit, fontsize=8, color=INK, loc='left')
    ax.grid(color=GRID, lw=0.5, zorder=0)
    ax.tick_params(labelsize=7)
for ax in axs[1, :2]:
    ax.set_xlabel('Cost vs SIM+C (%)')
for ax in axs[:, 0]:
    ax.set_ylabel('Quality loss vs SIM+C (p.p.)')
# (f) cumprimento
ax = axs[1, 2]
nomes = ['Bench.', 'SP', 'SP-TD', 'Ath.', 'Scale']
MC = [('SIM+C', 'SIM+C', INK, 'X')] + [x for x in MET if x[0] != 'BUF0.95+C']
for k, (c, _) in enumerate(CONJ):
    for j, (m, rot, cor, mk) in enumerate(MC):
        d = R[c]['metodos'].get(m)
        if not d:
            continue
        x = k + (j - 1.5) * 0.17
        ax.plot([x, x], [d['cumpre_pontual'], d['cumpre_lcb']], color=cor, lw=0.7, zorder=2)
        ax.scatter(x, d['cumpre_pontual'], s=20, marker=mk, color=cor, zorder=3)
        ax.scatter(x, d['cumpre_lcb'], s=20, marker=mk, facecolor=SURF, edgecolor=cor, linewidth=1.0, zorder=3)
ax.axhline(95, color=INK2, lw=0.5, ls=(0, (3, 2)))
ax.set_xticks(range(len(nomes))); ax.set_xticklabels(nomes, fontsize=6.5); ax.set_xlim(-0.5, len(nomes) - 0.5)
ax.set_ylabel('Chosen plans complying (%)'); ax.set_title('(f) Compliance', fontsize=8, color=INK, loc='left')
ax.grid(axis='y', color=GRID, lw=0.5, zorder=0); ax.tick_params(labelsize=7)
leg = [Line2D([], [], marker=mk, color=cor, lw=0, markersize=5, label=rot) for _, rot, cor, mk in MET]
leg += [Line2D([], [], marker='X', color=INK, lw=0, markersize=5, label='SIM+C'),
        Line2D([], [], marker='o', color=INK2, lw=0, markersize=5, label='point rule'),
        Line2D([], [], marker='o', markerfacecolor=SURF, markeredgecolor=INK2, lw=0, markersize=5, label='LCB rule')]
fig.legend(handles=leg, loc='lower center', ncol=7, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01),
           handletextpad=0.2, columnspacing=0.9)
fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.0, w_pad=0.8)
fig.savefig(os.path.join(OUT, 'fig_sintese.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig_sintese-1.png'), bbox_inches='tight', dpi=150)
print('figura:', os.path.join(OUT, 'fig_sintese.pdf'))
