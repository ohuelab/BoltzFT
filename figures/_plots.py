"""Plot screening and comparator results."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
from _style import COLORS, FIGWIDTH, N_COLORS, panel_label, savefig_both
from _data import read, targets, budgets, paired, mean_interval, METRICS, OUT


def ratio_axis(ax, axis='y'):
    getattr(ax, f'set_{axis}scale')('log')
    a = getattr(ax, f'{axis}axis')
    a.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2, 5, 10]))
    a.set_minor_locator(NullLocator())
    a.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))


def learning():
    ns = budgets()
    xs = np.arange(len(ns))
    fig, axes = plt.subplots(2, 3, figsize=(FIGWIDTH, 5.3), layout='constrained')
    for j, (metric, name) in enumerate(METRICS):
        ax = axes[0, j]
        values = np.column_stack([paired(metric, ns[0])[0]] + [paired(metric, n)[1] for n in ns])
        abs_x = np.arange(len(ns) + 1)
        for vals in values:
            ax.plot(abs_x, vals, 'o-', color='0.62', lw=0.65, ms=2.7, alpha=0.75, zorder=2)
        summary = np.array([mean_interval(values[:, i]) for i in range(len(abs_x))])
        m, lo, hi = summary.T
        ax.plot(abs_x, m, color=COLORS['lightning'], lw=1.4, zorder=3)
        for i in range(len(abs_x)):
            ax.errorbar(abs_x[i], m[i], yerr=[[m[i]-lo[i]], [hi[i]-m[i]]],
                        fmt='s' if i == 0 else 'D', color=COLORS['base'] if i == 0 else COLORS['lightning'],
                        ms=4.5, lw=1.4, capsize=3, zorder=4)
        ax.set_xticks(abs_x, ['No-FT'] + [str(n) for n in ns])
        ax.set_xlabel('Assay labels, N')
        ax.set_ylabel(name if metric != 'bedroc' else r'BEDROC ($\alpha=20$)')
        ax.set_ylim(bottom=min(0, lo.min()))
        ax.grid(axis='y', alpha=0.18)
        panel_label(ax, 'abc'[j]+')')
        ax = axes[1, j]
        ratios = np.array([paired(metric, n)[1] / paired(metric, n)[0] for n in ns]).T
        for vals in ratios:
            ax.plot(xs, vals, 'o-', color='0.62', lw=0.65, ms=2.7, alpha=0.75, zorder=2)
        summary = np.array([mean_interval(ratios[:, i], True) for i in range(len(ns))])
        m, lo, hi = summary.T
        ax.errorbar(xs, m, yerr=[m-lo, hi-m], fmt='D-', color=COLORS['lightning'],
                    ms=5, lw=1.5, capsize=3, zorder=4)
        ax.axhline(1, color=COLORS['base'], ls='--', lw=0.9, zorder=1)
        ratio_axis(ax)
        ax.set_ylim(0.3, 6)
        ax.set_xlim(-0.23, len(ns)-0.77)
        ax.set_xticks(xs, [str(n) for n in ns])
        ax.set_xlabel('Assay labels, N')
        ax.set_ylabel(f'{name} ratio (Head-FT / No-FT)')
        ax.grid(axis='y', alpha=0.18)
        panel_label(ax, 'def'[j]+')')
    fig.legend(handles=[Line2D([], [], color='0.62', marker='o', lw=0.7, ms=3, label='Individual assay'),
                        Line2D([], [], color=COLORS['lightning'], marker='D', lw=1.4, ms=4, label='Mean and 95% CI')],
               loc='outside lower center', ncol=2, frameon=False, fontsize=8)
    savefig_both(fig, OUT/'learning_curve.pdf')
    plt.close(fig)


def absolute():
    n = max(budgets())
    ys = np.arange(len(targets()))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(FIGWIDTH, 3.6), sharey=True, layout='constrained')
    for ax, letter, (metric, name) in zip(axes, 'abc', METRICS):
        base, ft = paired(metric, n)
        for y, b, f in zip(ys, base, ft):
            ax.plot([b, f], [y, y], color='0.65', lw=1.0, zorder=1)
        ax.scatter(base, ys, color=COLORS['base'], marker='s', s=23, label='No-FT', zorder=3)
        ax.scatter(ft, ys, color=COLORS['lightning'], marker='o', s=25, label=f'Head-FT, N = {n}', zorder=3)
        ax.set_xlabel(name if metric != 'bedroc' else r'BEDROC ($\alpha=20$)')
        ax.set_xlim(left=0)
        ax.grid(axis='x', alpha=0.18)
        panel_label(ax, letter+')')
    axes[0].set_yticks(ys, targets(), fontsize=8)
    axes[0].set_ylabel('PubChem assay ID')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=2,
               frameon=False, fontsize=8)
    savefig_both(fig, OUT/'absolute_per_target.pdf')
    plt.close(fig)


def baselines():
    from _baselines import load_baselines, common_targets
    df = load_baselines()
    arms = ['lightning', 'drugclip_headft', 'drugclip_zeroshot', 'graw', 'chemeleon', 'chemeleon_graw', 'ecfp', 'ecfp_graw']
    names = ['Boltz-2 head-FT', 'DrugCLIP head-FT', 'DrugCLIP zero-shot', 'Trunk', 'CheMeleon', 'CheMeleon + trunk', 'ECFP', 'ECFP + trunk']
    ys = np.arange(len(arms))[::-1]
    offsets = dict(zip(targets(), np.linspace(-0.16, 0.16, len(targets()))))
    fig, axes = plt.subplots(1, 3, figsize=(FIGWIDTH, 4.2), sharey=True, sharex=True, layout='constrained')
    all_values = []
    for ax, letter, n in zip(axes, 'abc', budgets()):
        sub = df[df.train_size == n]
        base = df[df.arm == 'base'].set_index('target').auprc
        keep = common_targets(df, n)
        for arm, y in zip(arms, ys):
            vals = sub[sub.arm == arm].set_index('target').auprc
            if arm == 'drugclip_zeroshot':
                vals = df[df.arm == arm].set_index('target').auprc
            ratios = vals.loc[keep].to_numpy() / base.loc[keep].to_numpy()
            all_values.extend(ratios)
            m, lo, hi = mean_interval(ratios, True)
            color = COLORS[arm]
            ax.scatter(ratios, [y+offsets[t] for t in keep], color=color, alpha=0.5,
                       s=13, edgecolors='none', zorder=2)
            marker = {'lightning': 'D', 'drugclip_headft': '^', 'drugclip_zeroshot': 'v'}.get(arm, 's')
            ax.errorbar(m, y, xerr=[[m-lo], [hi-m]], fmt=marker,
                        ms=4.5, color=color, markeredgecolor='black', markeredgewidth=0.5,
                        lw=1, capsize=2, zorder=3)
            all_values.extend([lo, hi])
        ratio_axis(ax, 'x')
        ax.axvline(1, color=COLORS['base'], ls='--', lw=0.9)
        ax.set_xlabel('AP ratio to Boltz-2 No-FT', fontsize=8)
        ax.set_title(f'N = {n}', fontsize=9)
        panel_label(ax, letter+')')
        ax.grid(axis='x', alpha=0.18)
        ax.set_ylim(-0.6, len(arms)-0.25)
    axes[0].set_yticks(ys, names, fontsize=8)
    axes[0].set_ylabel('Model / LightGBM features', fontsize=8)
    axes[0].set_xlim(min(all_values)/1.15, max(all_values)*1.18)
    fig.legend(handles=[Line2D([], [], color='0.5', marker='o', lw=0, ms=3, label='Individual assay'),
                        Line2D([], [], color='0.3', marker='s', lw=1, ms=4, label='Geometric mean and 95% CI')],
               loc='outside lower center', ncol=2, frameon=False, fontsize=8)
    savefig_both(fig, OUT/'supervised_baselines.pdf')
    plt.close(fig)


def cascade_curve(ax, df, n, k, raw=False):
    sub = df[(df.n_train == n) & (df.budget_k == k)]
    full = sub[sub.is_full].set_index('target').loc[targets()]
    cuts = sorted(sub.loc[~sub.is_full, 'cut_M'].unique())
    xs = np.array(cuts + [full.cut_M.mean()])
    vals = np.array([sub[sub.cut_M == cut].set_index('target').loc[targets(), 'hits'].to_numpy()
                     for cut in cuts] + [full.hits.to_numpy()]).T
    if raw:
        for values in vals:
            ax.plot(xs, values, '-', color='0.76', lw=0.6, zorder=1)
    summary = np.array([mean_interval(vals[:, i]) for i in range(len(xs))])
    m, lo, hi = summary.T
    color = COLORS['lightning'] if raw else N_COLORS[n]
    ax.errorbar(xs, m, yerr=[m-lo, hi-m], fmt='o-', color=color, ms=3.5,
                lw=1.3, capsize=2.5, zorder=3)
    ax.axhline(full.hits.mean(), color='0.25', ls=':', lw=1)
    ax.axhline(full.hits_stage1_only.mean(), color=COLORS['base'], ls='--', lw=1)
    ax.set_xscale('log')
    ax.set_xticks([cuts[0], 5000, xs[-1]], [f'{cuts[0]:,}', '5,000', 'All'])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(axis='y', alpha=0.18)
    ax.set_xlabel('Compounds rescored, M')
    return sub


def cascade():
    df = read('cascade.csv')
    n = max(budgets())
    k = int(df.budget_k.min())
    fig, axes = plt.subplots(1, 2, figsize=(FIGWIDTH, 3.4), gridspec_kw={'width_ratios':[1,1.15]}, layout='constrained')
    sub = cascade_curve(axes[0], df, n, k, raw=True)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel(f'Mean actives@{k} per assay')
    panel_label(axes[0], 'a)')
    sel = sub[sub.cut_M == 5000].set_index('target').loc[targets()]
    full = sub[sub.is_full].set_index('target').loc[targets()]
    ys = np.arange(len(targets()))[::-1]
    for y, b, f in zip(ys, sel.hits_stage1_only, sel.hits):
        axes[1].plot([b, f], [y, y], color='0.75', lw=0.9, zorder=1)
    axes[1].scatter(sel.hits_stage1_only, ys, color=COLORS['base'], marker='s', s=20, label='No-FT', zorder=3)
    axes[1].scatter(full.hits, ys+0.10, facecolors='white', edgecolors='0.2', marker='D', s=27,
                    label='Head-FT, all compounds', zorder=3)
    axes[1].scatter(sel.hits, ys-0.10, color=COLORS['lightning'], marker='o', s=23,
                    label='Head-FT, M = 5,000', zorder=4)
    axes[1].set_yticks(ys, targets(), fontsize=7.5)
    axes[1].set_xlabel(f'Actives@{k}')
    axes[1].set_xlim(left=-3)
    axes[1].grid(axis='x', alpha=0.18)
    panel_label(axes[1], 'b)')
    fig.legend(*axes[1].get_legend_handles_labels(), loc='outside lower center', ncol=3,
               fontsize=7.3, frameon=False)
    savefig_both(fig, OUT/'cascade.pdf')
    plt.close(fig)


def cascade_all():
    df = read('cascade.csv')
    ks = sorted(df.budget_k.unique())
    fig, axes = plt.subplots(len(ks), len(budgets()), figsize=(FIGWIDTH, 4.6), sharey='row', layout='constrained')
    for i, k in enumerate(ks):
        for j, n in enumerate(budgets()):
            ax = axes[i,j]
            cascade_curve(ax, df, n, k)
            panel_label(ax, chr(97+i*len(budgets())+j)+')')
            ax.set_title(f'N = {n}', fontsize=9)
            if j == 0:
                ax.set_ylabel(f'Mean actives@{k}\nper assay')
    for row in axes:
        row[0].set_ylim(bottom=0)
    fig.legend(handles=[Line2D([], [], color='0.35', marker='o', ms=3, lw=1.2, label='Head-FT'),
                        Line2D([], [], color='0.25', ls=':', lw=1, label='Head-FT, all compounds'),
                        Line2D([], [], color=COLORS['base'], ls='--', lw=1, label='No-FT')],
               loc='outside lower center', ncol=3, frameon=False, fontsize=8)
    savefig_both(fig, OUT/'cascade_all.pdf')
    plt.close(fig)
