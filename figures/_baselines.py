"""CV-selected comparators on the common evaluation population."""
import json

import numpy as np
import pandas as pd

from _data import DATA, METRICS, read, targets

ARM_MAP = {'boltz_base': 'base', 'boltz_headft': 'lightning',
           'drugclip_headft': 'drugclip_headft', 'drugclip_zeroshot': 'drugclip_zeroshot',
           **{f'{feature}_lgbm_cv': feature for feature in
              ['graw', 'ecfp', 'chemeleon', 'ecfp_graw', 'chemeleon_graw']}}


def load_baselines():
    root = DATA / 'baseline_cv'
    load = lambda name: pd.read_csv(root / name, dtype={'target': str})
    dc = load('metrics.csv')
    lgb = pd.concat([load('supervised_baselines.csv'), load('supervised_baselines_concat.csv')])
    lgb = lgb[lgb.tuning == 'cv']
    assert not lgb.duplicated(['target', 'arm', 'train_size']).any()
    assert not dc.duplicated(['target', 'arm', 'train_size', 'seed']).any()
    assert dc.n_dropped.eq(0).all()
    ref = json.loads((root / 'reference_digest.json').read_text())['targets']
    for target, group in pd.concat([dc, lgb]).groupby('target'):
        assert group.n_eval.eq(ref[target]['n_eval']).all()
        assert group.n_active.eq(ref[target]['n_active_eval']).all()
    controls = read('trainer_control.csv')
    metrics = [m for m, _ in METRICS]
    for new, old in [('boltz_base', 'base'), ('boltz_headft', 'lightning')]:
        a = dc[dc.arm == new].set_index(['target', 'train_size']).sort_index()
        b = controls[controls.arm == old].set_index(['target', 'n_train']).sort_index()
        assert list(a.index) == list(b.index)
        assert np.allclose(a[metrics], b[metrics], rtol=1e-12, atol=1e-14)
    for n in sorted(lgb.train_size.unique()):
        sub = lgb[lgb.train_size == n]
        counts = sub.groupby('target').n_train_active
        assert counts.nunique().eq(1).all()
        counts = counts.first()
        eligible = set(counts.index[(counts >= 3) & ((n - counts) >= 3)])
        for arm, group in sub.groupby('arm'):
            assert set(group.target) == set(targets())
            assert set(group.loc[group.auprc.notna(), 'target']) == eligible
            assert group.loc[group.auprc.notna(), 'note'].eq('cv').all()
        group = dc[(dc.arm == 'drugclip_headft') & (dc.train_size == n)]
        assert set(group.target) == eligible
        assert group.groupby('target').size().eq(5).all()
        assert group.n_train_used.eq(n).all() and group.hp_source.eq('cv').all()
        assert np.array_equal(group.n_train_active, group.target.map(counts))
    raw = pd.concat([dc, lgb])
    raw = raw[raw.arm.isin(ARM_MAP)]
    per = raw.groupby(['target', 'arm', 'train_size'])[metrics].mean().sort_index()
    per = per.reset_index()
    per.arm = per.arm.map(ARM_MAP)
    return per


def common_targets(df, n):
    sub = df[df.train_size == n]
    keep = set.intersection(*[set(g.loc[g.auprc.notna(), 'target']) for _, g in sub.groupby('arm')])
    return [target for target in targets() if target in keep]
