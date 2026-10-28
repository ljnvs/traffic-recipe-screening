"""Fixed-library L1 estimators; source-city-only fitting and nested selection.

No predictor training, raw data reading, or external test labels are required.
The v85 source adapter reads the already observed six-city corpus only.
"""
from __future__ import annotations
import copy
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

C004 = 'components_v5_current_pems_metadata_004'
SEEDS = [20261010, 20261011, 20261012]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()


def weighted_quantile(x, weights, q):
    x, weights = np.asarray(x, float), np.asarray(weights, float)
    order = np.argsort(x, kind='stable')
    x, weights = x[order], weights[order]
    return float(x[min(np.searchsorted(np.cumsum(weights), q * weights.sum()), len(x)-1)])


def city(data, t):
    return data['tasks'][t]['city']


def task_ids(data, cities):
    return sorted(t for t in data['tasks'] if city(data, t) in cities)


def task_weights(data, tids):
    counts = Counter(city(data, t) for t in tids)
    return np.array([1 / (len(counts) * counts[city(data, t)]) for t in tids])


def source_order(data, cities, pool):
    tids = task_ids(data, cities)
    weights = task_weights(data, tids)
    scores = {c: float(sum(w * np.mean(np.asarray(data['labels'][t][c]) /
        data['labels'][t][C004]) for t, w in zip(tids, weights))) for c in pool}
    return sorted(pool, key=lambda c: (scores[c], c)), scores


class TaskEncoder:
    """All numeric and categorical fitted preprocessing uses source tasks only."""
    def __init__(self, data, cities, feature_mode='full'):
        self.feature_mode = feature_mode
        self.tids = task_ids(data, cities)
        self.names = list(data['schema']['generic' if feature_mode == 'generic' else 'full'])
        self.names += ['native_horizon_steps']
        if feature_mode == 'no_H':
            self.names = [n for n in self.names if n not in data['schema']['no_H_deleted']]
        self.categories = ['variable', 'measurement_basis', 'missingness_category']
        self.cat_values = {n: sorted({data['metadata'][t][n] for t in self.tids}) for n in self.categories}
        self.raw_scalers = {}
        unique = {}
        for t in self.tids:
            m = data['tasks'][t]
            key = (city(data, t), m['variable'], tuple(m['feature_history_days']))
            unique[key] = t
        for var in sorted({data['metadata'][t]['variable'] for t in self.tids}):
            for name in ['native_mean_node_mean', 'native_mean_node_std']:
                vals = [data['features'][t].get(name) for t in unique.values()
                    if data['metadata'][t]['variable'] == var]
                vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
                center = float(np.median(vals)) if vals else 0.
                scale = float(np.percentile(vals, 75)-np.percentile(vals, 25)) if vals else 1.
                self.raw_scalers[var, name] = (center, scale if scale > 1e-12 else 1.)
        weights = task_weights(data, self.tids)
        self.scalers = {}
        for name in self.names:
            vals = np.array([self.value(data, t, name) for t in self.tids])
            ok = np.isfinite(vals)
            center = weighted_quantile(vals[ok], weights[ok], .5) if ok.any() else 0.
            scale = (weighted_quantile(vals[ok], weights[ok], .75) -
                weighted_quantile(vals[ok], weights[ok], .25)) if ok.any() else 1.
            self.scalers[name] = (center, scale if scale > 1e-12 else 1.)
        self.columns = [p+n for n in self.names for p in ['value:', 'missing:']]
        self.columns += [f'{n}={v}' for n in self.categories for v in self.cat_values[n] + ['<unknown>']]

    def value(self, data, t, name):
        val = (data['metadata'][t] if name == 'native_horizon_steps' else data['features'][t]).get(name)
        if val is None or not np.isfinite(val):
            return float('nan')
        if name in ['native_mean_node_mean', 'native_mean_node_std']:
            center, scale = self.raw_scalers.get((data['metadata'][t]['variable'], name), (0., 1.))
            val = (val-center)/scale
        return float(val)

    def transform(self, data, tids):
        rows = []
        for t in tids:
            row = []
            for n in self.names:
                v = self.value(data, t, n)
                center, scale = self.scalers[n]
                row.extend([(v-center)/scale if np.isfinite(v) else 0., float(not np.isfinite(v))])
            for n in self.categories:
                v = data['metadata'][t][n]
                values = self.cat_values[n]
                row.extend([float(v == k) for k in values] + [float(v not in values)])
            rows.append(row)
        return np.asarray(rows, dtype=np.float64)


def candidate_vectors(data, pool, representation):
    if representation == 'ID':
        return np.eye(len(pool))
    # Fixed public library descriptors, including absent-property indicators.
    keys = sorted(set().union(*(data['descriptors'][c] for c in pool)))
    rows = []
    for c in pool:
        original = data['descriptors'][c]
        row = {}
        for key in keys:
            value = original.get(key)
            if isinstance(value, (int, float, bool)):
                row[key+':numeric'] = float(value)
            elif value is not None:
                row[key+':category'] = str(value)
            row[key+':absent'] = float(value is None)
        rows.append(row)
    from sklearn.feature_extraction import DictVectorizer
    return DictVectorizer(sparse=False).fit_transform(rows)


def feature_rows(encoder, data, tids, pool, representation, kind):
    tx = encoder.transform(data, tids)
    cv = candidate_vectors(data, pool, representation)
    if kind == 'pair':
        pairs = [(i, j) for i in range(len(pool)) for j in range(len(pool)) if i != j]
        cx = np.asarray([np.concatenate([cv[i], cv[j], cv[i]-cv[j]]) for i, j in pairs])
        indexes = pairs
    else:
        cx, indexes = cv, list(range(len(pool)))
    return np.concatenate([np.repeat(tx, len(cx), axis=0), np.tile(cx, (len(tids), 1))], axis=1), indexes


def fit_estimator(data, cities, pool, spec, seed):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
    tids = task_ids(data, cities)
    enc = TaskEncoder(data, cities, spec.get('features', 'full'))
    kind = spec['kind']
    weights = task_weights(data, tids)
    if kind == 'rf':
        x = enc.transform(data, tids)
        y = np.array([min(pool, key=lambda c: (np.mean(data['labels'][t][c]), c)) for t in tids])
        if spec['prior'] != 'none':
            mass = {c: float(weights[y == c].sum()) for c in set(y)}
            weights = weights / np.array([mass[c] for c in y])
            for c in cities:
                ix = np.array([city(data, t) == c for t in tids])
                weights[ix] /= weights[ix].sum() * len(cities)
        mf = max(1, x.shape[1]//3) if spec['max_features'] == 'one_third' else 'sqrt'
        model = RandomForestClassifier(n_estimators=spec['trees'], max_features=mf,
            min_samples_leaf=spec['leaf'], random_state=seed, n_jobs=1, bootstrap=True)
    else:
        x, indexes = feature_rows(enc, data, tids, pool, spec['representation'], kind)
        y = []
        for t in tids:
            logmae = {c: float(np.log(data['labels'][t][c]).mean()) for c in pool}
            ref = float(np.log(data['labels'][t][C004]).mean())
            if kind == 'pair':
                y.extend(logmae[pool[j]]-logmae[pool[i]] for i, j in indexes)
            elif kind == 'old':
                y.extend(float(np.mean(np.asarray(data['labels'][t][c])/data['labels'][t][C004])) for c in pool)
            else:
                y.extend(logmae[c]-ref for c in pool)
        # Equal-seed squared-loss duplicates have the same optimum/GB updates
        # as the mean target with normalized weight (constant residual term).
        weights = np.repeat(weights / len(indexes), len(indexes))
        model = GradientBoostingRegressor(n_estimators=spec['trees'], max_depth=spec['depth'],
            learning_rate=.05, loss='squared_error', random_state=seed)
    model.fit(x, y, sample_weight=weights)
    return dict(encoder=enc, model=model, spec=copy.deepcopy(spec),
        source_cities=sorted(cities), source_tasks=tids, pool=list(pool),
        sample_weights=weights.tolist(), training_labels=np.asarray(y).tolist())


def predict_estimator(bundle, data, tids):
    spec, pool = bundle['spec'], bundle['pool']
    kind = spec['kind']
    if kind == 'rf':
        raw = bundle['model'].predict_proba(bundle['encoder'].transform(data, tids))
        result = np.zeros((len(tids), len(pool)))
        for j, c in enumerate(bundle['model'].classes_):
            result[:, pool.index(c)] = raw[:, j]
        return result
    x, indexes = feature_rows(bundle['encoder'], data, tids, pool, spec['representation'], kind)
    raw = bundle['model'].predict(x).reshape(len(tids), len(indexes))
    if kind == 'pair':
        result = np.zeros((len(tids), len(pool), len(pool)))
        for k, (i, j) in enumerate(indexes):
            result[:, i, j] = raw[:, k]
        return (result-result.transpose(0, 2, 1))/2
    return raw


def grids():
    def gb(kind='pair', rep='description', features='full', trees=80, depth=2):
        return dict(kind=kind, representation=rep, features=features, trees=trees, depth=depth)
    def gated(kind='pair', rep='description', features='full', no_u=False):
        return [dict(estimator=gb(kind, rep, features, depth=d), rule='gate',
            lam=lam, tau=-math.log1p(-gain)) for d, lam, gain in
            itertools.product([1, 2], [0.] if no_u else [.5, 1.], [0., .002, .005])]
    result = dict(S09=gated(), S08_ID=gated(rep='ID'),
        A_generic_description=gated(features='generic'),
        A_generic_ID=gated(rep='ID', features='generic'),
        A_no_H=gated(features='no_H'), A_no_u=gated(no_u=True),
        A_no_gate=[dict(estimator=gb(depth=d), rule='best') for d in [1, 2]],
        S04_direct=[dict(estimator=gb('direct', r, trees=n, depth=d), rule='best')
            for r, n, d in itertools.product(['ID', 'description'], [80, 160], [1, 2, 3])],
        S05_matched=gated('direct') + gated('direct', 'ID'),
        S03_oldGB=[dict(estimator=gb('old'), rule='best')],
        S07_FFORMS=[dict(estimator=dict(kind='rf', features='full', trees=1000,
            max_features=mf, leaf=leaf, prior=prior), rule='probability')
            for mf, leaf, prior in itertools.product(['one_third', 'sqrt'], [1, 2, 4], ['none', 'inverse_class_mass'])],
        S02_nearest=[dict(rule='nearest', k=k, norm=norm, physical_weight=p)
            for k, norm, p in itertools.product([1, 3], [1, 2], [.5, 1., 2.])],
        S00_fixed=[dict(rule='fixed')], S01_majority=[dict(rule='majority')])
    for method, configs in result.items():
        for i, config in enumerate(configs):
            config['config_id'] = f'{method}_{i:02d}'
    return result


def needed_keys(cities, methods, pool, seeds):
    """Exact base-estimator cache identities, including delete-source models."""
    required = {}
    def add(train, config, seed):
        if 'estimator' not in config:
            return
        subsets = [tuple(sorted(train))]
        if config['rule'] == 'gate' and config['lam'] > 0:
            subsets += [tuple(c for c in sorted(train) if c != omitted) for omitted in train]
        for subset in subsets:
            spec = config['estimator']
            key = digest(dict(cities=subset, pool=pool, estimator=spec, meta_seed=seed))
            required[key] = dict(cities=list(subset), pool=pool, estimator=spec, meta_seed=seed)
    for outer in cities:
        train = [c for c in cities if c != outer]
        for method in methods.values():
            for config in method:
                for seed in seeds:
                    add(train, config, seed)
                    if len(method) > 1:
                        for inner in train:
                            add([c for c in train if c != inner], config, seed)
    return required


class Engine:
    def __init__(self, data, pool, fetch):
        self.data, self.pool, self.fetch = data, sorted(pool), fetch
        self.tids = sorted(data['tasks'])

    def raw(self, train, spec, seed, target):
        array = self.fetch(tuple(sorted(train)), spec, seed)
        return array[[self.tids.index(t) for t in target]]

    def choose(self, train, targets, config, seed):
        data, pool = self.data, self.pool
        order, _ = source_order(data, train, pool)
        b = order[0]
        rule = config['rule']
        if rule == 'fixed':
            return [b]*len(targets), b
        source = task_ids(data, train)
        if rule == 'majority':
            votes = defaultdict(float)
            for t, w in zip(source, task_weights(data, source)):
                winner = min(pool, key=lambda c: (np.mean(data['labels'][t][c]), c))
                votes[winner] += w
            chosen = min(order, key=lambda c: (-votes[c], order.index(c)))
            return [chosen]*len(targets), b
        if rule == 'nearest':
            enc = TaskEncoder(data, train)
            sx, tx = enc.transform(data, source), enc.transform(data, targets)
            physical = np.array([any(n.endswith(f) for f in data['schema']['physical_block']) for n in enc.columns])
            sx[:, physical] *= config['physical_weight']
            tx[:, physical] *= config['physical_weight']
            chosen = []
            for x in tx:
                dist = np.linalg.norm(sx-x, ord=config['norm'], axis=1)
                closest = []
                for c in sorted(train):
                    eligible = [i for i, t in enumerate(source) if city(data, t) == c]
                    i = min(eligible, key=lambda i: (dist[i], source[i]))
                    closest.append((dist[i], c, source[i]))
                neighbors = [t for _, _, t in sorted(closest)[:config['k']]]
                score = {c: np.mean([np.mean(np.log(np.asarray(data['labels'][t][c]) /
                    data['labels'][t][C004])) for t in neighbors]) for c in pool}
                chosen.append(min(pool, key=lambda c: (score[c], c)))
            return chosen, b
        spec = config['estimator']
        raw = self.raw(train, spec, seed, targets)
        if rule == 'probability':
            return [pool[int(np.argmax(row))] for row in raw], b
        bi = pool.index(b)
        def gains(array):
            if spec['kind'] == 'pair':
                return array[:, :, bi]
            return array[:, bi, None]-array
        d = gains(raw)
        if rule == 'best':
            return [pool[int(np.argmax(row))] for row in d], b
        u = np.zeros_like(d)
        if config['lam'] > 0:
            # Crucially every child compares to PARENT b, not its own b.
            children = [gains(self.raw([c for c in train if c != excluded], spec, seed, targets)) for excluded in train]
            u = np.std(children, axis=0, ddof=0)
        score = d-config['lam']*u
        score[:, bi] = -np.inf
        indexes = np.argmax(score, axis=1)
        return [pool[i] if score[j, i] > config['tau'] else b for j, i in enumerate(indexes)], b

    def evaluate(self, targets, selected, b):
        rows = []
        for t, c in zip(targets, selected):
            values = np.asarray(self.data['labels'][t][c])
            base = np.asarray(self.data['labels'][t][b])
            oracle = np.min([self.data['labels'][t][p] for p in self.pool], axis=0)
            rows.append(dict(task_id=t, city=city(self.data, t), variable=self.data['tasks'][t]['variable'],
                selected=c, source_baseline=b, gain_pct=float(np.mean(100*(1-values/base))),
                regret_pct=float(np.mean(100*(values/oracle-1))),
                switched=c != b, beneficial_switch=c != b and float(np.mean(100*(1-values/base))) > 1e-6,
                seed_gains_pct=(100*(1-values/base)).tolist()))
        return rows

    def tune(self, train, configs, seed):
        if len(configs) == 1:
            return configs[0], []
        scores = []
        for cfg in configs:
            rows = []
            for held in sorted(train):
                inner = [c for c in train if c != held]
                targets = task_ids(self.data, [held])
                chosen, b = self.choose(inner, targets, cfg, seed)
                rows += self.evaluate(targets, chosen, b)
            summary = aggregate(rows)
            scores.append(dict(config=cfg, **summary))
        legal = [s for s in scores if s['worst_city_loss_pct'] <= .2]
        if not legal:
            return dict(rule='fixed', config_id='no_feasible_config_source_fallback'), scores
        best = min(legal, key=lambda s: (-s['macro_gain_pct'], s['worst_city_loss_pct'],
            s['switch_count'], s['config'].get('estimator', {}).get('depth', 0),
            -s['config'].get('tau', 0), -s['config'].get('lam', 0), s['config']['config_id']))
        return best['config'], scores

    def loco(self, cities, method, configs, seed):
        rows, calibration = [], []
        for held in sorted(cities):
            train = [c for c in cities if c != held]
            cfg, tuning = self.tune(train, configs, seed)
            targets = task_ids(self.data, [held])
            chosen, b = self.choose(train, targets, cfg, seed)
            rows += self.evaluate(targets, chosen, b)
            calibration.append(dict(held_city=held, training_cities=train,
                selected_config=cfg, inner_scores=tuning))
        return dict(method=method, meta_seed=seed, rows=rows,
            aggregate=aggregate(rows), flow_only=aggregate([r for r in rows if r['variable']=='flow']),
            calibration=calibration)


def aggregate(rows):
    by = defaultdict(list)
    for row in rows:
        by[row['city']].append(row['gain_pct'])
    means = {c: float(np.mean(v)) for c, v in sorted(by.items())}
    return dict(city_gain_pct=means, macro_gain_pct=float(np.mean(list(means.values()))),
        worst_city_loss_pct=max(0., -min(means.values())),
        switch_count=sum(r['switched'] for r in rows),
        beneficial_switch_count=sum(r['beneficial_switch'] for r in rows))


class FlowCalibrationEngine(Engine):
    """Fit all declared source tasks; tune the B flow-only primary objective."""
    def tune(self,train,configs,seed):
        if len(configs)==1:return configs[0],[]
        scores=[]
        for cfg in configs:
            rows=[]
            for held in sorted(train):
                inner=[c for c in train if c!=held]
                targets=[t for t in task_ids(self.data,[held]) if self.data['tasks'][t]['variable']=='flow']
                chosen,b=self.choose(inner,targets,cfg,seed);rows+=self.evaluate(targets,chosen,b)
            scores.append(dict(config=cfg,**aggregate(rows)))
        legal=[s for s in scores if s['worst_city_loss_pct']<=.2]
        if not legal:return dict(rule='fixed',config_id='no_feasible_config_source_fallback'),scores
        best=min(legal,key=lambda s:(-s['macro_gain_pct'],s['worst_city_loss_pct'],s['switch_count'],
            s['config'].get('estimator',{}).get('depth',0),-s['config'].get('tau',0),-s['config'].get('lam',0),s['config']['config_id']))
        return best['config'],scores

def final_grids():
    """Finite source-development revision, identical depths to v87/v96."""
    result = grids()
    for name in ['S09', 'S08_ID', 'A_generic_description', 'A_generic_ID',
                 'A_no_H', 'A_no_u', 'A_no_gate', 'S05_matched']:
        for config in result[name]:
            config['estimator']['depth'] += 1
    return result
