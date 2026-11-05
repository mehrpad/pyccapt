"""Portable learned regression state and prediction, without pickle execution.

These snapshots reproduce inference, not training continuation. Unknown model
types fail explicitly so a settings-only record cannot masquerade as a model.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from pyccapt.calibration.core.exceptions import CalibrationInputError


MODEL_SCHEMA = 'pyccapt-regression-1'


def export_model_state(model):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
    from sklearn.linear_model import LinearRegression, RANSACRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.tree import DecisionTreeRegressor
    from pyccapt.calibration.core.correction_models import EstimatorCorrectionModel, PolynomialCorrectionModel
    from pyccapt.calibration.core.multireference_tof import MultiReferenceToFModel

    state = {'schema': MODEL_SCHEMA, 'kind': type(model).__name__}
    if isinstance(model, MultiReferenceToFModel):
        if model.coefficients is None:
            raise CalibrationInputError('Cannot snapshot an unfitted multi-reference ToF model')
        state.update(provenance=model.provenance(), keep=model.keep.tolist())
    elif isinstance(model, PolynomialCorrectionModel):
        if model.coefficients is None:
            raise CalibrationInputError('Cannot snapshot an unfitted polynomial')
        state.update(provenance=model.provenance())
    elif isinstance(model, EstimatorCorrectionModel):
        state.update(kind='estimator_adapter', feature_names=list(model.feature_names),
                     valid_domain=model.valid_domain, estimator=export_model_state(model.estimator))
    elif isinstance(model, RANSACRegressor):
        state.update(kind='fitted_ransac', estimator=export_model_state(model.estimator_),
                     inlier_mask=np.asarray(model.inlier_mask_, dtype=bool).tolist())
    elif isinstance(model, Pipeline):
        state.update(steps=[export_model_state(step) for _, step in model.steps])
    elif isinstance(model, PolynomialFeatures):
        state.update(powers=model.powers_.tolist(), n_features=int(model.n_features_in_))
    elif isinstance(model, StandardScaler):
        state.update(mean=None if not model.with_mean else model.mean_.tolist(),
                     scale=None if not model.with_std else model.scale_.tolist(), n_features=int(model.n_features_in_))
    elif isinstance(model, LinearRegression):
        state.update(coefficients=np.asarray(model.coef_).tolist(), intercept=np.asarray(model.intercept_).tolist(),
                     n_features=int(model.n_features_in_))
    elif isinstance(model, DecisionTreeRegressor):
        if model.n_outputs_ != 1:
            raise CalibrationInputError('Only single-output calibration regressors are supported')
        tree = model.tree_
        state.update(left=tree.children_left.tolist(), right=tree.children_right.tolist(),
                     feature=tree.feature.tolist(), threshold=tree.threshold.tolist(),
                     value=tree.value[:, 0, 0].tolist(), n_features=int(model.n_features_in_),
                     missing_left=getattr(tree, 'missing_go_to_left', np.zeros(tree.node_count, dtype=bool)).tolist())
    elif isinstance(model, (RandomForestRegressor, ExtraTreesRegressor)):
        state.update(kind='forest_mean', trees=[export_model_state(tree) for tree in model.estimators_],
                     n_features=int(model.n_features_in_))
    elif isinstance(model, GradientBoostingRegressor):
        from sklearn.dummy import DummyRegressor
        if isinstance(model.init_, str) and model.init_ == 'zero':
            initial = 0.
        elif isinstance(model.init_, DummyRegressor):
            initial = float(np.asarray(model.init_.constant_).reshape(-1)[0])
        else:
            raise CalibrationInputError('Only constant gradient-boosting initializers have a portable snapshot')
        state.update(initial=initial, learning_rate=float(model.learning_rate), n_features=int(model.n_features_in_),
                     trees=[export_model_state(tree) for tree in model.estimators_[:, 0]])
    else:
        raise CalibrationInputError(f'No safe learned-state exporter for {type(model).__name__}; retain accepted output and mark model unavailable')
    return state


class StoredRegressor:
    """Read-only inference from a whitelisted numeric model snapshot."""
    def __init__(self, state):
        self._state = deepcopy(state)
        _validate_state(self._state)

    def predict(self, features):
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2:
            raise CalibrationInputError('Model features must be a two-dimensional matrix')
        return _evaluate(self._state, values)

    predict_factor = predict

    def provenance(self):
        return deepcopy(self._state)


def _validate_state(state, depth=0):
    if not isinstance(state, dict) or state.get('schema') != MODEL_SCHEMA or depth > 20:
        raise CalibrationInputError('Unsupported or excessively nested learned-model schema')
    kind = state.get('kind')
    if kind == 'MultiReferenceToFModel':
        _restore_physical_tof(state)
    elif kind in ('estimator_adapter', 'fitted_ransac'):
        _validate_state(state['estimator'], depth+1)
    elif kind == 'Pipeline':
        if not state['steps']:
            raise CalibrationInputError('Empty pipeline snapshot')
        for step in state['steps']:
            _validate_state(step, depth+1)
    elif kind in ('forest_mean', 'GradientBoostingRegressor'):
        if not state['trees']:
            raise CalibrationInputError('Empty learned forest')
        for tree in state['trees']:
            _validate_state(tree, depth+1)
    elif kind == 'DecisionTreeRegressor':
        count = len(state['left'])
        if count == 0 or any(len(state[key]) != count for key in ('right', 'feature', 'threshold', 'value', 'missing_left')):
            raise CalibrationInputError('Invalid learned tree arrays')
        # sklearn trees number children after their parent. Enforcing this also
        # prevents a corrupt snapshot from creating an infinite traversal.
        for index, (left, right, feature) in enumerate(zip(state['left'], state['right'], state['feature'])):
            if left == right == -1:
                continue
            if not (index < left < count and index < right < count and 0 <= feature < state['n_features']):
                raise CalibrationInputError('Invalid learned tree child/feature indices')
    elif kind not in ('PolynomialCorrectionModel', 'PolynomialFeatures', 'StandardScaler', 'LinearRegression'):
        raise CalibrationInputError(f'Unsupported learned-model kind: {kind}')


def _evaluate(state, values):
    kind = state['kind']
    if 'n_features' in state and values.shape[1] != state['n_features']:
        raise CalibrationInputError('Model feature count changed')
    if kind in ('estimator_adapter', 'fitted_ransac'):
        return _evaluate(state['estimator'], values)
    if kind == 'MultiReferenceToFModel':
        return _restore_physical_tof(state).predict_factor(values)
    if kind == 'Pipeline':
        for step in state['steps']:
            values = _evaluate(step, values)
        return values
    if kind == 'PolynomialCorrectionModel':
        from pyccapt.calibration.core.correction_models import PolynomialCorrectionModel
        p = state['provenance']
        model = PolynomialCorrectionModel(p['kind'], p['degree'], np.asarray(p['coefficients']))
        return model.predict_factor(values)
    if kind == 'PolynomialFeatures':
        powers = np.asarray(state['powers'], dtype=int)
        return np.column_stack([np.prod(values**power, axis=1) for power in powers])
    if kind == 'StandardScaler':
        return (values - (0 if state['mean'] is None else np.asarray(state['mean']))) / (1 if state['scale'] is None else np.asarray(state['scale']))
    if kind == 'LinearRegression':
        return values @ np.asarray(state['coefficients']).T + np.asarray(state['intercept'])
    if kind in ('forest_mean', 'GradientBoostingRegressor'):
        total = np.zeros(len(values))
        for tree in state['trees']:
            total += _evaluate(tree, values)
        return total / len(state['trees']) if kind == 'forest_mean' else state['initial'] + state['learning_rate'] * total
    if kind == 'DecisionTreeRegressor':
        # sklearn's tree inference converts input to float32 before thresholds.
        values = values.astype(np.float32)
        left, right = np.asarray(state['left']), np.asarray(state['right'])
        feature, threshold = np.asarray(state['feature']), np.asarray(state['threshold'])
        missing_left = np.asarray(state['missing_left'], dtype=bool)
        nodes = np.zeros(len(values), dtype=int)
        active = left[nodes] != -1
        while np.any(active):
            rows = np.flatnonzero(active)
            current = nodes[rows]
            sample = values[rows, feature[current]]
            choose_left = np.where(np.isnan(sample), missing_left[current], sample <= threshold[current])
            nodes[rows] = np.where(choose_left, left[current], right[current])
            active = left[nodes] != -1
        return np.asarray(state['value'])[nodes]
    raise CalibrationInputError(f'Unsupported learned-model kind: {kind}')


def _restore_physical_tof(state):
    """Rebuild explicit physical terms; never deserialize executable model code."""
    from pyccapt.calibration.core.multireference_tof import FEATURE_NAMES, MultiReferenceToFModel, ToFModelConfig
    try:
        p = state['provenance']
        model = MultiReferenceToFModel(ToFModelConfig(**p['config']))
        model.center, model.scale = np.asarray(p['center'], dtype=float), np.asarray(p['scale'], dtype=float)
        model.coefficients = np.asarray(p['coefficients'], dtype=float)
        raw_keep = np.asarray(state['keep'])
        if (model.center.shape != (4,) or model.scale.shape != (4,) or
                not np.isfinite(model.center).all() or not np.isfinite(model.scale).all() or
                np.any(model.scale <= 0) or raw_keep.dtype.kind != 'b'):
            raise ValueError('Invalid scaling or active terms')
        model.keep = raw_keep.copy()
        _, names = model._raw_design(np.array([[1., 0., 0., .5, 1.]]))
        model.term_names = list(p['term_names'])
        if (model.keep.shape != (len(names),) or not model.keep[0] or
                model.coefficients.shape != (int(model.keep.sum()),) or
                not np.isfinite(model.coefficients).all() or
                model.term_names != np.asarray(names)[model.keep].tolist() or
                p['feature_names'] != list(FEATURE_NAMES)):
            raise ValueError('Invalid coefficients, terms or feature order')
        model.t0_ns = float(p['t0_ns'])
        if not model.config.t0_bounds_ns[0] <= model.t0_ns <= model.config.t0_bounds_ns[1]:
            raise ValueError('Invalid ToF offset')
        # Canonical JSON sorts keys. Restore explicit feature order because domain
        # checks associate these ordered limits with columns of the feature matrix.
        model._valid_domain = {}
        for name in FEATURE_NAMES[:4]:
            bounds = np.asarray(p['valid_domain'][name], dtype=float)
            if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
                raise ValueError('Invalid fitted feature domain')
            model._valid_domain[name] = tuple(bounds)
        model.fit_diagnostics = deepcopy(p.get('diagnostics', {}))
        return model
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise CalibrationInputError(f'Invalid learned multi-reference ToF state: {exc}') from exc


def restore_model_state(state):
    if isinstance(state, dict) and state.get('kind') == 'MultiReferenceToFModel':
        _validate_state(state)
        return _restore_physical_tof(state)
    return StoredRegressor(state)
