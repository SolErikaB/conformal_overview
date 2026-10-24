import os
import yaml
import joblib
import numpy as np
import pandas as pd
import time
from collections import Counter

from conformal_prediction import ConformalPrediction

# ---------------------------------------------------------------------------
# Truncation windows
# ---------------------------------------------------------------------------
# e = 1 - top-1 accuracy. Breakpoints are named symbolically so that rows are
# comparable across datasets/models with different accuracies.

# S(0, 1) is the S-criterion of Vovk (2012)
# S(0, e) is the I_1 of Boman (2026)

BREAKPOINT_NAMES = ['0', 'e/4', 'e/2', 'e', '1']
WINDOWS = [(BREAKPOINT_NAMES[i], BREAKPOINT_NAMES[j])
           for i in range(len(BREAKPOINT_NAMES))
           for j in range(i + 1, len(BREAKPOINT_NAMES))]
WINDOW_LABELS = [lo + '-' + hi for lo, hi in WINDOWS]

def breakpoint_values(top1_accuracy):
    e = 1.0 - float(top1_accuracy)
    if not 0.0 < e < 1.0:
        raise ValueError("1 - top1_accuracy must be in (0, 1), got " + str(e))
    return dict(zip(BREAKPOINT_NAMES, [0, e / 4, e / 2, e, 1.0]))


class ConformalConfig:
    """
    Holds configuration and static data that doesn't change between runs.
    Any key of the 'conformal_prediction' section of the yaml can be overridden
    by keyword, e.g. ConformalConfig(dataset='cifar10', conformal_domain='logits').
    """

    def __init__(self, config_path="config.yaml", **overrides):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        cp_cfg = dict(self.config['conformal_prediction'])
        cp_cfg.update(overrides)

        evaluation_cfg = self.config.get('evaluation', {})
        training_cfg = self.config.get('training', {})

        def config_imbalance_flag(section):
            if section is None:
                return False
            return bool(section.get('simulate_class_imbalance', False))

        eval_imbalance = config_imbalance_flag(evaluation_cfg)
        train_imbalance = config_imbalance_flag(training_cfg)
        default_imbalance = eval_imbalance or train_imbalance
        self.simulate_class_imbalance = bool(
            cp_cfg.get('simulate_class_imbalance', default_imbalance)
        )

        self.evaluation_dir = self.config['evaluation']['output_directory']
        self.dataset = cp_cfg['dataset']
        self.model_architecture = cp_cfg['model_architecture']
        self.mondrian = cp_cfg.get('mondrian', False)
        self.imbalance_seed = int(cp_cfg.get('imbalance_seed', self.config['training'].get('imbalance_seed', 123)))
        train_minority_fraction = float(self.config['training'].get('minority_class_fraction', 0.10))
        cp_minority_fraction = cp_cfg.get('minority_class_fraction')
        if cp_minority_fraction is not None:
            cp_minority_fraction = float(cp_minority_fraction)
            if not np.isclose(cp_minority_fraction, train_minority_fraction):
                raise ValueError(
                    "conformal_prediction.minority_class_fraction must match training.minority_class_fraction; "
                    f"got {cp_minority_fraction} vs {train_minority_fraction}."
                )
        self.minority_class_fraction = train_minority_fraction
        self.calibration_minority_fraction = float(cp_cfg.get('calibration_minority_fraction', 1.0))
        temperature = cp_cfg.get('temperature', 1.0)
        self.uses_temperature_variants = (
            self.dataset == 'imagenet' and self.model_architecture == 'resnet50'
        )
        if temperature == 'calibrated':
            temperature_tag = 'calibrated'
        else:
            try:
                temperature = float(temperature)
            except (TypeError, ValueError):
                raise ValueError("temperature must be 0.5, 1.0, 2.0, 4.0, or 'calibrated'")
            if temperature not in (0.5, 1.0, 2.0, 4.0):
                raise ValueError("temperature must be 0.5, 1.0, 2.0, 4.0, or 'calibrated'")
            temperature_tag = str(temperature).replace('.', '_')

        if self.uses_temperature_variants:
            if temperature_tag == 'calibrated':
                output_suffix = '_T_calibrated'
                self.n_calib = 30000
            elif temperature == 1.0:
                output_suffix = ''  # T=1 is the plain outputs file
                self.n_calib = 40000
            else:
                output_suffix = '_T_' + temperature_tag
                self.n_calib = 40000
        else:
            if temperature_tag != '1_0':
                raise ValueError("Temperature options other than 1.0 apply only to ImageNet ResNet50")
            output_suffix = ''

        is_imbalanced = self.simulate_class_imbalance
        output_suffix += '_imbalanced' if is_imbalanced else ''

        output_stem = self.dataset + '_' + self.model_architecture
        outputs_path = os.path.join(self.evaluation_dir, output_stem + '_outputs' + output_suffix + '.npz')
        metrics_path = os.path.join(self.evaluation_dir, output_stem + '_metrics' + output_suffix + '.npz')
        for path in (outputs_path, metrics_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    path + " is missing. Run model_eval_and_save_features.py first and check that "
                    "simulate_class_imbalance matches the evaluated model."
                )
        self.data = np.load(outputs_path)
        self._arrays_cache = {}

        # Configuration parameters
        self.conformal_domain = cp_cfg['conformal_domain']
        self.score_function = cp_cfg['score_function']
        self.distance_metric = cp_cfg['distance_metric']
        self.alpha = cp_cfg['alpha']
        self.reg_k = cp_cfg['reg_k']
        self.reg_lambda = cp_cfg['reg_lambda']
        self.u = cp_cfg.get('u', 0.001)
        self.n_workers = cp_cfg['n_workers']

        # Load metrics
        metrics = np.load(metrics_path)
        self.top1_accuracy = float(np.squeeze(metrics['top_1_accuracy']))
        self.top5_accuracy = float(np.squeeze(metrics['top_5_accuracy']))

        self.n_classes = np.unique(self.data['labels']).shape[0]
        if not self.uses_temperature_variants:
            self.n_calib = 40000 if self.dataset == 'imagenet' else 10000

    # ---- Splitting ---------------------------------------------------

    def _needed_arrays(self):
        """Only the arrays a split uses; cached per domain so a sweep loads them once."""
        domain = self.conformal_domain
        if domain not in self._arrays_cache:
            keys = set(['labels', 'probabilities', domain])
            self._arrays_cache[domain] = {key: np.array(self.data[key]) for key in keys}
        return self._arrays_cache[domain]

    @staticmethod
    def _minority_class_ids(labels, minority_fraction, seed, classes=None):
        if minority_fraction <= 0:
            return set()
        classes = np.unique(labels) if classes is None else np.asarray(classes)
        n_classes = len(classes)
        n_minority = max(1, min(n_classes, int(round(n_classes * minority_fraction))))
        rng = np.random.default_rng(seed)
        return set(rng.choice(classes, size=n_minority, replace=False).tolist())

    @staticmethod
    def make_split(random_seed, data_arrays, n_calib, conformal_domain,
                   simulate_class_imbalance=False, minority_class_fraction=0.10,
                   calibration_minority_fraction=1.0, imbalance_seed=123,
                   calibration_retention="round"):
        """
        Create a random calibration/test split from a dict of plain numpy arrays.

        If `simulate_class_imbalance` is enabled, calibration can subsample
        minority-class examples; the test indices remain unchanged. The standard
        retention mode rounds the requested fraction; `floor_min_one` is available
        for analyses that define retention as a truncated count with a minimum of one.
        """
        labels = data_arrays['labels']

        rng = np.random.RandomState(random_seed)
        indices = rng.permutation(len(labels))

        calibration_pool_idx = indices[:n_calib]
        test_idx = indices[n_calib:]

        if simulate_class_imbalance:
            minority_classes = ConformalConfig._minority_class_ids(
                labels,
                minority_class_fraction,
                imbalance_seed,
                classes=np.arange(data_arrays['probabilities'].shape[1]),
            )
            calib_labels = labels[calibration_pool_idx]
            is_minority = np.isin(calib_labels, list(minority_classes))
            keep_mask = ~is_minority

            for minority_class in minority_classes:
                class_positions = np.flatnonzero(calib_labels == minority_class)
                if calibration_retention == "round":
                    n_keep = min(len(class_positions), max(0, int(round(
                        calibration_minority_fraction * len(class_positions)
                    ))))
                elif calibration_retention == "floor_min_one":
                    n_keep = min(len(class_positions), int(
                        calibration_minority_fraction * len(class_positions)
                    ))
                    if calibration_minority_fraction > 0:
                        n_keep = max(1, n_keep)
                else:
                    raise ValueError(f"Unsupported calibration_retention: {calibration_retention}")
                keep_mask[class_positions] = False
                if n_keep:
                    selected_positions = rng.choice(class_positions, size=n_keep, replace=False)
                    keep_mask[selected_positions] = True

            cal_idx = calibration_pool_idx[keep_mask]
        else:
            cal_idx = calibration_pool_idx

        domain = data_arrays[conformal_domain]
        probs = data_arrays['probabilities']

        return {
            'calibration_data': domain[cal_idx],
            'calibration_labels': labels[cal_idx],
            'calibration_preds': probs[cal_idx].argmax(axis=1),
            'test_data': domain[test_idx],
            'test_labels': labels[test_idx],
            'calibration_pool_idx': calibration_pool_idx,
            'calibration_idx': cal_idx,
            'test_idx': test_idx,
        }

    def create_split(self, random_seed, calibration_retention="round"):
        """Create a random calibration/test split using this config's own data"""
        return self.make_split(
            random_seed,
            self._needed_arrays(),
            self.n_calib,
            self.conformal_domain,
            simulate_class_imbalance=self.simulate_class_imbalance,
            minority_class_fraction=self.minority_class_fraction,
            calibration_minority_fraction=self.calibration_minority_fraction,
            imbalance_seed=self.imbalance_seed,
            calibration_retention=calibration_retention,
        )

    # ---- Worker payload ------------------------------------------------

    def to_worker_dict(self):
        """
        Bundle everything a joblib worker process needs into one picklable
        dict. Reads the current attribute values, so mutating
        conf.score_function / conf.distance_metric between calls works.
        """
        return {
            'alpha': self.alpha,
            'data_arrays': self._needed_arrays(),
            'n_classes': self.n_classes,
            'distance_metric': self.distance_metric,
            'score_function': self.score_function,
            'mondrian': self.mondrian,
            'reg_k': self.reg_k,
            'reg_lambda': self.reg_lambda,
            'u': self.u,
            'top1_accuracy': self.top1_accuracy,
            'top5_accuracy': self.top5_accuracy,
            'n_calib': self.n_calib,
            'conformal_domain': self.conformal_domain,
            'dataset': self.dataset,
            'model_architecture': self.model_architecture,
            'simulate_class_imbalance': self.simulate_class_imbalance,
            'minority_class_fraction': self.minority_class_fraction,
            'calibration_minority_fraction': self.calibration_minority_fraction,
            'imbalance_seed': self.imbalance_seed,
        }


def run_parallel_iterations(conf, worker_fn, n_iterations, desc):
    """
    Run `worker_fn(random_seed, conf_data)` for random_seed in
    range(n_iterations), in parallel across conf.n_workers processes.
    """
    conf_data = conf.to_worker_dict()

    return joblib.Parallel(n_jobs=conf.n_workers)(
        joblib.delayed(worker_fn)(random_seed, conf_data)
        for random_seed in range(n_iterations)
    )


# ---------------------------------------------------------------------------
# S-criterion over windows
# ---------------------------------------------------------------------------

def cumulative_S(cp, breakpoints, chunk_size=2000):
    """
    F(t) = mean_i sum_y min(p_iy, t) for each t in `breakpoints` (dict name -> value).

    Reads cp._test_scores (cached by compute_scores) and works in row chunks, so
    the full (n_test, n_classes) p-value matrix is never held in memory at once
    (this matters for ImageNet: ~10k x 1000). F('0') = 0 by definition.
    """
    scores = cp._test_scores
    n_test = len(scores)
    positive = {name: t for name, t in breakpoints.items() if t > 0}
    totals = dict.fromkeys(positive, 0.0)

    for start in range(0, n_test, chunk_size):
        P = cp._pvalues(scores[start:start + chunk_size])
        for name, t in positive.items():
            totals[name] += np.minimum(P, t).sum()

    F = {name: total / n_test for name, total in totals.items()}
    F['0'] = 0.0
    return F


def _overall_coverage(cp, test_labels, alpha):
    """Return the fraction of true labels included in the alpha prediction sets."""
    labels = np.asarray(test_labels, dtype=int)
    thresholds = np.asarray(cp._thresholds_for_alpha(alpha), dtype=float)
    test_rows = np.arange(len(labels))
    return float(np.mean(cp._test_scores[test_rows, labels] <= thresholds[labels]))


def find_S_single_iteration(alpha, calibration_data, calibration_labels, calibration_preds,
                            test_data, test_labels, n_classes, distance_metric,
                            score_function, mondrian, reg_k, reg_lambda, top1_accuracy,
                            model_architecture=None, dataset=None, u=0.001, random_state=None):
    """
    Truncated S for every window, for one calibration/test split.
    Returns {window_label: raw integral of mean set size over the window}.

    Scores are computed once; there is no alpha loop. `alpha` is only needed
    because the constructor requires it (S itself is alpha-free).
    """
    cp = ConformalPrediction(
        alpha=alpha,
        calibration_data=calibration_data,
        calibration_labels=calibration_labels,
        calibration_preds=calibration_preds,
        test_data=test_data,
        test_labels=test_labels,
        n_classes=n_classes,
        distance_metric=distance_metric,
        score_function=score_function,
        mondrian=mondrian,
        reg_k=reg_k,
        reg_lambda=reg_lambda,
        u=u,
        random_state=random_state,
        model_architecture=model_architecture,
        dataset=dataset
    )
    cp.compute_scores()

    F = cumulative_S(cp, breakpoint_values(top1_accuracy))
    result = {lo + '-' + hi: F[hi] - F[lo] for lo, hi in WINDOWS}
    result['coverage'] = _overall_coverage(cp, test_labels, alpha)
    return result


def compute_iteration(random_seed, conf_data):
    """Worker: truncated S for every window for one calibration/test split."""
    split = ConformalConfig.make_split(
        random_seed,
        conf_data['data_arrays'],
        conf_data['n_calib'],
        conf_data['conformal_domain'],
        simulate_class_imbalance=conf_data['simulate_class_imbalance'],
        minority_class_fraction=conf_data['minority_class_fraction'],
        calibration_minority_fraction=conf_data['calibration_minority_fraction'],
        imbalance_seed=conf_data['imbalance_seed'],
    )

    return find_S_single_iteration(
        conf_data['alpha'],
        split['calibration_data'],
        split['calibration_labels'],
        split['calibration_preds'],
        split['test_data'],
        split['test_labels'],
        conf_data['n_classes'],
        conf_data['distance_metric'],
        conf_data['score_function'],
        conf_data['mondrian'],
        conf_data['reg_k'],
        conf_data['reg_lambda'],
        conf_data['top1_accuracy'],
        model_architecture=conf_data['model_architecture'],
        dataset=conf_data['dataset'],
        u=conf_data['u'],
        random_state=random_seed,
    )


def summarise(values):
    values = np.asarray(values, dtype=float)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {
        'S_median': median,
        'S_mean': values.mean(),
        'S_min': values.min(),
        'S_max': values.max(),
        'S_q1': q1,
        'S_q3': q3,
    }


def summarise_coverage(values, alpha=None):
    """Summarize overall prediction coverage across split redraws."""
    values = np.asarray(values, dtype=float)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    summary = {
        'coverage_median': median,
        'coverage_mean': values.mean(),
        'coverage_min': values.min(),
        'coverage_max': values.max(),
        'coverage_q1': q1,
        'coverage_q3': q3,
    }
    if alpha is not None:
        summary['alpha'] = float(alpha)
    return summary


# ---------------------------------------------------------------------------
# S_table bookkeeping
# ---------------------------------------------------------------------------
# A "spec" is a dict with: dataset, model_architecture, conformal_domain,
# score_function, distance_metric, mondrian, reg_k, reg_lambda, alpha, u.

def spec_from_conf(conf):
    return {
        'dataset': conf.dataset,
        'model_architecture': conf.model_architecture,
        'conformal_domain': conf.conformal_domain,
        'score_function': conf.score_function,
        'distance_metric': conf.distance_metric,
        'mondrian': conf.mondrian,
        'reg_k': conf.reg_k,
        'reg_lambda': conf.reg_lambda,
        'alpha': conf.alpha,
        'u': conf.u,
    }


def describe_u(u):
    return 'Random' if u is None else str(float(u))


def table_key(spec):
    """
    Row key in S_table. reg_k / reg_lambda only count for the scores that use
    them (raps: both, saps: lambda); other scores store -1 so that changing
    the regularisation settings does not invalidate unrelated rows.
    """
    score = spec['score_function']
    return {
        'model_architecture': spec['model_architecture'],
        'dataset': spec['dataset'],
        'domain': spec['conformal_domain'],
        'score_function': score,
        'distance_metric': spec['distance_metric'],
        'mondrian': bool(spec['mondrian']),
        'reg_k': float(spec['reg_k']) if score == 'raps' else -1.0,
        'reg_lambda': float(spec['reg_lambda']) if score in ('raps', 'saps') else -1.0,
        'u': np.nan if spec['u'] is None else float(spec['u']),
    }


def describe(spec):
    return (spec['dataset'] + ' ' + spec['model_architecture'] + ' ' + spec['conformal_domain'] + ' '
            + spec['score_function'] + ' ' + spec['distance_metric'] + ' mondrian=' + str(spec['mondrian'])
            + ' reg_k=' + str(spec['reg_k']) + ' reg_lambda=' + str(spec['reg_lambda'])
            + ' U=' + describe_u(spec['u']))


def load_S_table(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def matching_mask(table, key):
    """Boolean mask of table rows belonging to the configuration `key`."""
    mask = np.ones(len(table), dtype=bool)
    for name, value in key.items():
        if name not in table.columns:
            return np.zeros(len(table), dtype=bool)
        column = table[name]
        if name in ('reg_k', 'reg_lambda', 'u'):
            if pd.isna(value):
                mask &= column.isna().to_numpy()
                continue
            mask &= np.isclose(column.astype(float).to_numpy(), value)
        else:
            mask &= (column == value).to_numpy()
    return mask


def is_done(table, spec, n_iterations):
    """True if S_table already holds all windows for this spec with >= n_iterations redraws."""
    if table is None or len(table) == 0 or 'n_iterations' not in table.columns:
        return False
    mask = matching_mask(table, table_key(spec))
    if mask.sum() != len(WINDOWS):
        return False
    return table.loc[mask, 'n_iterations'].min() >= n_iterations


def update_S_table(path, key, new_df):
    """Replace this configuration's rows and write atomically (safe if the job is killed)."""
    table = load_S_table(path)
    if table is not None:
        table = table[~matching_mask(table, key)]
        new_df = pd.concat([table, new_df], ignore_index=True)
    tmp_path = path + '.tmp'
    new_df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def write_S_table_views(path):
    """Write the 0-e summary CSV beside the full resumable S_table.csv."""
    table = load_S_table(path)
    if table is None:
        return

    output_dir = os.path.dirname(path)
    summary_path = os.path.join(output_dir, 'S_table_0-e.csv')
    summary = table.loc[table['window'] == '0-e'].copy()
    tmp_path = summary_path + '.tmp'
    summary.to_csv(tmp_path, index=False)
    os.replace(tmp_path, summary_path)


def add_row_to_S_table(n_iterations, conf):
    """
    Compute truncated S over `n_iterations` calibration/test redraws and write
    one row per window into figures/S_table.csv (replacing any earlier rows for
    the same configuration).
    Returns (window_labels, S_matrix), S_matrix of shape (n_iterations, n_windows).
    Row i used seed i for every config, so matrices from different configs are paired.
    """
    spec = spec_from_conf(conf)
    key = table_key(spec)

    computation_start = time.perf_counter()
    results = run_parallel_iterations(
        conf, compute_iteration, n_iterations, desc=describe(spec)
    )
    computation_time_seconds = time.perf_counter() - computation_start

    S_matrix = np.array([[r[label] for label in WINDOW_LABELS] for r in results])
    coverage_values = np.asarray([result['coverage'] for result in results], dtype=float)
    coverage_summary = summarise_coverage(coverage_values, alpha=conf.alpha)

    # Raw per-redraw values (columns follow WINDOW_LABELS), for paired comparisons later.
    raw_dir = os.path.join(conf.evaluation_dir, 'figures', 'S_raw')
    os.makedirs(raw_dir, exist_ok=True)
    raw_key_columns = (
        'dataset', 'model_architecture', 'domain', 'score_function',
        'distance_metric', 'mondrian', 'reg_k', 'reg_lambda', 'u',
    )
    tag = '_'.join([str(key[name]) for name in raw_key_columns])
    np.save(os.path.join(raw_dir, tag + '.npy'), S_matrix)

    bp = breakpoint_values(conf.top1_accuracy)
    window_widths = np.array([bp[hi] - bp[lo] for lo, hi in WINDOWS])
    S_per_unit_window = S_matrix / window_widths[np.newaxis, :]
    new_df = pd.DataFrame([
        dict(key,
             top1_accuracy=conf.top1_accuracy,
             top5_accuracy=conf.top5_accuracy,
             window=lo + '-' + hi,
             lower=bp[lo],
             upper=bp[hi],
             n_iterations=n_iterations,
             computation_time_seconds=computation_time_seconds,
             **summarise(S_per_unit_window[:, j]),
             **coverage_summary)
        for j, (lo, hi) in enumerate(WINDOWS)
    ])

    S_table_path = os.path.join(conf.evaluation_dir, 'figures', 'S_table.csv')
    os.makedirs(os.path.dirname(S_table_path), exist_ok=True)
    update_S_table(S_table_path, key, new_df)
    write_S_table_views(S_table_path)

    full = new_df.loc[new_df['window'] == '0-1', 'S_median'].item()
    i1_row = new_df.loc[new_df['window'] == '0-e'].iloc[0]
    print("Median S per unit window (0, 1): %.4f   |   (0, e) [I_1-style]: %.4f"
        % (full, i1_row['S_median']), flush=True)
    print("Computation time: %.3f seconds" % computation_time_seconds, flush=True)

    return WINDOW_LABELS, S_matrix


class ConformalPredictionEvaluator():

    def __init__(self, results_df, score_function, distance_metric, alpha, mondrian, n_classes):
        self.results_df = results_df
        self.score_function = score_function
        self.distance_metric = distance_metric
        self.alpha = alpha
        self.mondrian = mondrian
        self.n_classes = n_classes

    def get_accuracy(self, mondrian=False, print_acc=False):
        """Get accuracy metrics for prediction regions."""

        if self.results_df.empty:
            print("No results available. Run calibrate() and predict() on the ConformalPrediction instance first.")
            return

        if not mondrian:
            overall_correct = 0
            overall_empty = 0
            overall_count = 0
            overall_size = 0
        else:
            accuracy_per_class = {i: 0 for i in range(self.n_classes)}
            avg_size_per_class = {i: 0 for i in range(self.n_classes)}

        if print_acc:
            print(f"Accuracy Results:")
            print("-" * 50)

        n_classes = len(np.unique(self.results_df['label']))

        if mondrian:
            for i in range(n_classes):
                prediction_regions_label = self.results_df[self.results_df['label'] == i]['prediction_region'].values
                count = sum(i in region for region in prediction_regions_label)
                accuracy = count / len(prediction_regions_label)
                avg_size = np.mean([len(region) for region in prediction_regions_label])
                empty = sum(len(region) == 0 for region in prediction_regions_label)

                if print_acc:
                    print(f'Label {i}: {100 * accuracy:.2f}% coverage ({count}/{len(prediction_regions_label)})')
                    print(f'  Average prediction set size: {avg_size:.2f}')
                    print(f'  Empty prediction sets: {empty} ({100 * empty / len(prediction_regions_label):.2f}%)')

                accuracy_per_class[i] = accuracy
                avg_size_per_class[i] = avg_size

            return accuracy_per_class, avg_size_per_class

        else:
            for label, prediction_region in zip(self.results_df['label'], self.results_df['prediction_region']):
                overall_count += 1
                overall_size += len(prediction_region)
                if len(prediction_region) == 0:
                    overall_empty += 1
                if label in prediction_region:
                    overall_correct += 1

            overall_accuracy = overall_correct / overall_count
            overall_avg_size = overall_size / overall_count

            if print_acc:
                print(f'Overall coverage: {100 * overall_accuracy:.2f}%')
                print(f'Overall average prediction set size: {overall_avg_size:.2f}')
                print(f'Overall empty prediction sets: {overall_empty} ({100 * overall_empty / overall_count:.2f}%)')

            return overall_accuracy, overall_avg_size

