from conformal_prediction import *
import joblib
import copy
from collections import Counter

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

# ---------------------------------------------------------------------------
# Sweep grid for run_all()  -- EDIT THIS to match the experiments you want
# ---------------------------------------------------------------------------
DATASETS = ['cifar10', 'cifar100', 'imagenet']
MODELS = ['resnet50', 'resnet18', 'efficientnet_b0', 'vit_b_16']

DOMAINS = ['probabilities', 'logits', 'features']
LABEL_DOMAINS = ['probabilities']
MARGIN_DOMAINS = ['probabilities', 'logits']
APS_DOMAINS = ['probabilities']
GRADIENT_DOMAINS = ['features']

DISTANCES = ['euclidean', 'cosine']
APS_DISTANCES = ['euclidean']
GRADIENT_DISTANCES = ['euclidean']

MONDRIAN_VALUES = [False]

# Small defaults for a quick regularization sweep; pass custom grids to
# run_regularization_sweep() for a finer search.
RAPS_REG_K_GRID = [1,2,3]
RAPS_REG_LAMBDA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SAPS_REG_LAMBDA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

SCORE_SPECS = {
    'label':         (LABEL_DOMAINS, DISTANCES),
    'mean':          (DOMAINS, DISTANCES),
    'margin':        (MARGIN_DOMAINS, DISTANCES),
    'gradient':      (GRADIENT_DOMAINS, GRADIENT_DISTANCES),
    'fast_gradient': (GRADIENT_DOMAINS, GRADIENT_DISTANCES),
    'aps':           (APS_DOMAINS, APS_DISTANCES),
    'raps':          (APS_DOMAINS, APS_DISTANCES),
    'saps':          (APS_DOMAINS, APS_DISTANCES),
}


# Columns identifying one experimental configuration (a re-run replaces its rows).
CONFIG_KEYS = ['dataset', 'model_architecture', 'domain', 'score_function',
               'distance_metric', 'mondrian', 'reg_k', 'reg_lambda']


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
            output_suffix = '_T_' + temperature_tag
            self.n_calib = 30000 if temperature_tag == 'calibrated' else 40000
        else:
            if temperature_tag != '1_0':
                raise ValueError("Temperature options other than 1.0 apply only to ImageNet ResNet50")
            output_suffix = ''

        is_imbalanced = self.simulate_class_imbalance
        output_suffix += '_imbalanced' if is_imbalanced else ''

        # Load static data once (npz is lazy; arrays are read on first use)
        output_stem = self.dataset + '_' + self.model_architecture
        non_imbalance_suffix = output_suffix[:-len('_imbalanced')] if output_suffix.endswith('_imbalanced') else output_suffix
        balanced_outputs_path = os.path.join(
            self.evaluation_dir, output_stem + '_outputs' + non_imbalance_suffix + '.npz'
        )
        balanced_metrics_path = os.path.join(
            self.evaluation_dir, output_stem + '_metrics' + non_imbalance_suffix + '.npz'
        )
        imbalanced_outputs_path = os.path.join(
            self.evaluation_dir, output_stem + '_outputs' + non_imbalance_suffix + '_imbalanced.npz'
        )
        imbalanced_metrics_path = os.path.join(
            self.evaluation_dir, output_stem + '_metrics' + non_imbalance_suffix + '_imbalanced.npz'
        )

        outputs_path = balanced_outputs_path if not is_imbalanced else imbalanced_outputs_path
        metrics_path = balanced_metrics_path if not is_imbalanced else imbalanced_metrics_path

        mismatch_ok = False
        if is_imbalanced:
            mismatch_ok = ((not os.path.exists(outputs_path) or not os.path.exists(metrics_path))
                           and os.path.exists(balanced_outputs_path) and os.path.exists(balanced_metrics_path))
        else:
            mismatch_ok = ((not os.path.exists(outputs_path) or not os.path.exists(metrics_path))
                           and os.path.exists(imbalanced_outputs_path) and os.path.exists(imbalanced_metrics_path))
        if mismatch_ok:
            raise FileNotFoundError(
                "Requested imbalance configuration does not match the available evaluation files. "
                f"Expected {outputs_path} and {metrics_path}, but the opposite artifact set exists instead: "
                f"{balanced_outputs_path}/{balanced_metrics_path} vs {imbalanced_outputs_path}/{imbalanced_metrics_path}. "
                "Set config['evaluation']['simulate_class_imbalance'] to match the file suffix on disk."
            )
        if not os.path.exists(outputs_path) or not os.path.exists(metrics_path):
            raise FileNotFoundError(
                "Temperature-specific output/metrics NPZs are missing: %s and %s. "
                "Run model_eval_and_save_features.py first."
                % (outputs_path, metrics_path)
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
                   calibration_minority_fraction=1.0, imbalance_seed=123):
        """
        Create a random calibration/test split from a dict of plain numpy arrays.

        If `simulate_class_imbalance` is enabled, the ordinary calibration/test
        partition is created first. Calibration can then subsample minority-class
        examples within its fixed partition; the test indices remain unchanged.
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
                n_keep = int(round(calibration_minority_fraction * len(class_positions)))
                n_keep = min(len(class_positions), max(0, n_keep))
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
        }

    def create_split(self, random_seed):
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
        )

    # ---- Worker payload ------------------------------------------------

    def to_worker_dict(self):
        """
        Bundle everything a joblib worker process needs into one picklable
        dict. Reads the current attribute values, so mutating
        conf.score_function / conf.distance_metric between calls works.
        """
        data_arrays = self._needed_arrays()
        minority_class_ids = (
            self._minority_class_ids(
                data_arrays['labels'],
                self.minority_class_fraction,
                self.imbalance_seed,
                classes=np.arange(data_arrays['probabilities'].shape[1]),
            )
            if self.simulate_class_imbalance else set()
        )
        return {
            'alpha': self.alpha,
            'data_arrays': data_arrays,
            'n_classes': self.n_classes,
            'distance_metric': self.distance_metric,
            'score_function': self.score_function,
            'mondrian': self.mondrian,
            'reg_k': self.reg_k,
            'reg_lambda': self.reg_lambda,
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
            'minority_class_ids': minority_class_ids,
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


def run_cp_once(alpha, calibration_data, calibration_labels, calibration_preds, test_data, test_labels,
                n_classes, distance_metric, score_function, mondrian, reg_k, reg_lambda, model_architecture, dataset):

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
        model_architecture=model_architecture,
        dataset=dataset
    )
    cp.compute_scores()

    return cp.predict_with_scores(alpha=alpha)


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


def find_S_single_iteration(alpha, calibration_data, calibration_labels, calibration_preds,
                            test_data, test_labels, n_classes, distance_metric,
                            score_function, mondrian, reg_k, reg_lambda, top1_accuracy,
                            model_architecture=None, dataset=None):
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
        model_architecture=model_architecture,
        dataset=dataset
    )
    cp.compute_scores()

    F = cumulative_S(cp, breakpoint_values(top1_accuracy))
    return {lo + '-' + hi: F[hi] - F[lo] for lo, hi in WINDOWS}


def compute_iteration(random_seed, conf_data):
    """Worker: truncated S for every window for one calibration/test split."""
    split = ConformalConfig.make_split(
        random_seed,
        conf_data['data_arrays'],
        conf_data['n_calib'],
        conf_data['conformal_domain'],
        simulate_class_imbalance=conf_data.get('simulate_class_imbalance', False),
        minority_class_fraction=conf_data.get('minority_class_fraction', 0.10),
        calibration_minority_fraction=conf_data.get('calibration_minority_fraction', 1.0),
        imbalance_seed=conf_data.get('imbalance_seed', 123),
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
        dataset=conf_data['dataset']
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


# ---------------------------------------------------------------------------
# S_table bookkeeping
# ---------------------------------------------------------------------------
# A "spec" is a dict with: dataset, model_architecture, conformal_domain,
# score_function, distance_metric, mondrian, reg_k, reg_lambda.

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
    }


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
    }


def describe(spec):
    return (spec['dataset'] + ' ' + spec['model_architecture'] + ' ' + spec['conformal_domain'] + ' '
            + spec['score_function'] + ' ' + spec['distance_metric'] + ' mondrian=' + str(spec['mondrian']))


def load_S_table(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def matching_mask(table, key):
    """Boolean mask of table rows belonging to the configuration `key`."""
    mask = np.ones(len(table), dtype=bool)
    for name, value in key.items():
        if name not in table.columns:      # table from an older schema: no match
            return np.zeros(len(table), dtype=bool)
        column = table[name]
        if name in ('reg_k', 'reg_lambda'):
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


def add_row_to_S_table(n_iterations, conf=None):
    """
    Compute truncated S over `n_iterations` calibration/test redraws and write
    one row per window into figures/S_table.csv (replacing any earlier rows for
    the same configuration). With conf=None the settings come from config.yaml.
    Returns (window_labels, S_matrix), S_matrix of shape (n_iterations, n_windows).
    Row i used seed i for every config, so matrices from different configs are paired.
    """
    if conf is None:
        conf = ConformalConfig()
    spec = spec_from_conf(conf)
    key = table_key(spec)

    results = run_parallel_iterations(
        conf, compute_iteration, n_iterations, desc=describe(spec)
    )

    S_matrix = np.array([[r[label] for label in WINDOW_LABELS] for r in results])

    # Raw per-redraw values (columns follow WINDOW_LABELS), for paired comparisons later.
    raw_dir = os.path.join(conf.evaluation_dir, 'figures', 'S_raw')
    os.makedirs(raw_dir, exist_ok=True)
    tag = '_'.join([str(key[name]) for name in CONFIG_KEYS])
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
             **summarise(S_per_unit_window[:, j]))
        for j, (lo, hi) in enumerate(WINDOWS)
    ])

    S_table_path = os.path.join(conf.evaluation_dir, 'figures', 'S_table.csv')
    os.makedirs(os.path.dirname(S_table_path), exist_ok=True)
    update_S_table(S_table_path, key, new_df)

    full = new_df.loc[new_df['window'] == '0-1', 'S_median'].item()
    i1_row = new_df.loc[new_df['window'] == '0-e'].iloc[0]
    print("Median S per unit window (0, 1): %.4f   |   (0, e) [I_1-style]: %.4f"
        % (full, i1_row['S_median']), flush=True)

    return WINDOW_LABELS, S_matrix


# ---------------------------------------------------------------------------
# Sweep over all configurations (resumable)
# ---------------------------------------------------------------------------

def build_plan(reg_k, reg_lambda):
    """All configurations to run, ordered so that consecutive specs share (dataset, model, domain)."""
    plan = []
    for dataset in DATASETS:
        for model in MODELS:
            for domain in DOMAINS:
                for score, (domains, distances) in SCORE_SPECS.items():
                    if domain not in domains:
                        continue
                    for distance in distances:
                        for mondrian in MONDRIAN_VALUES:
                            plan.append({
                                'dataset': dataset,
                                'model_architecture': model,
                                'conformal_domain': domain,
                                'score_function': score,
                                'distance_metric': distance,
                                'mondrian': mondrian,
                                'reg_k': reg_k,
                                'reg_lambda': reg_lambda,
                            })
    return plan


def run_all(n_iterations=100, config_path="config.yaml", dry_run=False):
    """
    Run every configuration in the grid that is not yet in S_table.csv.
    Safe to restart: finished configurations are skipped, and the table is
    rewritten atomically after each one. dry_run=True only lists what is pending.
    """
    with open(config_path, 'r') as f:
        base = yaml.safe_load(f)
    cp_cfg = base['conformal_prediction']
    S_table_path = os.path.join(base['evaluation']['output_directory'], 'figures', 'S_table.csv')

    plan = build_plan(cp_cfg['reg_k'], cp_cfg['reg_lambda'])
    table = load_S_table(S_table_path)
    pending = [spec for spec in plan if not is_done(table, spec, n_iterations)]

    print("%d configurations planned, %d already in S_table, %d to run"
          % (len(plan), len(plan) - len(pending), len(pending)), flush=True)

    if dry_run:
        for spec in pending:
            print("  todo: " + describe(spec))
        return

    failures = []
    conf = None
    current_group = None
    n_done = 0

    for i, spec in enumerate(pending, 1):
        group = (spec['dataset'], spec['model_architecture'], spec['conformal_domain'])
        if group != current_group:
            conf = None      # release the previous group's arrays before loading the next
            current_group = group
            try:
                conf = ConformalConfig(
                    config_path,
                    dataset=spec['dataset'],
                    model_architecture=spec['model_architecture'],
                    conformal_domain=spec['conformal_domain'],
                )
            except Exception as e:
                print("FAILED to load %s: %s" % (group, repr(e)), flush=True)

        if conf is None:
            failures.append((describe(spec), 'could not load data'))
            continue

        conf.score_function = spec['score_function']
        conf.distance_metric = spec['distance_metric']
        conf.mondrian = spec['mondrian']
        conf.reg_k = spec['reg_k']
        conf.reg_lambda = spec['reg_lambda']

        print("[%d/%d] %s" % (i, len(pending), describe(spec)), flush=True)
        try:
            add_row_to_S_table(n_iterations, conf=conf)
            n_done += 1
        except Exception as e:
            print("FAILED: %s: %s" % (describe(spec), repr(e)), flush=True)
            failures.append((describe(spec), repr(e)))

    print("Finished: %d computed, %d failed" % (n_done, len(failures)), flush=True)
    for name, reason in failures:
        print("  failed: %s -> %s" % (name, reason))


def compute_prevalence_iteration(random_seed, conf_data):
    """Worker: compute minority-class prevalence for one calibration/test split."""
    split = ConformalConfig.make_split(
        random_seed,
        conf_data['data_arrays'],
        conf_data['n_calib'],
        conf_data['conformal_domain'],
        simulate_class_imbalance=conf_data.get('simulate_class_imbalance', False),
        minority_class_fraction=conf_data.get('minority_class_fraction', 0.10),
        calibration_minority_fraction=conf_data.get('calibration_minority_fraction', 1.0),
        imbalance_seed=conf_data.get('imbalance_seed', 123),
    )

    results_df = run_cp_once(
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
        model_architecture=conf_data['model_architecture'],
        dataset=conf_data['dataset']
    )

    evaluator = ConformalPredictionEvaluator(
        results_df,
        conf_data['score_function'],
        conf_data['distance_metric'],
        conf_data['alpha'],
        conf_data['mondrian'],
        conf_data['n_classes']
    )

    return evaluator.prevalence_of_minority_classes(
        conf_data.get('minority_class_ids', set())
    )


def compute_prevalence_of_minority_classes(n_iterations):
    conf = ConformalConfig()

    results = run_parallel_iterations(
        conf, compute_prevalence_iteration, n_iterations,
        desc="Computing prevalence of minority classes"
    )

    true_proportions, expected_proportions = zip(*results)
    median_true_proportion = np.median(true_proportions)
    median_expected_proportion = np.median(expected_proportions)

    return median_true_proportion, median_expected_proportion


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

    def prevalence_of_minority_classes(self, minority_classes):
        """Compare prediction-region prevalence for the configured minority classes.

        Minority class identities come from the seeded imbalance configuration,
        not from observed label frequencies, which may vary with the calibration
        sampling fraction.
        """
        minority_classes = set(minority_classes)
        print(f"Evaluating prevalence of minority classes: {minority_classes}")
        if not minority_classes or self.results_df.empty:
            return 0.0, 0.0

        labels = self.results_df['label'].to_numpy()
        prediction_regions = self.results_df['prediction_region'].tolist()
        n_regions = len(prediction_regions)

        # Single pass over all prediction regions, tallying how many times
        # each class appears anywhere in a region. Previously this rescanned
        # the full list of prediction regions once per minority class
        # (O(minority_classes * n_regions)); this does it in one pass
        # (O(n_regions)) and just looks up the counts we need afterward.
        class_region_counts = Counter()
        for region in prediction_regions:
            class_region_counts.update(region)

        true_proportions = []
        expected_proportions = []
        for minority_class in minority_classes:
            expected_proportions.append(np.mean(labels == minority_class))
            true_proportions.append(class_region_counts.get(minority_class, 0) / n_regions)

        return float(np.mean(true_proportions)), float(np.mean(expected_proportions))

    def _pvalue_matrix(self):
        return self.results_df[[f'pvalue_class_{c}' for c in range(self.n_classes)]].to_numpy()

    def vovk_s_criterion(self):
        p = self._pvalue_matrix()
        s_values = p.sum(axis=1)
        return {'s_per_example': s_values, 's_mean': s_values.mean()}

    def window_S(self, lower, upper, normalize=False):
        """
        Raw integral of mean |P(alpha)| over [lower, upper] (= truncated S).
        normalize=True divides by (upper - lower), giving the mean set size over the window.
        """
        integral = (np.clip(self._pvalue_matrix(), lower, upper) - lower).sum(axis=1).mean()
        return integral / (upper - lower) if normalize else integral

    def i1_closed_form(self, upper, lower):
        """
        Integral of mean |P(alpha)| over [lower, upper], divided by `upper`
        (this matches the old trapezoid loop, which divided by 1 - accuracy).
        For lower = 0 this equals the mean set size over (0, upper).
        """
        integral = (np.clip(self._pvalue_matrix(), lower, upper) - lower).sum(axis=1).mean()
        return integral / upper

    def truncated_S_curve(self, P, U_grid):
        """Return S(U) over ``U_grid``; plotting is handled by the caller."""
        P = np.asarray(P)
        U_grid = np.asarray(U_grid)
        return np.array([np.minimum(P, U).sum(axis=1).mean() for U in U_grid])



if __name__ == "__main__":

    # Do a quick minority-class prevalence check to verify that the imbalance simulation is working as expected.
    # n_iterations = 100
    # median_true_proportion, median_expected_proportion = compute_prevalence_of_minority_classes(n_iterations)
    # print(f"Median true proportion of minority classes in prediction regions: {median_true_proportion:.4f}")
    # print(f"Median expected proportion of minority classes in the dataset: {median_expected_proportion:.4f}")

    # Run CP once
    conf = ConformalConfig()
    split = conf.create_split(random_seed=42)
    results_df = run_cp_once(
        conf.alpha,
        split['calibration_data'],
        split['calibration_labels'],
        split['calibration_preds'],
        split['test_data'],
        split['test_labels'],
        conf.n_classes,
        conf.distance_metric,
        conf.score_function,
        conf.mondrian,
        conf.reg_k,
        conf.reg_lambda,
        model_architecture=conf.model_architecture,
        dataset=conf.dataset
    )

    evaluator = ConformalPredictionEvaluator(
        results_df,
        conf.score_function,
        conf.distance_metric,
        conf.alpha,
        conf.mondrian,
        conf.n_classes
    )

    # Get accuracy metrics
    overall_accuracy, overall_avg_size = evaluator.get_accuracy(mondrian=conf.mondrian, print_acc=True)

    true_proportions, expected_proportions = evaluator.prevalence_of_minority_classes(conf._minority_class_ids(
        conf._needed_arrays()['labels'],
        conf.minority_class_fraction,
        conf.imbalance_seed,
        classes=np.arange(conf._needed_arrays()['probabilities'].shape[1]),
    ))
    print(f"Median true proportion of minority classes in prediction regions: {true_proportions:.4f}")
    print(f"Median expected proportion of minority classes in the dataset: {expected_proportions:.4f}")