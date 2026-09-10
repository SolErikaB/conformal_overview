from conformal_prediction import *
import joblib
import copy
from collections import Counter


class ConformalConfig:
    """Holds configuration and static data that doesn't change between runs"""

    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.evaluation_dir = self.config['evaluation']['output_directory']
        self.dataset = self.config['conformal_prediction']['dataset']
        self.model_architecture = self.config['conformal_prediction']['model_architecture']
        self.mondrian = self.config['conformal_prediction'].get('mondrian', False)

        # Load static data once
        self.data = np.load(
            os.path.join(self.evaluation_dir,
                        f'{self.dataset}_{self.model_architecture}_outputs.npz')
        )

        # Configuration parameters
        self.conformal_domain = self.config['conformal_prediction']['conformal_domain']
        self.score_function = self.config['conformal_prediction']['score_function']
        self.distance_metric = self.config['conformal_prediction']['distance_metric']
        self.alpha = self.config['conformal_prediction']['alpha']
        self.reg_k = self.config['conformal_prediction']['reg_k']
        self.reg_lambda = self.config['conformal_prediction']['reg_lambda']
        self.n_workers = self.config['conformal_prediction']['n_workers']

        # Load metrics
        metrics_path = os.path.join(
            self.evaluation_dir,
            f"{self.dataset}_{self.model_architecture}_metrics.npz"
        )
        metrics = np.load(metrics_path)
        self.top1_accuracy = metrics['top_1_accuracy']
        self.top5_accuracy = metrics['top_5_accuracy']

        self.n_classes = np.unique(self.data['labels']).shape[0]
        self.n_calib = 40000 if self.dataset == 'imagenet' else 10000

    # ---- Splitting ---------------------------------------------------

    @staticmethod
    def make_split(random_seed, data_arrays, n_calib, conformal_domain):
        """
        Create a random calibration/test split from a dict of plain numpy
        arrays. This is the single source of truth for the split logic -
        both `create_split` (used in-process) and the joblib worker
        functions below (which only have picklable plain arrays, not a
        ConformalConfig instance with an open npz file handle) call this.
        """
        np.random.seed(random_seed)
        indices = np.random.permutation(len(data_arrays['labels']))
        shuffled = {key: arr[indices] for key, arr in data_arrays.items()}

        return {
            'calibration_data': shuffled[conformal_domain][:n_calib],
            'calibration_labels': shuffled['labels'][:n_calib],
            'calibration_preds': shuffled['probabilities'][:n_calib].argmax(axis=1),
            'test_data': shuffled[conformal_domain][n_calib:],
            'test_labels': shuffled['labels'][n_calib:],
        }

    def create_split(self, random_seed):
        """Create a random calibration/test split using this config's own data"""
        data_arrays = {key: np.array(self.data[key]) for key in self.data.files}
        return self.make_split(random_seed, data_arrays, self.n_calib, self.conformal_domain)

    # ---- Worker payload ------------------------------------------------

    def to_worker_dict(self):
        """
        Bundle everything a joblib worker process needs into one picklable
        dict. self.data is an npz NpzFile backed by an open file handle, so
        we can't hand `self` to a subprocess directly - unpack it into
        plain arrays once here instead.
        """
        return {
            'alpha': self.alpha,
            'data_arrays': {key: np.array(self.data[key]) for key in self.data.files},
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
        }


def run_parallel_iterations(conf, worker_fn, n_iterations, desc):
    """
    Run `worker_fn(random_seed, conf_data)` for random_seed in
    range(n_iterations), in parallel across conf.n_workers processes.
    Shared driver for add_row_to_I_table and
    compute_prevalence_of_minority_classes, which previously duplicated
    this exact Parallel/tqdm setup.
    """
    conf_data = conf.to_worker_dict()
    return joblib.Parallel(n_jobs=conf.n_workers)(
        joblib.delayed(worker_fn)(random_seed, conf_data)
        for random_seed in tqdm(range(n_iterations), desc=desc)
    )


def run_cp_once(alpha, calibration_data, calibration_labels, calibration_preds, test_data, test_labels,
                 n_classes, distance_metric, score_function, mondrian, reg_k, reg_lambda,
                 parallel=False, n_workers=1):
    """Run conformal prediction for a single alpha value."""

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
        reg_lambda=reg_lambda
    )
    cp.compute_scores()  # scores (calibration + test) computed once

    if not parallel or n_workers <= 1:
        return cp.predict(alpha=alpha)

    # Parallel path: only useful if per-point compute_score() itself is the
    # bottleneck (e.g. very large test sets). Thresholding is already cheap,
    # so we parallelize by splitting the cached test_scores matrix, not by
    # re-running compute_scores() per chunk.
    test_score_chunks = np.array_split(cp._test_scores, n_workers)
    test_label_chunks = np.array_split(np.asarray(test_labels), n_workers)
    thresholds = np.array(cp._thresholds_for_alpha(alpha))

    def process_chunk(scores_chunk, labels_chunk):
        mask = scores_chunk <= thresholds[None, :]
        regions = [np.flatnonzero(row).tolist() for row in mask]
        return pd.DataFrame({'label': labels_chunk, 'prediction_region': regions})

    results_dfs = joblib.Parallel(n_jobs=n_workers)(
        joblib.delayed(process_chunk)(test_score_chunks[i], test_label_chunks[i])
        for i in range(n_workers)
    )
    return pd.concat(results_dfs, ignore_index=True)


def find_I_single_iteration(alpha, calibration_data, calibration_labels, calibration_preds,
                             test_data, test_labels, n_classes, distance_metric,
                             score_function, mondrian, reg_k, reg_lambda, top1_accuracy):
    """
    I is the integral of set size over alpha, approximated via the trapezoidal rule.
    Nonconformity scores are computed once; each alpha only redoes thresholding.
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
        reg_lambda=reg_lambda
    )
    cp.compute_scores()

    values = []
    for a in np.linspace(0.001, 1 - top1_accuracy, 50):
        results_df = cp.predict(alpha=a)
        evaluator = ConformalPredictionEvaluator(results_df, score_function, distance_metric, a, mondrian, n_classes)
        _, avg_size = evaluator.get_accuracy()
        values.append((a, avg_size))

    I = 0.0
    for i in range(len(values) - 1):
        alpha1, size1 = values[i]
        alpha2, size2 = values[i + 1]
        I += (alpha2 - alpha1) * (size1 + size2) / 2

    I /= (1 - top1_accuracy)

    return I


def create_I_table():
    columns = [
        'model_architecture',
        'dataset',
        'top1_accuracy',
        'top5_accuracy',
        'domain',
        'score_function',
        'distance_metric',
        'mondrian',
        'I_value',
    ]
    return pd.DataFrame(columns=columns)


def compute_iteration(random_seed, conf_data):
    """Worker: compute a single I value for one calibration/test split."""
    split = ConformalConfig.make_split(
        random_seed,
        conf_data['data_arrays'],
        conf_data['n_calib'],
        conf_data['conformal_domain'],
    )

    return find_I_single_iteration(
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
        conf_data['top1_accuracy']
    )


def add_row_to_I_table(n_iterations):
    conf = ConformalConfig()

    I_values = list(run_parallel_iterations(
        conf, compute_iteration, n_iterations, desc="Computing I values"
    ))

    def spread_stats(values, prefix):
        arr = np.array(values)
        median = np.median(arr)
        p5, p25, p75, p95 = np.percentile(arr, [5, 25, 75, 95])
        return {
            f'{prefix}': median,
            f'{prefix}_std': np.std(arr, ddof=1),
            f'{prefix}_mad': np.median(np.abs(arr - median)),
            f'{prefix}_p5': p5,
            f'{prefix}_p25': p25,
            f'{prefix}_p75': p75,
            f'{prefix}_p95': p95,
            f'{prefix}_iqr': p75 - p25,
        }

    I_stats = spread_stats(I_values, 'I_value')
    median_I_value = I_stats['I_value']

    print(f'Median I value: {median_I_value}')

    I_table_path = os.path.join(conf.evaluation_dir, 'figures', 'I_table.csv')
    I_table = (pd.read_csv(I_table_path) if os.path.exists(I_table_path) else create_I_table())

    mask = (
        (I_table['model_architecture'] == conf.model_architecture) &
        (I_table['dataset'] == conf.dataset) &
        (I_table['domain'] == conf.conformal_domain) &
        (I_table['score_function'] == conf.score_function) &
        (I_table['distance_metric'] == conf.distance_metric) &
        (I_table['mondrian'] == conf.mondrian)
    )
    I_table = I_table[~mask]

    new_row = {
        'model_architecture': conf.model_architecture,
        'dataset': conf.dataset,
        'top1_accuracy': conf.top1_accuracy,
        'top5_accuracy': conf.top5_accuracy,
        'domain': conf.conformal_domain,
        'score_function': conf.score_function,
        'distance_metric': conf.distance_metric,
        'mondrian': conf.mondrian,
        **I_stats,
    }
    I_table = pd.concat([I_table, pd.DataFrame([new_row])], ignore_index=True)
    #I_table.to_csv(I_table_path, index=False)


def compute_prevalence_iteration(random_seed, conf_data):
    """Worker: compute minority-class prevalence for one calibration/test split."""
    split = ConformalConfig.make_split(
        random_seed,
        conf_data['data_arrays'],
        conf_data['n_calib'],
        conf_data['conformal_domain'],
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
        parallel=False,
        n_workers=1
    )

    evaluator = ConformalPredictionEvaluator(
        results_df,
        conf_data['score_function'],
        conf_data['distance_metric'],
        conf_data['alpha'],
        conf_data['mondrian'],
        conf_data['n_classes']
    )

    return evaluator.prevalence_of_minority_classes()


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


def create_size_over_alpha_graph(calibration_data, calibration_labels, calibration_preds, test_data, test_labels,
                                  n_classes, distance_metric, score_function, mondrian, reg_k, reg_lambda,
                                  top1_accuracy, steps=3000):

    cp = ConformalPrediction(
        alpha=0.001,  # placeholder; overridden per-call to predict()
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
        reg_lambda=reg_lambda
    )
    cp.compute_scores()  # expensive step, runs exactly once

    alphas = np.linspace(1/steps, 1, steps)
    values = []
    for alpha in tqdm(alphas, desc="Calculating size over alpha"):
        results_df = cp.predict(alpha=alpha)
        evaluator = ConformalPredictionEvaluator(results_df, score_function, distance_metric, alpha, mondrian, n_classes)
        _, avg_size = evaluator.get_accuracy()
        values.append((alpha, avg_size))

    # Full plot
    alphas, avg_sizes = zip(*values)
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, avg_sizes, label='Average Prediction Set Size', color='blue')
    plt.axvline(x=(1 - top1_accuracy) / 2, color='orange', linestyle='--', label='(1 - Top-1 Accuracy) / 2')
    plt.axvline(x=1 - top1_accuracy, color='red', linestyle='--', label='1 - Top-1 Accuracy')
    plt.title('Average Prediction Set Size vs Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Average Prediction Set Size')
    plt.legend()
    plt.grid()
    plt.savefig(f'size_over_alpha_{score_function}_{distance_metric}.png')
    plt.savefig(f'size_over_alpha_{score_function}_{distance_metric}.eps', format='eps')
    plt.close()

    # Zoomed plot: (1-top1_accuracy)/2 to (1-top1_accuracy)
    alphas_arr = np.array(alphas)
    lo, hi = (1 - top1_accuracy) / 2, 1 - top1_accuracy
    mask = (alphas_arr >= lo) & (alphas_arr <= hi)
    zoom_alphas = alphas_arr[mask]
    zoom_sizes = np.array(avg_sizes)[mask]

    plt.figure(figsize=(10, 6))
    plt.plot(zoom_alphas, zoom_sizes, label='Average Prediction Set Size', color='blue')
    plt.title('Average Prediction Set Size vs Alpha (Zoomed In)')
    plt.xlabel('Alpha')
    plt.ylabel('Average Prediction Set Size')
    plt.xlim(lo, hi)
    plt.legend()
    plt.grid()
    plt.savefig(f'size_over_alpha_zoomed_{score_function}_{distance_metric}.png')
    plt.savefig(f'size_over_alpha_zoomed_{score_function}_{distance_metric}.eps', format='eps')
    plt.close()


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

    def prevalence_of_minority_classes(self):

        counts = self.results_df["label"].value_counts()
        max_count = counts.max()
        minority_classes = counts[counts < 0.3 * max_count].index.tolist()

        if not minority_classes:
            return 0.0, 0.0

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
            expected_proportions.append(counts[minority_class] / counts.sum())
            true_proportions.append(class_region_counts.get(minority_class, 0) / n_regions)

        return float(np.mean(true_proportions)), float(np.mean(expected_proportions))


if __name__ == "__main__":

    add_row_to_I_table(n_iterations=100)