from conformal_prediction import *
import joblib
import copy


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
    
    def create_split(self, random_seed):
        """Create a random calibration/test split"""
        np.random.seed(random_seed)
        indices = np.random.permutation(len(self.data['labels']))
        
        data_shuffled = {key: self.data[key][indices] for key in self.data.files}
        
        return {
            'calibration_data': data_shuffled[self.conformal_domain][:self.n_calib],
            'calibration_labels': data_shuffled['labels'][:self.n_calib],
            'calibration_preds': data_shuffled['probabilities'][:self.n_calib].argmax(axis=1),
            'test_data': data_shuffled[self.conformal_domain][self.n_calib:],
            'test_labels': data_shuffled['labels'][self.n_calib:]
        }

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
    """
    Worker function that runs a single iteration with pre-loaded config data.
    conf_data is a dict containing all necessary config values (picklable).
    """
    alpha = conf_data['alpha']
    data_arrays = conf_data['data_arrays']
    n_classes = conf_data['n_classes']
    distance_metric = conf_data['distance_metric']
    score_function = conf_data['score_function']
    mondrian = conf_data['mondrian']
    reg_k = conf_data['reg_k']
    reg_lambda = conf_data['reg_lambda']
    top1_accuracy = conf_data['top1_accuracy']
    n_calib = conf_data['n_calib']
    conformal_domain = conf_data['conformal_domain']

    np.random.seed(random_seed)
    indices = np.random.permutation(len(data_arrays['labels']))
    data_shuffled = {key: data_arrays[key][indices] for key in data_arrays.keys()}

    split = {
        'calibration_data': data_shuffled[conformal_domain][:n_calib],
        'calibration_labels': data_shuffled['labels'][:n_calib],
        'calibration_preds': data_shuffled['probabilities'][:n_calib].argmax(axis=1),
        'test_data': data_shuffled[conformal_domain][n_calib:],
        'test_labels': data_shuffled['labels'][n_calib:]
    }

    return find_I_single_iteration(
        alpha,
        split['calibration_data'],
        split['calibration_labels'],
        split['calibration_preds'],
        split['test_data'],
        split['test_labels'],
        n_classes,
        distance_metric,
        score_function,
        mondrian,
        reg_k,
        reg_lambda,
        top1_accuracy
    )


def add_row_to_I_table(n_iterations):
    conf = ConformalConfig()
    data_arrays = {key: np.array(conf.data[key]) for key in conf.data.files}
    conf_data = {
        'alpha': conf.alpha,
        'data_arrays': data_arrays,
        'n_classes': conf.n_classes,
        'distance_metric': conf.distance_metric,
        'score_function': conf.score_function,
        'mondrian': conf.mondrian,
        'reg_k': conf.reg_k,
        'reg_lambda': conf.reg_lambda,
        'top1_accuracy': conf.top1_accuracy,
        'top5_accuracy': conf.top5_accuracy,
        'n_calib': conf.n_calib,
        'conformal_domain': conf.conformal_domain,
    }

    I_values = joblib.Parallel(n_jobs=conf.n_workers)(
        joblib.delayed(compute_iteration)(random_seed, conf_data)
        for random_seed in tqdm(range(n_iterations), desc="Computing I values")
    )
    I_values = list(I_values)

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
    I_table.to_csv(I_table_path, index=False)

def compute_prevalence_iteration(random_seed, conf_data):
    alpha = conf_data['alpha']
    data_arrays = conf_data['data_arrays']
    n_classes = conf_data['n_classes']
    distance_metric = conf_data['distance_metric']
    score_function = conf_data['score_function']
    mondrian = conf_data['mondrian']
    reg_k = conf_data['reg_k']
    reg_lambda = conf_data['reg_lambda']
    top1_accuracy = conf_data['top1_accuracy']
    n_calib = conf_data['n_calib']
    conformal_domain = conf_data['conformal_domain']

    np.random.seed(random_seed)
    indices = np.random.permutation(len(data_arrays['labels']))
    data_shuffled = {key: data_arrays[key][indices] for key in data_arrays.keys()}

    split = {
        'calibration_data': data_shuffled[conformal_domain][:n_calib],
        'calibration_labels': data_shuffled['labels'][:n_calib],
        'calibration_preds': data_shuffled['probabilities'][:n_calib].argmax(axis=1),
        'test_data': data_shuffled[conformal_domain][n_calib:],
        'test_labels': data_shuffled['labels'][n_calib:]
    }

    results_df = run_cp_once(
        alpha,
        split['calibration_data'],
        split['calibration_labels'],
        split['calibration_preds'],
        split['test_data'],
        split['test_labels'],
        n_classes,
        distance_metric,
        score_function,
        mondrian,
        reg_k,
        reg_lambda,
        parallel=False,
        n_workers=1
    )

    evaluator = ConformalPredictionEvaluator(
        results_df,
        score_function,
        distance_metric,
        alpha,
        mondrian,
        n_classes
    )

    return evaluator.prevalence_of_minority_classes()

def compute_prevalence_of_minority_classes(n_iterations):

    conf = ConformalConfig()
    data_arrays = {key: np.array(conf.data[key]) for key in conf.data.files}
    conf_data = {
        'alpha': conf.alpha,
        'data_arrays': data_arrays,
        'n_classes': conf.n_classes,
        'distance_metric': conf.distance_metric,
        'score_function': conf.score_function,
        'mondrian': conf.mondrian,
        'reg_k': conf.reg_k,
        'reg_lambda': conf.reg_lambda,
        'top1_accuracy': conf.top1_accuracy,
        'top5_accuracy': conf.top5_accuracy,
        'n_calib': conf.n_calib,
        'conformal_domain': conf.conformal_domain,
    }

    results = joblib.Parallel(n_jobs=conf.n_workers)(
        joblib.delayed(compute_prevalence_iteration)(random_seed, conf_data)
        for random_seed in tqdm(range(n_iterations), desc="Computing prevalence of minority classes")
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
        
        prediction_regions = self.results_df['prediction_region'].tolist()
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
            for prediction_region in prediction_regions:
                overall_count += 1
                overall_size += len(prediction_region)
                if len(prediction_region) == 0:
                    overall_empty += 1
                if self.results_df.iloc[overall_count - 1]['label'] in prediction_region:
                    overall_correct += 1

            overall_accuracy = overall_correct / overall_count
            overall_avg_size = overall_size / overall_count

            if print_acc:
                print(f'Overall coverage: {100 * overall_accuracy:.2f}%')
                print(f'Overall average prediction set size: {overall_avg_size:.2f}')
                print(f'Overall empty prediction sets: {overall_empty} ({100 * overall_empty / overall_count:.2f}%)')

            return overall_accuracy, overall_avg_size

    def size_stratified_coverage_violation(self):
        """Compute the size-stratified coverage violation as described in equation (5) in https://arxiv.org/pdf/2009.14193"""

        def create_adaptive_bins(min_bin_size=100, max_bins=10):
            """Create approximately equal sized bins"""

            sizes = self.results_df['prediction_region'].apply(len)
            bins = []

            low = 1

            quartile = None

            while len(bins) < max_bins:

                # If this is the last bin, just take everything remaining
                if len(bins) == max_bins-1:
                    bins.append((low, self.n_classes))
                    return bins

                for high in range(low, self.n_classes+1):

                    # Check how many examples are available to go into the next bin to see if this is the last bin
                    n_remaining_after = (sizes > high).sum()

                    # If not enough for even one bin, just make one final bin
                    if n_remaining_after < min_bin_size:
                        bins.append((low, self.n_classes))
                        return bins

                    if not quartile:
                        max_remaining_bins = min(max_bins-len(bins), n_remaining_after//min_bin_size)
                        quartile = n_remaining_after//max_remaining_bins

                    # Randomize so bin size sometimes is just below quartile, sometimes above it
                    n_in_bin = ((sizes >= low) & (sizes <= high + np.random.randint(2))).sum()

                    if n_in_bin >= quartile:
                        bins.append((low, high))
                        break

                if high==self.n_classes:
                    bins.append((low,high))
                    return bins

                low = high+1

            return bins

        bins = create_adaptive_bins()

        SSCV = 0

        for low, high in bins:

            sizes = self.results_df['prediction_region'].apply(len)
            result_bin = self.results_df[(sizes >= low) & (sizes <= high)]

            J = len(result_bin)
            if J==0: continue

            covered = result_bin.apply(
                lambda row: row['label'] in row['prediction_region'],
                axis=1
            )

            I = covered.sum()

            value = abs(I/J-(1-self.alpha))


            if value>SSCV:
                SSCV = value
        
        return SSCV
        
    def prevalence_of_minority_classes(self):

        counts = self.results_df["label"].value_counts()
        max_count = counts.max()
        minority_classes = counts[counts<0.3*max_count].index.tolist()

        prediction_regions = self.results_df['prediction_region'].tolist()

        mean_true_proportion = 0
        mean_expected_proportion = 0

        for minority_class in minority_classes:
            expected_proportion = counts[minority_class]/counts.sum()

            count_class = 0

            for prediction_region in prediction_regions:
                if minority_class in prediction_region:
                    count_class += 1

            true_proportion = count_class / len(prediction_regions)

            mean_true_proportion += true_proportion/len(minority_classes)
            mean_expected_proportion += expected_proportion/len(minority_classes)

        return mean_true_proportion, mean_expected_proportion
        



if __name__ == "__main__":

    # # Create plot
    # conf = ConformalConfig()

    # split = conf.create_split(random_seed=42)

    # create_size_over_alpha_graph(
    #     split['calibration_data'],
    #     split['calibration_labels'],
    #     split['calibration_preds'],
    #     split['test_data'],
    #     split['test_labels'],
    #     conf.n_classes,
    #     conf.distance_metric,
    #     conf.score_function,
    #     conf.mondrian,
    #     conf.reg_k,
    #     conf.reg_lambda,
    #     conf.top1_accuracy
    # )

    # find_I_single_iteration(
    #     conf.alpha,
    #     split['calibration_data'],
    #     split['calibration_labels'],
    #     split['calibration_preds'],
    #     split['test_data'],
    #     split['test_labels'],
    #     conf.n_classes,
    #     conf.distance_metric,
    #     conf.score_function,
    #     conf.mondrian,
    #     conf.reg_k,
    #     conf.reg_lambda,
    #     conf.top1_accuracy
    # )

    #add_row_to_I_table(n_iterations=100)

    # prevalence of minority classes
    median_true_proportion, median_expected_proportion = compute_prevalence_of_minority_classes(n_iterations=100)
    print(f'Median true proportion of minority classes: {median_true_proportion:.4f}')
    print(f'Median expected proportion of minority classes: {median_expected_proportion:.4f}')