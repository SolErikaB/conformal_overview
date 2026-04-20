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

def run_cp_once(alpha, calibration_data, calibration_labels, calibration_preds, test_data, test_labels, n_classes, distance_metric, score_function, mondrian, reg_k, reg_lambda, parallel=False, n_workers=1):


    # Multiprocessing to speed up prediction
    # Split test data into chunks for each worker

    
    test_data_chunks = np.array_split(test_data, n_workers)
    test_labels_chunks = np.array_split(test_labels, n_workers)

    # Helper function to process each chunk
    def process_chunk(i):
        # Create a deep copy of the calibrated instance
        cp_chunk = copy.deepcopy(cp)
        cp_chunk.test_data = test_data_chunks[i]
        cp_chunk.test_labels = test_labels_chunks[i]
        
        results_df_chunk = cp_chunk.predict()
        return results_df_chunk

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

    cp.calibrate()

    if parallel:
        results_dfs = joblib.Parallel(n_jobs=n_workers)(
            joblib.delayed(process_chunk)(i) for i in range(n_workers)
        )
        # Combine results from all workers
        results_df = pd.concat(results_dfs, ignore_index=True)
    else:
        results_df = cp.predict()

    return results_df


def find_I_single_iteration(alpha, calibration_data, calibration_labels, calibration_preds, test_data, test_labels, n_classes, distance_metric, score_function, mondrian, reg_k, reg_lambda, n_workers, top1_accuracy, top5_accuracy):
    """P is the integral of the set size as a function of alpha. We can approximate it by computing the set sizes at different alphas and using the trapezoidal rule."""

    values = []

    for alpha in tqdm(np.linspace(0.001, 1-top1_accuracy, 50), desc="Calculating I value"):
    #for alpha in np.linspace(0.001, 1-top1_accuracy, 50):

        results_df = run_cp_once(
            alpha,
            calibration_data,
            calibration_labels,
            calibration_preds,
            test_data,
            test_labels,
            n_classes,
            distance_metric,
            score_function,
            mondrian,
            reg_k,
            reg_lambda,
            parallel=False,
            n_workers=1
        )
        evaluator = ConformalPredictionEvaluator(results_df, score_function, distance_metric, alpha, mondrian, n_classes)
        coverage, avg_size = evaluator.get_accuracy()
        sscv = evaluator.size_stratified_coverage_violation()

        # Store (alpha, avg_size) pairs for trapezoidal rule calculation
        values.append((alpha, avg_size, sscv))

    # Calculate P using trapezoidal rule
    I = 0
    mean_sscv = 0
    max_sscv = 0

    for i in range(len(values)):
        alpha1, size1, sscv = values[i]

        if i+1<len(values):
            alpha2, size2, _ = values[i + 1]
        else:
            break

        I += (alpha2 - alpha1) * (size1 + size2) / 2

        mean_sscv += sscv

        if sscv > max_sscv:
            max_sscv = sscv

    mean_sscv /= len(values)
    I /= (1-top1_accuracy)

    #print(I)

    return I, mean_sscv, max_sscv

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
        'SSCV_mean',
        'SSCV_worst',
        'I_value',
    ]

    I_df = pd.DataFrame(columns=columns)

    return I_df


def compute_iteration(random_seed, conf_data):
    """
    Worker function that runs a single iteration with pre-loaded config data.
    conf_data is a dict containing all necessary config values (picklable).
    """
    # Unpack config data
    alpha = conf_data['alpha']
    data_arrays = conf_data['data_arrays']  # dict of plain numpy arrays
    n_classes = conf_data['n_classes']
    distance_metric = conf_data['distance_metric']
    score_function = conf_data['score_function']
    mondrian = conf_data['mondrian']
    reg_k = conf_data['reg_k']
    reg_lambda = conf_data['reg_lambda']
    top1_accuracy = conf_data['top1_accuracy']
    top5_accuracy = conf_data['top5_accuracy']
    n_calib = conf_data['n_calib']
    conformal_domain = conf_data['conformal_domain']
    
    # Create split for this iteration
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
        1,
        top1_accuracy,
        top5_accuracy
    )


def add_row_to_I_table(n_iterations):
    # Load configuration ONCE at the start
    conf = ConformalConfig()
    
    # Convert npz object to plain numpy arrays dict for pickling
    data_arrays = {key: np.array(conf.data[key]) for key in conf.data.files}
    
    # Extract all necessary config data into picklable format
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
        joblib.delayed(compute_iteration)(random_seed, conf_data)
        for random_seed in range(n_iterations)
    )

    I_values, mean_SSCV_values, worst_SSCV_values = zip(*results)
    I_values = list(I_values)
    mean_SSCV_values = list(mean_SSCV_values)
    worst_SSCV_values = list(worst_SSCV_values)
    
    median_I_value = np.median(I_values)
    median_mean_sscv = np.median(mean_SSCV_values)
    median_worst_sscv = np.median(worst_SSCV_values)
    
    print(f'Median I value: {median_I_value}, '
          f'Median of means SSCV: {median_mean_sscv}, '
          f'Median of Max SSCV: {median_worst_sscv}')

    I_table_path = os.path.join(conf.evaluation_dir, 'figures', 'I_table.csv')
    I_table = (pd.read_csv(I_table_path) if os.path.exists(I_table_path) else create_I_table())
    
    # Remove existing row if present
    mask = (
        (I_table['model_architecture'] == conf.model_architecture) &
        (I_table['dataset'] == conf.dataset) &
        (I_table['domain'] == conf.conformal_domain) &
        (I_table['score_function'] == conf.score_function) &
        (I_table['distance_metric'] == conf.distance_metric) &
        (I_table['mondrian'] == conf.mondrian)
    )
    I_table = I_table[~mask]
    
    # Add new row
    new_row = {
        'model_architecture': conf.model_architecture,
        'dataset': conf.dataset,
        'top1_accuracy': conf.top1_accuracy,
        'top5_accuracy': conf.top5_accuracy,
        'domain': conf.conformal_domain,
        'score_function': conf.score_function,
        'distance_metric': conf.distance_metric,
        'mondrian': conf.mondrian,
        'SSCV_mean': median_mean_sscv,
        'SSCV_worst': median_worst_sscv,
        'I_value': median_I_value,
    }
    
    I_table = pd.concat([I_table, pd.DataFrame([new_row])], ignore_index=True)
    I_table.to_csv(os.path.join(conf.evaluation_dir, 'figures', 'I_table.csv'), index=False)


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

    #add_row_to_I_table(n_iterations=100)
    pass