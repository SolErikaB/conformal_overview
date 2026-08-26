import os
import yaml
import numpy as np
import pandas as pd
import prepare_models
from tqdm import *
from model_eval_and_save_features import FeatureExtractorWrapper
import torch


with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

class DistanceMetric:
    """Class for distance metric computations."""
    
    def __init__(self, distance_metric):
        self.distance_metric = distance_metric
        
        if distance_metric == 'euclidean':
            self._distance_function = self.euclidean_distance
        elif distance_metric == 'cosine':
            self._distance_function = self.cosine_distance
        else:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

    def euclidean_distance(self, A, B, **kwargs):
        """Compute euclidean distance."""
        return np.sqrt(np.sum((A - B) ** 2))

    def cosine_distance(self, A, B, **kwargs):
        """Compute cosine distance."""
        dot_product = np.dot(A, B)
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        return 1 - (dot_product / (norm_A * norm_B))
    
    def compute_distance(self, A, B, **statistics):
        """Compute distance using the specified distance function."""
        return self._distance_function(A, B, **statistics)


class LabelDistanceScore(DistanceMetric):
    """Nonconformity score using label distance."""
    
    def __init__(self, distance_metric, n_classes):
        super().__init__(distance_metric)
        self.n_classes = n_classes
    
    def _label_to_one_hot(self, label):
        """Convert label to one-hot encoding."""
        one_hot = np.zeros(self.n_classes)
        one_hot[label] = 1
        return one_hot
    
    def compute_score(self, datapoint, label):
        """Compute nonconformity score using label distance."""
        one_hot_label = self._label_to_one_hot(label)
        distance = self.compute_distance(datapoint, one_hot_label)
        return distance


class MarginDistanceScore(DistanceMetric):
    """Nonconformity score using margin distance."""

    def __init__(self, distance_metric, n_classes):
        super().__init__(distance_metric)
        self.n_classes = n_classes
        self.distance_metric = distance_metric

    def compute_score(self, datapoint, label):
        
        current_pred = np.argmax(datapoint)

        if self.distance_metric == 'euclidean':
            if current_pred == label:
                distances = [abs(datapoint[label] - datapoint[i])
                            for i in range(len(datapoint)) if i != label]
                distance = -min(distances)
            else:
                distance = datapoint[current_pred] - datapoint[label]

            return distance

        elif self.distance_metric == 'cosine':
            # Find the largest element not equal to the label

            z = np.argmax(datapoint)
            if z == label:
                z = np.argsort(datapoint)[-2]  # second largest if largest is label

            pi_y = datapoint[label]
            pi_z = datapoint[z]

            a = (pi_y + pi_z) / 2

            b = datapoint.copy()
            b[label] = a
            b[z] = a

            distance = 1-np.linalg.norm(b)/np.linalg.norm(datapoint)

            if current_pred == label:
                return -abs(distance)
            else:
                return abs(distance)


class MeanDistanceScore(DistanceMetric):
    """Nonconformity score using distance to class means."""
    
    def __init__(self, data, labels, preds, distance_metric, n_classes):
        """
        Initialize mean-based score.
        
        Args:
            distance_metric: 'euclidean' or 'cosine'
            data: The dataset features
            labels: The dataset labels
            preds: The dataset predictions
            n_classes: Number of classes
        """
        super().__init__(distance_metric)
        self.data = data
        self.labels = labels
        self.preds = preds
        self.n_classes = n_classes
        self.class_means = []
        self.class_sizes = []

        for i in range(self.n_classes):
            class_data = self.data[self.labels == i]
            
            # Optional: keep only examples where the class is predicted correctly
            # class_data = class_data[self.preds[self.labels == i] == i]
            
            self.class_sizes.append(len(class_data))
            self.class_means.append(np.mean(class_data, axis=0))
    
    def compute_score(self, datapoint, label, exclude_datapoint=False):
        """Compute distance to class mean, optionally excluding the datapoint.
        
        Args:
            datapoint: The data point to score
            label: The class label
            exclude_datapoint: If True, compute mean without this datapoint
        
        Returns:
            distance: The distance to the class mean
        """
        if exclude_datapoint:
            # Remove datapoint from mean efficiently
            current_mean = self.class_means[label]
            n = self.class_sizes[label]
            if n <= 1:
                raise ValueError("Cannot exclude datapoint from a class with only 1 sample")
            adjusted_mean = (current_mean * n - datapoint) / (n - 1)
            distance = self.compute_distance(datapoint, adjusted_mean)
        else:
            distance = self.compute_distance(datapoint, self.class_means[label])
    
        return distance

class KMeans3DistanceScore(DistanceMetric):

    """Nonconformity score using distance to class means."""

    def __init__(self, data, labels, preds, distance_metric, n_classes):
        """
        Initialize mean-based score.

        Args:
            distance_metric: 'euclidean' or 'cosine'
            data: The dataset features
            labels: The dataset labels
            preds: The dataset predictions
            n_classes: Number of classes
        """
        super().__init__(distance_metric)
        self.data = data
        self.labels = labels
        self.preds = preds
        self.n_classes = n_classes
        self.class_data = [data[labels == i] for i in range(n_classes)]

        self.KMeans = [KMeans(n_clusters=3, random_state=0).fit(self.class_data[i]) for i in range(n_classes)]
        
    def compute_score(self, datapoint, label, exclude_datapoint=False, idx=None):
        
        if exclude_datapoint and idx is not None:
            data_excluded = np.delete(self.data, idx, axis=0)
            labels_excluded = np.delete(self.labels, idx, axis=0)
            class_data_excluded = data_excluded[labels_excluded == label]
            kmeans = KMeans(n_clusters=3, random_state=0).fit(class_data_excluded)
            distances = [self.compute_distance(datapoint, center) for center in kmeans.cluster_centers_]
            return min(distances)
        else:
            class_data_excluded = self.class_data[label]
            distances = [self.compute_distance(datapoint, center) for center in self.KMeans[label].cluster_centers_]
            return min(distances)

class RAPS(DistanceMetric):
    """Nonconformity score using Regularized Adaptive Prediction Sets."""
    
    def __init__(self, distance_metric, n_classes, reg_k, reg_lambda):
        super().__init__(distance_metric)
        self.n_classes = n_classes
        self.reg_k = reg_k
        self.reg_lambda = reg_lambda
        
    def compute_score(self, datapoint, label):

        target = datapoint[label]

        mask = datapoint > target

        r = np.sum(mask) + 1

        E = np.sum(datapoint[mask]) if r > 1 else 0

        u = np.random.uniform(0, 1)
        #u = 0.001
        score = E + u * datapoint[label] + self.reg_lambda * max(r-self.reg_k,0)

#        if r<=2:
#            print(E, datapoint[label], r, max(r-self.reg_k,0), score)
        return score


class SAPS(DistanceMetric):
    """Nonconformity score using Regularized Adaptive Prediction Sets."""
    
    def __init__(self, distance_metric, n_classes, reg_lambda):
        super().__init__(distance_metric)
        self.n_classes = n_classes
        self.reg_lambda = reg_lambda

    def compute_score(self, datapoint, label):
        
        target = datapoint[label]

        r = np.sum(datapoint > target) + 1
        biggest_score = np.max(datapoint)

        u = np.random.uniform(0, 1)
        #u = 0.001

        if r == 1:
            E = u * biggest_score
        else:
            E = biggest_score

        score = E + self.reg_lambda * (r - 2 + u)

        return score





class GradientDistanceScore(DistanceMetric):
    """Nonconformity score using gradient-based distance on feature vectors z."""
    
    def __init__(self, distance_metric, n_classes):
        super().__init__(distance_metric)

        self.n_classes = n_classes

        dataset = config['conformal_prediction']['dataset']
        model_architecture = config['conformal_prediction']['model_architecture']
        data_dir = config['training']['data_directory']
        model_dir = config['training']['model_directory']
        self.device = torch.device('cpu')

        print("before getting datasets")
        _, _, _, num_classes, input_size = prepare_models.get_datasets(dataset, data_dir, seed=123)
        print("after getting datasets")

        # Load model
        model = prepare_models.get_model(model_architecture, dataset, num_classes, input_size)
        model_path = os.path.join(model_dir, dataset, f"{model_architecture}.pth")
        if dataset != 'imagenet':
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])

        wrapped_model = FeatureExtractorWrapper(model, model_architecture)
        self.model = wrapped_model.to(self.device)
        self.model.eval()

        self.classifier = wrapped_model.classifier
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()

    def compute_score(self, datapoint, label):
        """Compute nonconformity score using gradient-based feature distance."""
        
        y_target = torch.tensor([label], dtype=torch.long).to(self.device)
        z_pred = torch.tensor(datapoint, dtype=torch.float32).to(self.device)

        if z_pred.dim() == 1:
            z_pred = z_pred.unsqueeze(0)

        # Find features that would give perfect prediction
        z_perfect = self._find_perfect_prediction(z_pred, y_target)

        # Distance from predicted features to "perfect" features
        distance = self.compute_distance(z_pred.cpu().numpy(), z_perfect.cpu().numpy())

        return distance

    def _find_perfect_prediction(self, z_pred, y_target, lr=0.1, steps=100, tolerance=1e-3):
        """Move z toward features that produce perfect probability for y_target."""
        
        z = z_pred.detach().clone()
        z.requires_grad_(True)
        
        optimizer = torch.optim.Adam([z], lr=lr)
        
        target_probs = torch.zeros(1, self.n_classes).to(self.device)
        target_probs[0, y_target] = 1.0
        
        for _ in range(steps):
            optimizer.zero_grad()
            
            logits = self.classifier(z)
            probs = torch.softmax(logits, dim=1)
            
            # MSE loss to match target probabilities
            loss = torch.nn.functional.mse_loss(probs, target_probs)
            
            loss.backward()
            optimizer.step()

            # Check convergence
            with torch.no_grad():
                if loss.item() < tolerance:
                    break
        
        return z.detach()

    

class FastGradientDistanceScore(DistanceMetric):
    """Nonconformity score using gradient-based distance on feature vectors z."""
    
    def __init__(self, distance_metric, n_classes):
        super().__init__(distance_metric)

        self.n_classes = n_classes

        dataset = config['conformal_prediction']['dataset']
        model_architecture = config['conformal_prediction']['model_architecture']
        data_dir = config['training']['data_directory']
        model_dir = config['training']['model_directory']
        self.device = torch.device('cpu')
        _, _, _, num_classes, input_size = prepare_models.get_datasets(dataset, data_dir, seed=123)

        # Load model
        model = prepare_models.get_model(model_architecture, dataset, num_classes, input_size)
        model_path = os.path.join(model_dir, dataset, f"{model_architecture}.pth")
        if dataset != 'imagenet':
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])

        wrapped_model = FeatureExtractorWrapper(model, model_architecture)
        self.model = wrapped_model.to(self.device)
        self.model.eval()

        self.classifier = wrapped_model.classifier
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()

    def compute_score(self, datapoint, label):
        """Compute nonconformity score using gradient-based feature distance."""
        
        y_target = torch.tensor([label], dtype=torch.long).to(self.device)
        z_pred = torch.tensor(datapoint, dtype=torch.float32).to(self.device)

        if z_pred.dim() == 1:
            z_pred = z_pred.unsqueeze(0)

        logit = self.classifier(z_pred)
        output = torch.nn.functional.softmax(logit, dim=1)

        grad = self._compute_gradient(z_pred, y_target)
        grad = grad.norm(p=2, dim=1, keepdim=True)
        grad = grad.detach().cpu().numpy()
        grad = grad[0][0]

        distance1 = self.compute_distance(output.detach().cpu().numpy(), self._label_to_one_hot(label))
        grad_distance = distance1 / grad
        
        return grad_distance

    def _compute_gradient(self, z_pred, y_target):
        """Compute gradient of the target class logit with respect to features."""
        
        self.classifier.eval()
        
        z = z_pred.detach().clone().to(self.device)
        z.requires_grad_(True)
        
        # Forward pass
        logits = self.classifier(z)
        probs = torch.softmax(logits, dim=1)
        target_prob = probs[0, y_target]
        grad = torch.autograd.grad(target_prob, z)[0]
        
        return grad.detach()


    def _label_to_one_hot(self, label):
        """Convert label to one-hot encoding."""
        one_hot = np.zeros(self.n_classes)
        one_hot[label] = 1
        return one_hot



class NonconformityScore:

    def __init__(self,
                 alpha,
                 calibration_data,
                 calibration_labels,
                 test_data,
                 test_labels,
                 calibration_preds,
                 n_classes,
                 distance_metric,
                 score_function,
                 mondrian,
                 n_clusters=None,
                 reg_k=2,
                 reg_lambda=3):
        """
        Initialize nonconformity score.
        """
        self.alpha = alpha
        self.calibration_data = calibration_data
        self.calibration_labels = calibration_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.calibration_preds = calibration_preds
        self.n_classes = n_classes
        self.distance_metric = distance_metric
        self.mondrian = mondrian
        self.n_clusters = n_clusters
        
        if score_function == 'label':
            self.nonconformity_score = LabelDistanceScore(distance_metric, n_classes=n_classes)

        if score_function == 'margin':
            self.nonconformity_score = MarginDistanceScore(distance_metric, n_classes=n_classes)

        if score_function == 'mean':
            self.nonconformity_score = MeanDistanceScore(calibration_data, calibration_labels, calibration_preds, distance_metric, n_classes=n_classes)
            
        if score_function == 'kmeans3':
            self.nonconformity_score = KMeans3DistanceScore(calibration_data, calibration_labels, calibration_preds, distance_metric, n_classes=n_classes)

        if score_function == 'knn':
            self.nonconformity_score = KNearestNeighborScore(distance_metric, n_classes=n_classes, k=5)
        
        if score_function == 'aps':
            reg_k = 0
            reg_lambda = 0
            score_function = 'raps'

        if score_function == 'raps':
            self.nonconformity_score = RAPS(distance_metric, n_classes=n_classes, reg_k=reg_k, reg_lambda=reg_lambda)

        if score_function == 'saps':
            self.nonconformity_score = SAPS(distance_metric, n_classes=n_classes, reg_lambda=reg_lambda)

        if score_function == 'gradient':
            self.nonconformity_score = GradientDistanceScore(distance_metric, n_classes=n_classes)

        if score_function == 'fast_gradient':
            self.nonconformity_score = FastGradientDistanceScore(distance_metric, n_classes=n_classes)

        self.score_function = score_function

        
class ConformalPrediction(NonconformityScore):
    """Conformal prediction with flexible nonconformity methods."""

    def __init__(self,
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
                 reg_k=2,
                 reg_lambda=3):
        super().__init__(
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
        
        self.alpha = alpha
        self.reg_k = reg_k
        self.reg_lambda = reg_lambda

        self.calibration_df = None
        self.results_df = None
        self.thresholds = []

        # Cached, alpha-independent artifacts
        self._calib_distances_by_class = None   # dict: class -> np.array of calib distances
        self._test_scores = None                # (n_test, n_classes) matrix
        self._scores_computed = False

    def compute_scores(self):
        """
        Compute all nonconformity scores once. This is the expensive step and
        does NOT depend on alpha, so it should only be called once per
        calibration/test split, not once per alpha.
        """
        # --- Calibration scores ---
        distances = []
        for i in range(len(self.calibration_data)):
            if i%1000==0:
                print("calibrating", i)
            data_point = self.calibration_data[i]
            if self.score_function in ['mean', 'kmeans3']:
                d = self.nonconformity_score.compute_score(
                    data_point, self.calibration_labels[i], exclude_datapoint=True
                )
            else:
                d = self.nonconformity_score.compute_score(data_point, self.calibration_labels[i])
            distances.append(d)

        self.calibration_df = pd.DataFrame({
            'label': self.calibration_labels,
            'distance': distances
        })

        # Pre-group calibration distances by class so per-alpha threshold
        # lookups don't repeatedly filter/groupby the DataFrame.
        self._calib_distances_by_class = {
            c: self.calibration_df.loc[self.calibration_df['label'] == c, 'distance'].to_numpy()
            for c in range(self.n_classes)
        }

        # --- Test scores: n_test x n_classes matrix ---
        n_test = len(self.test_data)
        test_scores = np.empty((n_test, self.n_classes))
        for i in range(n_test):
            if i%1000==0:
                print("testing", i, "/", n_test)
            data_point = self.test_data[i]
            for c in range(self.n_classes):
                test_scores[i, c] = self.nonconformity_score.compute_score(data_point, c)

        self._test_scores = test_scores
        self._scores_computed = True

    def _thresholds_for_alpha(self, alpha):
        """Cheap: just a quantile lookup over cached distances."""
        if not self.mondrian:
            n = len(self.calibration_df)
            modified_alpha = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
            threshold = np.quantile(self.calibration_df['distance'], modified_alpha)
            return [threshold] * self.n_classes

        thresholds = []
        for c in range(self.n_classes):
            class_distances = self._calib_distances_by_class[c]
            n_labels = len(class_distances)
            if n_labels == 0:
                thresholds.append(np.inf)  # no calib examples for this class -> never excluded
                continue
            modified_alpha = min(np.ceil((1 - alpha) * (n_labels + 1)) / n_labels, 1.0)
            thresholds.append(np.quantile(class_distances, modified_alpha))
        return thresholds

    def calibrate(self, alpha=None):
        """
        Backward-compatible entry point. Computes scores if needed (once),
        then sets self.thresholds for the given alpha (defaults to self.alpha).
        """
        if not self._scores_computed:
            self.compute_scores()
        alpha = self.alpha if alpha is None else alpha
        self.thresholds = self._thresholds_for_alpha(alpha)

    def predict(self, alpha=None):
        """
        Run conformal prediction on test data for a given alpha.
        Reuses cached scores — only recomputes thresholds + set membership.
        """
        if not self._scores_computed:
            self.compute_scores()

        alpha = self.alpha if alpha is None else alpha
        self.thresholds = self._thresholds_for_alpha(alpha)

        thresholds_arr = np.array(self.thresholds)          # (n_classes,)
        pred_mask = self._test_scores <= thresholds_arr[None, :]   # (n_test, n_classes) bool

        prediction_regions = [np.flatnonzero(row).tolist() for row in pred_mask]

        self.results_df = pd.DataFrame({
            'label': self.test_labels,
            'prediction_region': prediction_regions
        })
        return self.results_df
