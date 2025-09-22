import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


# +-----------------------------------------------------------------------------
## AUC


def estimate_sample_size_for_auc(
    target_auc=0.8,
    margin_of_error=0.05,
    prevalence=None,
    n_bootstraps=500,
    n_jobs=-1,
    max_sample_size=10000,
    min_sample_size=100,
    tolerance=0.01,
    max_iterations=20,
    ci_width=0.95,
    min_power=0.8,
    max_type1_error=0.05,
    random_seed=None
):
    """
    Estimates sample size for binary classification AUC with user-defined:
    - Target AUC
    - Margin of error
    - Minimum power
    - Maximum Type I error
    
    Parameters:
    -----------
    random_seed : int or None
        Seed for all random operations (data generation, train-test split, etc.)
        If None, no seed is set (non-reproducible random behavior)
    
    Returns: Dict or DataFrame with sample size and performance metrics.
    """
    # Set global random state if seed is provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    def calculate_min_sample_size(prevalence, min_per_class=10):
        """Calculate minimum sample size to ensure adequate samples per class"""
        min_for_pos = min_per_class / prevalence
        min_for_neg = min_per_class / (1 - prevalence)
        return int(np.ceil(max(min_for_pos, min_for_neg)))
    
    def simulate_auc(n_samples, pos_prevalence, iteration):
        """Generate data where theoretical AUC = target_auc with error handling"""
        try:
            # Use iteration number to get different seeds for each simulation
            current_seed = random_seed + iteration if random_seed is not None else None
            
            n_pos = int(n_samples * pos_prevalence)
            n_neg = n_samples - n_pos
            
            # Check minimum requirements for stratified split
            if n_pos < 2 or n_neg < 2:
                return np.nan
            
            # Set random state for this simulation
            rng = np.random.RandomState(current_seed)
            
            delta = np.sqrt(2) * norm.ppf(target_auc)
            neg_samples = rng.normal(loc=0, scale=1, size=n_neg)
            pos_samples = rng.normal(loc=delta, scale=1, size=n_pos)
            
            X = np.concatenate([neg_samples, pos_samples]).reshape(-1, 1)
            y = np.array([0]*n_neg + [1]*n_pos)
            
            # Additional check before train_test_split
            if len(np.unique(y)) < 2:
                return np.nan
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=current_seed
            )
            
            # Check if both classes are present in train and test sets
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                return np.nan
            
            model = LogisticRegression(random_state=current_seed, max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:, 1]
            return roc_auc_score(y_test, y_pred)
            
        except Exception:
            # Return NaN for any error (including stratification errors)
            return np.nan

    def bootstrap_auc(n_samples, pos_prevalence):
        """Bootstrap AUC with CI and error rates"""
        # Ensure minimum sample size
        min_required = calculate_min_sample_size(pos_prevalence)
        if n_samples < min_required:
            n_samples = min_required
        
        aucs = Parallel(n_jobs=n_jobs)(
            delayed(simulate_auc)(n_samples, pos_prevalence, i) 
            for i in range(n_bootstraps)
        )
        
        # Filter out NaN values (failed simulations)
        valid_aucs = [auc for auc in aucs if not np.isnan(auc)]
        
        # Check if we have enough valid simulations
        if len(valid_aucs) < n_bootstraps * 0.5:  # At least 50% success rate
            return None
        
        aucs = np.array(valid_aucs)
        
        # Handle edge case where AUC + margin >= 1.0
        effective_upper_bound = min(target_auc + margin_of_error, 1.0)
        
        alpha = (1 - ci_width) / 2
        ci_lower = np.percentile(aucs, 100 * alpha)
        ci_upper = np.percentile(aucs, 100 * (1 - alpha))
        
        return {
            'sample_size': n_samples,
            'mean_auc': np.mean(aucs),
            'auc_ci_lower': ci_lower,
            'auc_ci_upper': ci_upper,
            'power': np.mean(aucs >= (target_auc - margin_of_error)),
            'type1_error': np.mean(aucs >= effective_upper_bound),
            'prevalence': pos_prevalence,
            'valid_simulations': len(valid_aucs),
            'success_rate': len(valid_aucs) / n_bootstraps
        }

    def find_sample_size(pos_prevalence):
        """Binary search for sample size meeting ALL criteria"""
        # Calculate minimum sample size for this prevalence
        min_required = calculate_min_sample_size(pos_prevalence)
        effective_min = max(min_sample_size, min_required)
        
        low = effective_min
        high = max_sample_size
        best_sample = None
        
        for iteration in range(max_iterations):
            mid = (low + high) // 2
            result = bootstrap_auc(mid, pos_prevalence)
            
            # If bootstrap failed, try larger sample size
            if result is None:
                low = mid + 50
                continue
            
            # Check ALL conditions
            conditions_met = (
                (abs(result['mean_auc'] - target_auc) <= tolerance) and
                (result['power'] >= min_power) and
                (result['type1_error'] <= max_type1_error) and
                (result['success_rate'] >= 0.8)  # At least 80% successful simulations
            )
            
            if conditions_met:
                best_sample = result
                high = mid  # Try smaller sample
            else:
                # More precise direction selection
                if result['mean_auc'] < target_auc - tolerance:
                    low = mid + 1
                elif result['power'] < min_power:
                    low = mid + 1
                elif result['success_rate'] < 0.8:
                    low = mid + 1
                else:  # Type I error too high or AUC too high
                    high = mid - 1
                    
            if high - low <= 20:  # More lenient stopping threshold
                break
                
        # Return best sample found, or try with final parameters
        if best_sample is None:
            final_result = bootstrap_auc(high, pos_prevalence)
            if final_result is not None and all([
                abs(final_result['mean_auc'] - target_auc) <= tolerance * 2,  # More lenient
                final_result['power'] >= min_power,
                final_result['type1_error'] <= max_type1_error,
                final_result['success_rate'] >= 0.7  # More lenient
            ]):
                return final_result
            return None
            
        return best_sample

    # Input validation
    if margin_of_error <= 0:
        raise ValueError("margin_of_error must be positive")
    if target_auc + margin_of_error > 1.0:
        warnings.warn(f"target_auc + margin_of_error = {target_auc + margin_of_error} > 1.0. Type I error may be underestimated")
    
    # Handle prevalence cases
    if prevalence is not None:
        if prevalence <= 0 or prevalence >= 1:
            raise ValueError("Prevalence must be between 0 and 1")
        
        # Check if this prevalence is feasible with max_sample_size
        min_required = calculate_min_sample_size(prevalence)
        if min_required > max_sample_size:
            raise ValueError(f"Prevalence {prevalence} requires minimum sample size {min_required}, "
                           f"but max_sample_size is {max_sample_size}")
        
        return find_sample_size(prevalence)
    else:
        # Filter prevalences to only include feasible ones
        all_prevalences = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        feasible_prevalences = []
        
        for prev in all_prevalences:
            min_required = calculate_min_sample_size(prev)
            if min_required <= max_sample_size:
                feasible_prevalences.append(prev)
        
        if not feasible_prevalences:
            raise ValueError("No feasible prevalences with current max_sample_size")
        
        results = []
        for prev in tqdm(feasible_prevalences, desc="Testing prevalences"):
            res = find_sample_size(prev)
            if res is not None:  # Only append valid results
                results.append(res)
        
        return pd.DataFrame(results) if results else None
    

# +-----------------------------------------------------------------------------
## Other metrics

def estimate_sample_size_auto(
    metric='sensitivity',
    target_value=0.8,
    margin_of_error=0.05,
    prevalence=0.3,
    random_seed=42,
    n_bootstraps=500,
    min_class_size=35
):
    """
    Automated sample size estimation for classification metrics.
    
    Parameters:
    -----------
    metric : str
        'sensitivity', 'specificity', or 'accuracy'
    target_value : float
        Target metric value (0-1)
    margin_of_error : float
        Acceptable margin of error
    prevalence : float
        Class prevalence (proportion of positive cases)
    random_seed : int
        Random seed for reproducibility
    n_bootstraps : int
        Number of bootstrap iterations
    min_class_size : int
        Minimum cases per class
    
    Returns:
    --------
    dict : Results containing metric, value, CI, power, type1_error, sample sizes
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Helper functions
    def calculate_min_n(prev, min_class):
        min_n_pos = min_class / prev
        min_n_neg = min_class / (1 - prev)
        return int(np.ceil(max(min_n_pos, min_n_neg)))
    
    def validate_n(n, prev, min_class):
        n_pos = int(n * prev)
        n_neg = n - n_pos
        if n_pos < min_class or n_neg < min_class:
            return calculate_min_n(prev, min_class)
        return n
    
    def calc_metric(y_true, y_pred, metric_type):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        if metric_type == 'sensitivity':
            return tp / (tp + fn) if (tp + fn) > 0 else np.nan
        elif metric_type == 'specificity':
            return tn / (tn + fp) if (tn + fp) > 0 else np.nan
        elif metric_type == 'accuracy':
            return (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
        return np.nan
    
    def find_optimal_threshold(X_train, y_train, X_test, y_test, target, metric_type):
        model = LogisticRegression(max_iter=1000, random_state=random_seed)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        best_threshold = 0.5
        best_diff = float('inf')
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_pred_proba >= threshold).astype(int)
            metric_val = calc_metric(y_test, y_pred, metric_type)
            
            if not np.isnan(metric_val):
                diff = abs(metric_val - target)
                if diff < best_diff:
                    best_diff = diff
                    best_threshold = threshold
        
        return best_threshold
    
    # Step 1: Find optimal effect size and threshold
    def test_effect_threshold_combo(effect_size, optimize_thresh=True):
        results = []
        thresholds = []
        
        for i in range(50):  # Quick test with 50 iterations
            rng = np.random.RandomState(random_seed + i)
            
            n_test = 1000
            n_pos = int(n_test * prevalence)
            n_neg = n_test - n_pos
            
            neg_samples = rng.normal(loc=0, scale=1, size=n_neg)
            pos_samples = rng.normal(loc=effect_size, scale=1, size=n_pos)
            
            X = np.concatenate([neg_samples, pos_samples]).reshape(-1, 1)
            y = np.array([0]*n_neg + [1]*n_pos)
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, stratify=y, random_state=random_seed + i
                )
                
                if optimize_thresh:
                    threshold = find_optimal_threshold(X_train, y_train, X_test, y_test, target_value, metric)
                else:
                    threshold = 0.5
                
                model = LogisticRegression(max_iter=1000, random_state=random_seed + i)
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = (y_pred_proba >= threshold).astype(int)
                
                metric_val = calc_metric(y_test, y_pred, metric)
                
                if not np.isnan(metric_val):
                    results.append(metric_val)
                    thresholds.append(threshold)
                    
            except:
                continue
        
        if len(results) > 0:
            mean_metric = np.mean(results)
            mean_threshold = np.mean(thresholds)
            return mean_metric, mean_threshold
        return None, None
    
    # Find best effect size and threshold combination
    best_effect_size = 1.5
    best_threshold = 0.5
    best_diff = float('inf')
    
    for effect_size in np.arange(0.5, 4.0, 0.5):
        # Test with threshold optimization
        mean_metric, mean_threshold = test_effect_threshold_combo(effect_size, True)
        
        if mean_metric is not None:
            diff = abs(mean_metric - target_value)
            if diff < best_diff:
                best_diff = diff
                best_effect_size = effect_size
                best_threshold = mean_threshold
    
    # Step 2: Sample size estimation with optimal parameters
    def simulate_metric_final(n_samples, iteration):
        n_samples = validate_n(n_samples, prevalence, min_class_size)
        current_seed = random_seed + iteration if random_seed is not None else None
        
        n_pos = int(n_samples * prevalence)
        n_neg = n_samples - n_pos
        
        if n_pos < min_class_size or n_neg < min_class_size:
            return np.nan
        
        rng = np.random.RandomState(current_seed)
        
        neg_samples = rng.normal(loc=0, scale=1, size=n_neg)
        pos_samples = rng.normal(loc=best_effect_size, scale=1, size=n_pos)
        
        X = np.concatenate([neg_samples, pos_samples]).reshape(-1, 1)
        y = np.array([0]*n_neg + [1]*n_pos)
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=current_seed
            )
            
            # Check test set requirements
            if metric == 'sensitivity' and np.sum(y_test) == 0:
                return np.nan
            elif metric == 'specificity' and np.sum(y_test == 0) == 0:
                return np.nan
            
            model = LogisticRegression(random_state=current_seed, max_iter=1000)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= best_threshold).astype(int)
            
            return calc_metric(y_test, y_pred, metric)
            
        except Exception:
            return np.nan
    
    def bootstrap_final(n_samples):
        n_samples = validate_n(n_samples, prevalence, min_class_size)
        
        metric_values = Parallel(n_jobs=-1)(
            delayed(simulate_metric_final)(n_samples, i) 
            for i in range(n_bootstraps)
        )
        
        metric_values = np.array([m for m in metric_values if not np.isnan(m)])
        
        if len(metric_values) < n_bootstraps * 0.8:
            return None
        
        mean_metric = np.mean(metric_values)
        ci_lower = np.percentile(metric_values, 2.5)
        ci_upper = np.percentile(metric_values, 97.5)
        
        final_n_pos = int(n_samples * prevalence)
        final_n_neg = n_samples - final_n_pos
        
        observed_target = mean_metric
        power = np.mean(metric_values >= (observed_target - margin_of_error))
        effective_upper_bound = min(observed_target + margin_of_error, 1.0)
        type1_error = np.mean(metric_values >= effective_upper_bound)
        
        return {
            'metric': metric,
            'value': mean_metric,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'power': power,
            'type1_error': type1_error,
            'total_n': n_samples,
            'positive_class_n': final_n_pos,
            'negative_class_n': final_n_neg
        }
    
    # Step 3: Find optimal sample size
    min_n = calculate_min_n(prevalence, min_class_size)
    
    # Binary search for sample size
    low = min_n
    high = 20000
    best_result = None
    
    for iteration in range(15):  # Max 15 iterations
        mid = (low + high) // 2
        mid = validate_n(mid, prevalence, min_class_size)
        
        result = bootstrap_final(mid)
        
        if result is None:
            low = mid + 1
            continue
        
        meets_criteria = (result['power'] >= 0.8 and result['type1_error'] <= 0.05)
        
        if meets_criteria:
            best_result = result
            high = mid - 1
        else:
            low = mid + 1
            
        if high <= low:
            break
    
    # If no solution found, return result with minimum sample size
    if best_result is None:
        best_result = bootstrap_final(min_n)
    
    return best_result

# Example usage:
if __name__ == "__main__":
    # Test sensitivity
    result_sens = estimate_sample_size_auto(
        metric='sensitivity',
        target_value=0.85,
        margin_of_error=0.05,
        prevalence=0.3,
        random_seed=42,
        n_bootstraps=500
    )
    
    print("Sensitivity Results:")
    for key, value in result_sens.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "="*40)
    
    # Test specificity
    result_spec = estimate_sample_size_auto(
        metric='specificity',
        target_value=0.6,
        margin_of_error=0.1,
        prevalence=0.3,
        random_seed=42,
        n_bootstraps=500
    )
    
    print("Specificity Results:")
    for key, value in result_spec.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "="*40)
    
    # Test accuracy
    result_acc = estimate_sample_size_auto(
        metric='accuracy',
        target_value=0.8,
        margin_of_error=0.05,
        prevalence=0.5,
        random_seed=42,
        n_bootstraps=500
    )
    
    print("Accuracy Results:")
    for key, value in result_acc.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")