import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import f_oneway
from typing import Dict, List, Tuple, Optional, Union, Any


class MultiClassMatcher:
    """
    Multi-class matching without replacement across K groups.
    
    Supports 1:1:1 matching (balanced sets) and flexible ratio matching.
    Uses greedy nearest-neighbor matching anchored on the smallest group.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing all subjects.
    group_col : str
        Column name indicating group/class membership.
    covariates : List[str]
        List of column names for matching covariates (must be numeric).
    group_order : List, optional
        Order of groups for matching. If None, uses smallest → largest.
        The first group serves as the anchor for matching.
    
    Attributes
    ----------
    df : pd.DataFrame
        Copy of input dataframe.
    group_col : str
        Name of group column.
    covariates : List[str]
        List of covariate names.
    group_order : List
        Ordered list of groups.
    group_sizes : Dict
        Original sample sizes per group.
    group_dfs : Dict[str, pd.DataFrame]
        Subset dataframes per group.
    
    Examples
    --------
    >>> matcher = MultiClassMatcher(
    ...     df=data,
    ...     group_col='treatment_group',
    ...     covariates=['age', 'bmi', 'blood_pressure'],
    ...     group_order=['control', 'treatment_A', 'treatment_B']
    ... )
    >>> matched_df, diagnostics = matcher.match_greedy_1to1to1()
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        group_col: str, 
        covariates: List[str], 
        group_order: Optional[List] = None
    ):
        # Store inputs
        self.df = df.copy()
        self.group_col = group_col
        self.covariates = covariates
        
        # Validate inputs
        self._validate_inputs()
        
        # Get group counts
        group_counts = df[group_col].value_counts()
        
        if len(group_counts) < 2:
            raise ValueError(
                f"Need at least 2 groups for matching; found {len(group_counts)}. "
                f"Groups present: {group_counts.index.tolist()}"
            )
        
        # Set group order (smallest → largest by default for optimal matching)
        if group_order is None:
            self.group_order = group_counts.sort_values().index.tolist()
        else:
            self._validate_group_order(group_order, df[group_col].unique())
            self.group_order = list(group_order)
        
        # Store group sizes and subset dataframes
        self.group_sizes = {grp: group_counts.get(grp, 0) for grp in self.group_order}
        self.group_dfs = {
            grp: df[df[group_col] == grp].copy() 
            for grp in self.group_order
        }
    
    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        # Check group column exists
        if self.group_col not in self.df.columns:
            raise KeyError(f"Group column '{self.group_col}' not found in dataframe")
        
        # Check covariates exist and are numeric
        for cov in self.covariates:
            if cov not in self.df.columns:
                raise KeyError(f"Covariate '{cov}' not found in dataframe")
            if not pd.api.types.is_numeric_dtype(self.df[cov]):
                raise TypeError(
                    f"Covariate '{cov}' must be numeric. "
                    f"Current dtype: {self.df[cov].dtype}"
                )
        
        # Check for missing values in covariates
        missing = self.df[self.covariates].isnull().sum()
        if missing.any():
            missing_covs = missing[missing > 0].to_dict()
            raise ValueError(
                f"Covariates contain missing values: {missing_covs}. "
                "Please handle missing values before matching."
            )
    
    def _validate_group_order(
        self, 
        group_order: List, 
        actual_groups: np.ndarray
    ) -> None:
        """Validate user-specified group order."""
        missing = set(group_order) - set(actual_groups)
        if missing:
            raise ValueError(
                f"Group order contains groups not present in data: {missing}. "
                f"Available groups: {set(actual_groups)}"
            )
        
        extra = set(actual_groups) - set(group_order)
        if extra:
            raise ValueError(
                f"Group order is missing groups present in data: {extra}. "
                f"Specify all groups: {set(actual_groups)}"
            )
    
    def _standardize(self) -> Dict[Any, np.ndarray]:
        """
        Standardize covariates using RobustScaler on the FULL sample.
        
        Returns
        -------
        Dict[Any, np.ndarray]
            Dictionary mapping group → scaled covariate matrix.
        """
        scaler = RobustScaler()
        scaler.fit(self.df[self.covariates])
        
        scaled_data = {}
        for grp in self.group_order:
            if len(self.group_dfs[grp]) > 0:
                scaled_data[grp] = scaler.transform(self.group_dfs[grp][self.covariates])
            else:
                scaled_data[grp] = np.empty((0, len(self.covariates)))
        
        return scaled_data
    
    def _compute_retention_rates(
        self, 
        matched_df: pd.DataFrame
    ) -> Dict[Any, float]:
        """
        Compute proportion of each group retained after matching.
        
        Parameters
        ----------
        matched_df : pd.DataFrame
            Matched subjects dataframe.
        
        Returns
        -------
        Dict[Any, float]
            Dictionary mapping group → retention rate (0.0 to 1.0).
        """
        n_per_group = matched_df[self.group_col].value_counts().to_dict()
        return {
            grp: n_per_group.get(grp, 0) / self.group_sizes[grp]
            for grp in self.group_order
        }
    
    def match_greedy_1to1to1(
        self, 
        caliper: Optional[float] = None, 
        metric: str = 'euclidean'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Greedy 1:1:...:1 matching across K groups without replacement.
        
        Forms matched sets with exactly one subject per group.
        Sequentially matches each anchor subject to nearest neighbors
        in other groups.
        
        Parameters
        ----------
        caliper : float, optional
            Maximum distance threshold on scaled covariates.
            Subjects with distance > caliper are not matched.
            None means no constraint.
        metric : str, default='euclidean'
            Distance metric for nearest neighbor search.
            Options: 'euclidean', 'manhattan', 'minkowski', etc.
        
        Returns
        -------
        matched_df : pd.DataFrame
            Matched subjects with '_match_set_id' column identifying sets.
        diagnostics : Dict
            Matching diagnostics including:
            - method : str
            - caliper : float or None
            - total_matched_sets : int
            - n_per_group : Dict
            - original_sizes : Dict
            - retention_rates : Dict
        
        Raises
        ------
        RuntimeError
            If no matched sets can be formed.
        
        Examples
        --------
        >>> matched_df, diag = matcher.match_greedy_1to1to1(caliper=0.5)
        """
        scaled_data = self._standardize()
        
        # Track available indices per group using boolean masks
        available = {
            grp: np.ones(len(self.group_dfs[grp]), dtype=bool)
            for grp in self.group_order
        }
        
        matched_sets = []
        set_id = 0
        
        # Anchor on first group (smallest by default)
        anchor_group = self.group_order[0]
        anchor_scaled = scaled_data[anchor_group]
        
        # Process each anchor subject
        for anchor_pos in range(len(anchor_scaled)):
            if not available[anchor_group][anchor_pos]:
                continue
            
            # Initialize match with anchor
            matches = {
                anchor_group: (
                    self.group_dfs[anchor_group].index[anchor_pos], 
                    anchor_pos
                )
            }
            valid_match = True
            distances = {}
            
            # Find matches in each other group
            for grp in self.group_order[1:]:
                avail_mask = available[grp]
                if not np.any(avail_mask):
                    valid_match = False
                    break
                
                # Get available scaled vectors
                ctrl_scaled = scaled_data[grp][avail_mask]
                ctrl_indices = np.where(avail_mask)[0]
                
                # Find nearest neighbor
                nbrs = NearestNeighbors(
                    n_neighbors=1, 
                    metric=metric, 
                    algorithm='ball_tree'
                )
                nbrs.fit(ctrl_scaled)
                dist, idx_pos = nbrs.kneighbors(
                    anchor_scaled[anchor_pos:anchor_pos+1]
                )
                dist = dist[0][0]
                match_pos_in_avail = idx_pos[0][0]
                match_abs_pos = ctrl_indices[match_pos_in_avail]
                
                # Apply caliper if specified
                if caliper is not None and dist > caliper:
                    valid_match = False
                    break
                
                matches[grp] = (
                    self.group_dfs[grp].index[match_abs_pos], 
                    match_abs_pos
                )
                distances[grp] = dist
            
            # Commit match if valid
            if valid_match:
                # Mark matched subjects as unavailable
                for grp, (_, abs_pos) in matches.items():
                    available[grp][abs_pos] = False
                
                # Store matched set
                set_record = {grp: idx for grp, (idx, _) in matches.items()}
                set_record['_match_set_id'] = set_id
                matched_sets.append(set_record)
                set_id += 1
        
        if not matched_sets:
            raise RuntimeError(
                "No matched sets formed. This may be due to:\n"
                "  1. Insufficient overlap between groups\n"
                "  2. Caliper too restrictive (try larger caliper or None)\n"
                "  3. Very different group sizes\n"
                "Check covariate distributions and consider relaxing constraints."
            )
        
        # Build matched dataframe
        matched_indices = []
        set_ids = []
        for match_set in matched_sets:
            sid = match_set['_match_set_id']
            for grp in self.group_order:
                matched_indices.append(match_set[grp])
                set_ids.append(sid)
        
        matched_df = self.df.loc[matched_indices].copy()
        matched_df['_match_set_id'] = set_ids
        
        # Compute diagnostics
        n_per_group = matched_df[self.group_col].value_counts().to_dict()
        retention_rates = self._compute_retention_rates(matched_df)
        
        diagnostics = {
            'method': 'greedy_1to1to1',
            'caliper': caliper,
            'metric': metric,
            'total_matched_sets': set_id,
            'n_per_group': {grp: n_per_group.get(grp, 0) for grp in self.group_order},
            'original_sizes': self.group_sizes,
            'retention_rates': retention_rates
        }
        
        return matched_df, diagnostics
    
    def match_stratified_ratio(
        self, 
        target_ratios: Optional[Dict[Any, float]] = None, 
        caliper: Optional[float] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Match with flexible ratios (e.g., 1:2:3 across groups).
        
        Uses sequential matching anchored on the first group in group_order.
        Each anchor subject is matched to k subjects from each other group,
        where k is determined by the ratio.
        
        Parameters
        ----------
        target_ratios : Dict[Any, float], optional
            Target ratio of subjects per group relative to anchor.
            Example: {'control': 1.0, 'treatment_A': 1.5, 'treatment_B': 2.0}
            Means: 1 control : 1.5 treatment_A : 2 treatment_B (rounded).
            Default is 1:1:...:1.
        caliper : float, optional
            Maximum distance threshold on scaled covariates.
            None means no constraint.
        
        Returns
        -------
        matched_df : pd.DataFrame
            Matched subjects with '_match_set_id' column.
        diagnostics : Dict
            Matching diagnostics including retention rates and balance.
        
        Raises
        ------
        RuntimeError
            If no matches can be formed with specified ratios/caliper.
        
        Examples
        --------
        >>> ratios = {'control': 1, 'treatment': 2}
        >>> matched_df, diag = matcher.match_stratified_ratio(
        ...     target_ratios=ratios, 
        ...     caliper=0.3
        ... )
        """
        # Default to 1:1:...:1 ratios
        if target_ratios is None:
            target_ratios = {grp: 1.0 for grp in self.group_order}
        else:
            # Validate ratios cover all groups
            missing = set(self.group_order) - set(target_ratios.keys())
            if missing:
                raise ValueError(
                    f"target_ratios missing groups: {missing}. "
                    f"Must specify ratio for all groups: {self.group_order}"
                )
        
        scaled_data = self._standardize()
        
        # Track availability with boolean masks
        available = {
            grp: np.ones(len(self.group_dfs[grp]), dtype=bool)
            for grp in self.group_order
        }
        
        matched_records = []
        set_id = 0
        
        # Anchor on first group
        anchor_group = self.group_order[0]
        anchor_scaled = scaled_data[anchor_group]
        
        for anchor_pos in range(len(anchor_scaled)):
            if not available[anchor_group][anchor_pos]:
                continue
            
            # Determine required matches per group (round to integer)
            n_needed = {
                grp: max(1, int(round(target_ratios[grp]))) 
                for grp in self.group_order[1:]
            }
            
            # Find matches for each group
            matches_found = {
                anchor_group: [
                    (self.group_dfs[anchor_group].index[anchor_pos], anchor_pos)
                ]
            }
            valid_set = True
            
            for grp in self.group_order[1:]:
                avail_mask = available[grp]
                n_available = np.sum(avail_mask)
                
                if n_available < n_needed[grp]:
                    valid_set = False
                    break
                
                # Get available vectors
                ctrl_scaled = scaled_data[grp][avail_mask]
                ctrl_indices = np.where(avail_mask)[0]
                
                # Find k nearest neighbors
                k_neighbors = min(n_needed[grp], len(ctrl_scaled))
                nbrs = NearestNeighbors(
                    n_neighbors=k_neighbors,
                    metric='euclidean',
                    algorithm='ball_tree'
                )
                nbrs.fit(ctrl_scaled)
                distances, idx_pos = nbrs.kneighbors(
                    anchor_scaled[anchor_pos:anchor_pos+1]
                )
                
                # Apply caliper filter
                valid_positions = []
                for d, pos in zip(distances[0], idx_pos[0]):
                    if caliper is None or d <= caliper:
                        valid_positions.append(pos)
                        if len(valid_positions) >= n_needed[grp]:
                            break
                
                if len(valid_positions) < n_needed[grp]:
                    valid_set = False
                    break
                
                # Record absolute positions
                matches_found[grp] = [
                    (self.group_dfs[grp].index[ctrl_indices[p]], ctrl_indices[p])
                    for p in valid_positions[:n_needed[grp]]
                ]
            
            # Commit matches
            if valid_set:
                # Add anchor
                anchor_idx, anchor_abs_pos = matches_found[anchor_group][0]
                matched_records.append((anchor_idx, anchor_group, set_id))
                available[anchor_group][anchor_abs_pos] = False
                
                # Add matches from other groups
                for grp in self.group_order[1:]:
                    for match_idx, abs_pos in matches_found[grp]:
                        matched_records.append((match_idx, grp, set_id))
                        available[grp][abs_pos] = False
                
                set_id += 1
        
        if not matched_records:
            raise RuntimeError(
                "No matches formed with specified ratios/caliper. "
                "Consider:\n"
                "  1. Relaxing or removing caliper\n"
                "  2. Reducing target ratios\n"
                "  3. Checking covariate overlap between groups"
            )
        
        # Build dataframe
        indices, groups, set_ids = zip(*matched_records)
        matched_df = self.df.loc[list(indices)].copy()
        matched_df['_match_set_id'] = list(set_ids)
        
        # Diagnostics
        n_per_group = matched_df[self.group_col].value_counts().to_dict()
        retention_rates = self._compute_retention_rates(matched_df)
        
        diagnostics = {
            'method': 'stratified_ratio',
            'target_ratios': target_ratios,
            'caliper': caliper,
            'total_matched_sets': set_id,
            'n_per_group': {grp: n_per_group.get(grp, 0) for grp in self.group_order},
            'original_sizes': self.group_sizes,
            'retention_rates': retention_rates
        }
        
        return matched_df, diagnostics
    
    def assess_balance(
        self, 
        matched_df: pd.DataFrame, 
        alpha: float = 0.05
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Assess covariate balance after matching.
        
        Computes ANOVA p-values and pairwise standardized mean differences
        before and after matching.
        
        Parameters
        ----------
        matched_df : pd.DataFrame
            Matched subjects dataframe.
        alpha : float, default=0.05
            Significance level for balance assessment.
        
        Returns
        -------
        balance_df : pd.DataFrame
            Per-covariate balance metrics:
            - covariate : str
            - anova_pval_pre : float
            - anova_pval_post : float
            - max_smd_pre : float (maximum pairwise SMD before matching)
            - max_smd_post : float (maximum pairwise SMD after matching)
            - balanced_post : bool (p > alpha AND max_smd < 0.1)
        summary : Dict
            Summary statistics:
            - mean_anova_pval_post : float
            - mean_max_smd_post : float
            - pct_balanced : float (percentage of covariates balanced)
            - n_matched_total : int
            - n_per_group : Dict
        """
        results = []
        
        for cov in self.covariates:
            # Pre-matching ANOVA
            groups_pre = [
                self.group_dfs[grp][cov].values 
                for grp in self.group_order 
                if len(self.group_dfs[grp]) > 0
            ]
            
            if len(groups_pre) > 1:
                anova_pre = f_oneway(*groups_pre)
                pval_pre = (
                    anova_pre.pvalue 
                    if hasattr(anova_pre, 'pvalue') and not np.isnan(anova_pre.pvalue) 
                    else 1.0
                )
            else:
                pval_pre = 1.0
            
            # Post-matching ANOVA
            groups_post = [
                matched_df[matched_df[self.group_col] == grp][cov].values
                for grp in self.group_order
                if (matched_df[self.group_col] == grp).sum() > 0
            ]
            
            if len(groups_post) > 1:
                anova_post = f_oneway(*groups_post)
                pval_post = (
                    anova_post.pvalue 
                    if hasattr(anova_post, 'pvalue') and not np.isnan(anova_post.pvalue) 
                    else 1.0
                )
            else:
                pval_post = 1.0
            
            # Compute pairwise SMDs
            smds_pre, smds_post = [], []
            
            for i, grp1 in enumerate(self.group_order):
                for grp2 in self.group_order[i+1:]:
                    # Pre-matching SMD
                    m1 = self.group_dfs[grp1][cov].mean()
                    m2 = self.group_dfs[grp2][cov].mean()
                    s1 = self.group_dfs[grp1][cov].std()
                    s2 = self.group_dfs[grp2][cov].std()
                    pooled = np.sqrt((s1**2 + s2**2) / 2)
                    smd_pre = abs(m1 - m2) / pooled if pooled > 0 else 0
                    smds_pre.append(smd_pre)
                    
                    # Post-matching SMD
                    df1 = matched_df[matched_df[self.group_col] == grp1][cov]
                    df2 = matched_df[matched_df[self.group_col] == grp2][cov]
                    
                    if len(df1) > 1 and len(df2) > 1:
                        m1p = df1.mean()
                        m2p = df2.mean()
                        s1p = df1.std()
                        s2p = df2.std()
                        pooledp = np.sqrt((s1p**2 + s2p**2) / 2)
                        smd_post = abs(m1p - m2p) / pooledp if pooledp > 0 else 0
                        smds_post.append(smd_post)
            
            max_smd_pre = max(smds_pre) if smds_pre else np.nan
            max_smd_post = max(smds_post) if smds_post else np.nan
            
            # Determine if balanced (common threshold: SMD < 0.1)
            balanced = (pval_post > alpha) and (max_smd_post < 0.1)
            
            results.append({
                'covariate': cov,
                'anova_pval_pre': pval_pre,
                'anova_pval_post': pval_post,
                'max_smd_pre': max_smd_pre,
                'max_smd_post': max_smd_post,
                'balanced_post': balanced
            })
        
        balance_df = pd.DataFrame(results)
        
        summary = {
            'mean_anova_pval_post': balance_df['anova_pval_post'].mean(),
            'mean_max_smd_post': balance_df['max_smd_post'].mean(),
            'pct_balanced': balance_df['balanced_post'].mean() * 100,
            'n_matched_total': len(matched_df),
            'n_per_group': matched_df[self.group_col].value_counts().to_dict()
        }
        
        return balance_df, summary


def match_groups(
    df: pd.DataFrame,
    group_col: str,
    covariates: List[str],
    method: str = '1to1to1',
    group_order: Optional[List] = None,
    caliper: Optional[float] = None,
    target_ratios: Optional[Dict[Any, float]] = None,
    assess_balance: bool = True,
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Match subjects across groups without duplicates.
    
    A convenience wrapper around MultiClassMatcher for common use cases.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing all subjects.
    group_col : str
        Column name indicating group membership.
    covariates : List[str]
        List of covariate column names for matching.
    method : str, default='1to1to1'
        Matching method:
        - '1to1to1' : Balanced matching (1 subject per group per set)
        - 'ratio' : Flexible ratio matching
    group_order : List, optional
        Order of groups. First group is anchor (matched first).
        If None, uses smallest → largest.
    caliper : float, optional
        Maximum distance threshold. None = no constraint.
    target_ratios : Dict, optional
        For method='ratio', target ratios per group.
        Example: {'A': 1.0, 'B': 2.0, 'C': 3.0}
    assess_balance : bool, default=True
        Whether to compute balance diagnostics.
    **kwargs
        Additional arguments passed to matching method.
    
    Returns
    -------
    matched_df : pd.DataFrame
        Matched subjects with '_match_set_id' column.
    diagnostics : Dict
        Matching diagnostics and balance metrics.
    
    Raises
    ------
    ValueError
        If invalid method or missing required parameters.
    
    Examples
    --------
    Basic 1:1:1 matching:
    
    >>> matched_df, diag = match_groups(
    ...     df=data,
    ...     group_col='treatment',
    ...     covariates=['age', 'sex', 'bmi'],
    ...     method='1to1to1'
    ... )
    
    Ratio matching with caliper:
    
    >>> matched_df, diag = match_groups(
    ...     df=data,
    ...     group_col='treatment',
    ...     covariates=['age', 'sex', 'bmi'],
    ...     method='ratio',
    ...     target_ratios={'control': 1, 'treatment': 2},
    ...     caliper=0.3
    ... )
    
    Custom group order (anchor on specific group):
    
    >>> matched_df, diag = match_groups(
    ...     df=data,
    ...     group_col='severity',
    ...     covariates=['age', 'baseline_score'],
    ...     group_order=['severe', 'moderate', 'mild'],  # severe is anchor
    ...     method='1to1to1'
    ... )
    """
    # Initialize matcher
    matcher = MultiClassMatcher(
        df=df,
        group_col=group_col,
        covariates=covariates,
        group_order=group_order
    )
    
    # Execute matching
    if method == '1to1to1':
        matched_df, diag = matcher.match_greedy_1to1to1(
            caliper=caliper,
            metric=kwargs.get('metric', 'euclidean')
        )
    elif method == 'ratio':
        matched_df, diag = matcher.match_stratified_ratio(
            target_ratios=target_ratios,
            caliper=caliper
        )
    else:
        raise ValueError(
            f"Unknown method: '{method}'. "
            f"Supported methods: '1to1to1', 'ratio'"
        )
    
    # Add balance diagnostics
    if assess_balance:
        balance_df, summary = matcher.assess_balance(matched_df)
        diag['balance_summary'] = summary
        diag['balance_details'] = balance_df
    
    return matched_df, diag


# =============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON USE CASES
# =============================================================================

def match_two_groups(
    df: pd.DataFrame,
    group_col: str,
    covariates: List[str],
    anchor_group: Optional[str] = None,
    ratio: int = 1,
    caliper: Optional[float] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Match two groups (case-control style matching).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    group_col : str
        Group column name.
    covariates : List[str]
        Matching covariates.
    anchor_group : str, optional
        Group to use as anchor (cases). If None, uses smaller group.
    ratio : int, default=1
        Number of controls per case (for 1:r matching).
    caliper : float, optional
        Distance threshold.
    
    Returns
    -------
    matched_df : pd.DataFrame
    diagnostics : Dict
    
    Examples
    --------
    1:2 case-control matching:
    
    >>> matched_df, diag = match_two_groups(
    ...     df=data,
    ...     group_col='case_control',
    ...     covariates=['age', 'sex'],
    ...     ratio=2
    ... )
    """
    groups = df[group_col].unique()
    if len(groups) != 2:
        raise ValueError(
            f"match_two_groups requires exactly 2 groups, found {len(groups)}: {list(groups)}"
        )
    
    if anchor_group is not None and anchor_group not in groups:
        raise ValueError(f"anchor_group '{anchor_group}' not found in data")
    
    # Determine group order
    if anchor_group is None:
        group_order = None  # Will use smallest first
    else:
        other = [g for g in groups if g != anchor_group][0]
        group_order = [anchor_group, other]
    
    # Build ratios
    if anchor_group:
        target_ratios = {anchor_group: 1.0, other: float(ratio)}
    else:
        # Will be determined by smallest group as anchor
        target_ratios = None
    
    return match_groups(
        df=df,
        group_col=group_col,
        covariates=covariates,
        method='ratio' if ratio > 1 else '1to1to1',
        group_order=group_order,
        caliper=caliper,
        target_ratios=target_ratios
    )


def match_multiple_groups(
    df: pd.DataFrame,
    group_col: str,
    covariates: List[str],
    group_order: Optional[List] = None,
    caliper: Optional[float] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Match K ≥ 2 groups with 1:1:...:1 ratio.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    group_col : str
        Group column name.
    covariates : List[str]
        Matching covariates.
    group_order : List, optional
        Custom group order. First group is anchor.
    caliper : float, optional
        Distance threshold.
    verbose : bool, default=True
        Print progress information.
    
    Returns
    -------
    matched_df : pd.DataFrame
    diagnostics : Dict
    
    Examples
    --------
    Match 3 treatment groups:
    
    >>> matched_df, diag = match_multiple_groups(
    ...     df=data,
    ...     group_col='treatment_arm',
    ...     covariates=['age', 'sex', 'baseline_score'],
    ...     group_order=['placebo', 'low_dose', 'high_dose']
    ... )
    """
    if verbose:
        groups = df[group_col].value_counts()
        print(f"Original group sizes:")
        for grp, n in groups.items():
            print(f"  {grp}: {n}")
    
    matched_df, diag = match_groups(
        df=df,
        group_col=group_col,
        covariates=covariates,
        method='1to1to1',
        group_order=group_order,
        caliper=caliper
    )
    
    if verbose:
        print(f"\nMatched sets: {diag['total_matched_sets']}")
        print(f"Matched per group: {diag['n_per_group']}")
        print(f"Retention rates: {diag['retention_rates']}")
        
        if 'balance_summary' in diag:
            print(f"\nBalance:")
            print(f"  Mean max SMD: {diag['balance_summary']['mean_max_smd_post']:.3f}")
            print(f"  % balanced: {diag['balance_summary']['pct_balanced']:.1f}%")
    
    return matched_df, diag
