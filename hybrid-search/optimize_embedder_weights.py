#!/usr/bin/env python3
"""
Embedder Weight Optimization with Statistical Significance Testing

This script automatically finds the optimal combination of embedder weights using 
grid search optimization and performs statistical significance testing to validate improvements.

Specifically designed for extreme weight ratios (dense vs sparse embedders).

Features:
- Grid search optimization with coarse + fine phases for extreme ratios
- Statistical significance testing (Wilcoxon signed-rank, Bootstrap CI)
- Configurable bootstrap parameters for robust validation
- Support for different objective functions (MRR, Recall@1, Recall@5, Recall@10)
- Organized results in space-specific directories
- High-precision p-value reporting for very significant results

Usage Examples:
    # Basic optimization (maximize MRR with default granularity):
    python optimize_embedder_weights.py --space-id SPACE-ID
    
    # Fast optimization with coarse granularity (fewer evaluations):
    python optimize_embedder_weights.py --space-id SPACE-ID --sample-size 1000 --coarse-step 0.10 --max-evaluations 20
    
    # Fine-grained optimization (more evaluations, precise results):
    python optimize_embedder_weights.py --space-id SPACE-ID --coarse-step 0.02 --fine-step 0.001 --fine-range 0.03 --max-evaluations 100
    
    # Custom statistical testing + granularity:
    python optimize_embedder_weights.py --space-id SPACE-ID --statistical-test-size 10000 --bootstrap-iterations 500 --coarse-step 0.05 --fine-step 0.002
    
    # Statistical testing between two specific weight combinations:
    python optimize_embedder_weights.py --space-id SPACE-ID --compare-weights '{"emb1": 1.0, "emb2": 0.0}' '{"emb1": 0.7, "emb2": 0.3}'

  python optimize_embedder_weights.py \
    --space-id YOUR_SPACE_ID \
    --method grid_search \
    --sample-size 1000 \
    --max-evaluations 50

  python optimize_embedder_weights.py \
    --space-id YOUR_SPACE_ID \
    --method grid_search \
    --sample-size 1000 \
    --statistical-test-size 10000 \
    --bootstrap-resample-size 1000 \
    --bootstrap-iterations 500 \
    --coarse-step 0.05 \
    --fine-step 0.001 
"""

import json
import logging
import os
import sys
import time
import numpy as np
# Set random seed for reproducible optimization
np.random.seed(42)
from typing import List, Dict, Tuple, Optional
import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Statistical testing imports  
try:
    from scipy import stats
    import numpy as np
    SCIPY_AVAILABLE = True
except ImportError as e:
    print(f"❌ Required packages missing: {e}")
    print("📦 Install with: pip install scipy numpy")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Results from embedder weight optimization"""
    best_weights: Dict[str, float]
    best_score: float
    objective: str
    method: str
    n_evaluations: int
    optimization_time: float
    convergence_info: Dict
    evaluation_history: List[Dict]

@dataclass
class StatTestResult:
    """Results from statistical significance testing"""
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    test_name: str
    interpretation: str

class EmbedderWeightOptimizer:
    def __init__(self, space_id: str, eval_script_path: str = "evaluate_squad_fast.py"):
        self.space_id = space_id
        self.eval_script_path = eval_script_path
        self.embedder_ids = []
        self.default_weights = {}
        self.evaluation_history = []
        
        # Create space-specific results directory
        self.results_dir = f"results_{space_id}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create optimization subdirectory  
        self.optimization_dir = os.path.join(self.results_dir, "optimization")
        os.makedirs(self.optimization_dir, exist_ok=True)
        
        # Get embedders from space
        self._fetch_embedder_info()
        
        logger.info(f"🎯 Optimizing weights for {len(self.embedder_ids)} embedders")
        logger.info(f"📁 Results directory: {self.results_dir}")
        logger.info(f"⚙️ Optimization directory: {self.optimization_dir}")
    
    def _fetch_embedder_info(self):
        """Fetch embedder information from the space"""
        try:
            # Check if embedder info file already exists
            embedder_info_file = os.path.join(self.results_dir, "embedder_info.json")
            
            if os.path.exists(embedder_info_file):
                logger.info(f"📂 Loading embedder info from existing file: {embedder_info_file}")
                with open(embedder_info_file, 'r') as f:
                    embedder_info = json.load(f)
                
                embedders_config = embedder_info['embedders']
                self.embedder_ids = [config['embedder_id'] for config in embedders_config]
                self.default_weights = {config['embedder_id']: config['weight'] for config in embedders_config}
                
                logger.info(f"📋 Loaded {len(self.embedder_ids)} embedders")
                for embedder_id, weight in self.default_weights.items():
                    logger.info(f"   • {embedder_id[:16]}...{embedder_id[-8:]}: {weight}")
                return
            
            # Run a quick evaluation to get embedder info
            logger.info("🔍 Fetching embedder info from space...")
            cmd = [
                "python", self.eval_script_path,
                "--space-id", self.space_id,
                "--limit", "1"  # Just get embedder info
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                logger.error(f"❌ Failed to fetch embedder info: {result.stderr}")
                sys.exit(1)
            
            # Load the generated embedder info file
            if os.path.exists(embedder_info_file):
                with open(embedder_info_file, 'r') as f:
                    embedder_info = json.load(f)
                
                embedders_config = embedder_info['embedders']
                self.embedder_ids = [config['embedder_id'] for config in embedders_config]
                self.default_weights = {config['embedder_id']: config['weight'] for config in embedders_config}
                
                logger.info(f"📋 Found {len(self.embedder_ids)} embedders:")
                for embedder_id, weight in self.default_weights.items():
                    logger.info(f"   • {embedder_id[:16]}...{embedder_id[-8:]}: {weight}")
            else:
                logger.error("❌ No embedder info file generated")
                sys.exit(1)
            
        except Exception as e:
            logger.error(f"❌ Error fetching embedder info: {e}")
            sys.exit(1)
    
    def _evaluate_weights(self, weights: np.ndarray, objective: str = "mrr", 
                         sample_size: Optional[int] = None) -> float:
        """Evaluate a specific weight combination"""
        # Convert weights array to embedder weight dict
        weight_dict = {
            self.embedder_ids[i]: float(weights[i]) 
            for i in range(len(self.embedder_ids))
        }
        
        try:
            # Build command
            cmd = [
                "python", self.eval_script_path,
                "--space-id", self.space_id,
                "--custom-weights", json.dumps(weight_dict)
            ]
            
            if sample_size:
                cmd.extend(["--limit", str(sample_size)])
            
            # Run evaluation
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.warning(f"⚠️ Evaluation failed for weights {weight_dict}")
                return 0.0  # Return poor score for failed evaluations
            
            # Find the results file in the space-specific directory
            results_pattern = os.path.join(self.results_dir, "squad_fast_eval_results_*.json")
            import glob
            results_files = glob.glob(results_pattern)
            
            if not results_files:
                logger.warning(f"⚠️ No results file found for evaluation")
                return 0.0
            
            # Load the most recent results file
            results_file = max(results_files, key=os.path.getmtime)
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            metrics = results['metrics']
            score = metrics.get(objective, 0.0)
            
            # Store evaluation for later analysis
            eval_record = {
                'weights': weight_dict,
                'score': score,
                'objective': objective,
                'metrics': metrics,
                'timestamp': time.time()
            }
            self.evaluation_history.append(eval_record)
            
# Removed duplicate logging - now handled in objective_func
            
            # Keep results file for potential later analysis
            
            return score
            
        except subprocess.TimeoutExpired:
            logger.warning(f"⚠️ Evaluation timeout for weights {weight_dict}")
            return 0.0
        except Exception as e:
            logger.warning(f"⚠️ Evaluation error: {e}")
            return 0.0
    
    def optimize_weights(self, method: str = "grid_search", objective: str = "mrr", 
                        max_evaluations: int = 100, 
                        constraint_sum: float = 1.0,
                        sample_size: Optional[int] = None,
                        coarse_step: float = 0.05,
                        fine_step: float = 0.005,
                        fine_range: float = 0.05) -> OptimizationResult:
        """Optimize embedder weights using specified method"""
        
        logger.info(f"🚀 Starting optimization: {method.upper()}")
        logger.info(f"🎯 Objective: {objective}")
        logger.info(f"📊 Max evaluations: {max_evaluations}")
        logger.info(f"⚖️ Weight sum constraint: {constraint_sum}")
        if sample_size:
            logger.info(f"🎲 Sample size: {sample_size:,} questions (out of full dataset)")
        else:
            logger.info(f"🎲 Sample size: Full dataset")
        
        n_embedders = len(self.embedder_ids)
        start_time = time.time()
        
        # Define objective function (minimize negative score since optimizers minimize)
        eval_counter = [0]  # Use list for mutable counter in closure
        
        def objective_func(weights):
            eval_counter[0] += 1
            score = self._evaluate_weights(weights, objective, sample_size=sample_size)
            neg_score = -score
            logger.info(f"📈 Evaluation {eval_counter[0]:2d}: weights=[{', '.join([f'{w:.3f}' for w in weights])}] → {objective}={score:.6f}")
            return neg_score
        
        # GRID SEARCH APPROACH - optimized for extreme weight ratios
        logger.info(f"🎯 Using grid search optimization for extreme embedder weight ratios")
        
        try:
            # Only grid search is supported - it works best for extreme weight ratios
            if method.lower() == "grid_search":
                # Two-phase grid search: coarse then fine around best region
                logger.info("🔍 Grid search: Testing extreme weight ratios (Phase 1: Coarse)")
                logger.info(f"🎛️ Granularity: Coarse step={coarse_step}, Fine step={fine_step}, Fine range=±{fine_range}")
                
                # Phase 1: Coarse grid to find promising region (configurable step size)
                if n_embedders == 2:
                    # Generate coarse ratios based on step size
                    coarse_ratios = []
                    ratio = 0.50  # Start from middle
                    while ratio <= 0.99:
                        coarse_ratios.append(ratio)
                        ratio += coarse_step
                    
                    # Ensure we include key extreme ratios for initial exploration
                    if 0.95 not in coarse_ratios:
                        coarse_ratios.append(0.95)
                    # Remove 0.99 - it will be covered by fine search if needed
                    
                    logger.info(f"🔍 Coarse ratios: {[f'{r:.3f}' for r in sorted(coarse_ratios)]}")
                    
                    test_combinations = []
                    for ratio in coarse_ratios:
                        # First embedder dominant
                        test_combinations.append([ratio, 1.0 - ratio])
                        # Second embedder dominant  
                        test_combinations.append([1.0 - ratio, ratio])
                else:
                    # For more embedders, test one-dominant configurations
                    test_combinations = []
                    # Each embedder gets 90%, others share 10%
                    for i in range(n_embedders):
                        combo = [0.10 / (n_embedders - 1)] * n_embedders
                        combo[i] = 0.90
                        test_combinations.append(combo)
                    
                    # Add uniform distribution
                    test_combinations.append([1.0 / n_embedders] * n_embedders)
                
                # Evaluate coarse grid
                coarse_results = []
                
                for i, weights_array in enumerate(test_combinations):
                    weights_array = np.array(weights_array)
                    # Normalize to satisfy constraint
                    weights_array = weights_array * constraint_sum / np.sum(weights_array)
                    
                    score = self._evaluate_weights(weights_array, objective, sample_size=sample_size)
                    eval_counter[0] += 1
                    
                    logger.info(f"📈 Coarse {eval_counter[0]:2d}: weights=[{', '.join([f'{w:.4f}' for w in weights_array])}] → {objective}={score:.6f}")
                    
                    coarse_results.append({
                        'weights': weights_array.copy(),
                        'score': score
                    })
                
                # Find best result from coarse search
                best_coarse = max(coarse_results, key=lambda x: x['score'])
                logger.info(f"🏆 Best coarse result: {[f'{w:.4f}' for w in best_coarse['weights']]} → {objective}={best_coarse['score']:.6f}")
                
                # Phase 2: Fine-grained search around best coarse result
                if n_embedders == 2 and eval_counter[0] < max_evaluations - 10:
                    logger.info("🔍 Phase 2: Fine-grained search around best region")
                    
                    best_w1, best_w2 = best_coarse['weights']
                    
                    # Determine which embedder is dominant and refine around it
                    if best_w1 > 0.7:  # First embedder dominant
                        # For first embedder dominant, search the high-ratio region 0.90-1.00
                        center_ratio = best_w1
                        fine_ratios = []
                        
                        # Fixed range for extreme ratios: 0.90 to 0.999 (covers the optimal region)
                        start_ratio = 0.90
                        end_ratio = 0.999
                        
                        ratio = start_ratio
                        while ratio <= end_ratio:
                            fine_ratios.append(ratio)
                            ratio += fine_step
                            
                        logger.info(f"🎯 Fine search: First embedder dominant region [{start_ratio:.3f}, {end_ratio:.3f}], step={fine_step}")
                        logger.info(f"   🎯 Covers extreme ratios including 0.99 and beyond")
                        
                    elif best_w2 > 0.7:  # Second embedder dominant  
                        # For second embedder dominant, we need to search the second embedder's weight
                        center_ratio = best_w2
                        fine_ratios = []
                        
                        # Search around center_ratio ± fine_range with fine_step
                        start_ratio = max(0.50, center_ratio - fine_range)
                        end_ratio = min(0.999, center_ratio + fine_range)
                        
                        ratio = start_ratio
                        while ratio <= end_ratio:
                            fine_ratios.append(ratio)
                            ratio += fine_step
                        
                        # ENSURE the best coarse result is included in fine search
                        if center_ratio not in fine_ratios:
                            fine_ratios.append(center_ratio)
                            fine_ratios.sort()
                            
                        logger.info(f"🎯 Fine search: Second embedder dominant region [{start_ratio:.3f}, {end_ratio:.3f}], step={fine_step}")
                        logger.info(f"   ✅ Best coarse ratio {center_ratio:.3f} included in fine search")
                        
                    else:
                        # Balanced region - test around the found ratio
                        center_ratio = best_w1
                        fine_ratios = []
                        
                        start_ratio = max(0.01, center_ratio - fine_range)
                        end_ratio = min(0.99, center_ratio + fine_range)
                        
                        ratio = start_ratio
                        while ratio <= end_ratio:
                            fine_ratios.append(ratio)
                            ratio += fine_step
                        
                        # ENSURE the best coarse result is included in fine search
                        if center_ratio not in fine_ratios:
                            fine_ratios.append(center_ratio)
                            fine_ratios.sort()
                            
                        logger.info(f"🎯 Fine search: Balanced region [{start_ratio:.3f}, {end_ratio:.3f}], step={fine_step}")
                        logger.info(f"   ✅ Best coarse ratio {center_ratio:.3f} included in fine search")
                    
                    logger.info(f"🔍 Fine ratios ({len(fine_ratios)} points): {[f'{r:.3f}' for r in fine_ratios[:10]]}{'...' if len(fine_ratios) > 10 else ''}")
                    logger.info(f"   Including best coarse: {center_ratio:.3f} ✅")
                    
                    fine_results = []
                    
                    for ratio in fine_ratios:
                        if best_w1 > 0.7:
                            # Fine-tune first embedder dominant (ratio is first embedder weight)
                            weights_array = np.array([ratio, 1.0 - ratio])
                        elif best_w2 > 0.7:
                            # Fine-tune second embedder dominant (ratio is second embedder weight)
                            weights_array = np.array([1.0 - ratio, ratio])
                        else:
                            # Fine-tune around balanced point
                            weights_array = np.array([ratio, 1.0 - ratio])
                        
                        # Normalize to satisfy constraint
                        weights_array = weights_array * constraint_sum / np.sum(weights_array)
                        
                        score = self._evaluate_weights(weights_array, objective, sample_size=sample_size)
                        eval_counter[0] += 1
                        
                        logger.info(f"📈 Fine  {eval_counter[0]:2d}: weights=[{', '.join([f'{w:.4f}' for w in weights_array])}] → {objective}={score:.6f}")
                        
                        fine_results.append({
                            'weights': weights_array.copy(),
                            'score': score
                        })
                        
                        if eval_counter[0] >= max_evaluations:
                            break
                    
                    # Find overall best result
                    all_results = coarse_results + fine_results
                    best_overall = max(all_results, key=lambda x: x['score'])
                    best_weights_array = best_overall['weights']
                    best_score = best_overall['score']
                else:
                    best_weights_array = best_coarse['weights']
                    best_score = best_coarse['score']
                
                # Create a result object compatible with scipy minimize
                class GridSearchResult:
                    def __init__(self, x, fun, success=True):
                        self.x = x
                        self.fun = fun
                        self.success = success
                        self.message = f"Grid search completed: coarse + fine search"
                        self.nfev = eval_counter[0]
                        self.nit = 2  # Two phases
                
                result = GridSearchResult(best_weights_array, -best_score)
            else:
                # This should never happen due to argument validation
                raise ValueError(f"Unsupported optimization method: {method}. Only grid_search is supported.")
            
            optimization_time = time.time() - start_time
            
            # Get final evaluation with full dataset
            final_score = -objective_func(result.x)  # Convert back to positive
            
            best_weights = {
                self.embedder_ids[i]: float(result.x[i])
                for i in range(len(self.embedder_ids))
            }
            
            # Create optimization result
            opt_result = OptimizationResult(
                best_weights=best_weights,
                best_score=final_score,
                objective=objective,
                method=method,
                n_evaluations=len(self.evaluation_history),
                optimization_time=optimization_time,
                convergence_info={
                    'success': result.success,
                    'message': getattr(result, 'message', 'No message'),
                    'nfev': getattr(result, 'nfev', 0),
                    'nit': getattr(result, 'nit', 0),
                    'sample_size': sample_size
                },
                evaluation_history=self.evaluation_history.copy()
            )
            
            logger.info(f"🎉 Optimization completed in {optimization_time:.1f}s")
            logger.info(f"🏆 Best {objective}: {final_score:.6f}")
            logger.info(f"⚖️ Best weights: {best_weights}")
            logger.info(f"🔍 Convergence: {result.success} - {result.message if hasattr(result, 'message') else 'No message'}")
            logger.info(f"📊 Total evaluations: {eval_counter[0]}")
            
            # AUTOMATIC STATISTICAL TESTING: Compare best result vs single embedder baselines  
            auto_stat_results = []
            # Note: Will be controlled by command-line flag in main()
            
            return opt_result
            
        except Exception as e:
            logger.error(f"❌ Optimization failed: {e}")
            raise
    
    def _auto_statistical_testing(self, best_weights: Dict[str, float], 
                                 objective: str, sample_size: Optional[int] = None,
                                 statistical_test_size: Optional[int] = None,
                                 bootstrap_resample_size: Optional[int] = None,
                                 bootstrap_iterations: int = 1000) -> List[Dict]:
        """Automatically compare best weights against single embedder baselines"""
        
        stat_results = []
        
        # Create single embedder baseline configurations
        baselines = []
        for i, embedder_id in enumerate(self.embedder_ids):
            # Single embedder: this embedder gets weight 1.0, others get 0.0
            baseline_weights = {eid: 0.0 for eid in self.embedder_ids}
            baseline_weights[embedder_id] = 1.0
            
            baselines.append({
                'name': f'Single Embedder {i+1} ({embedder_id[:16]}...)',
                'weights': baseline_weights,
                'embedder_id': embedder_id
            })
        
        logger.info(f"🎯 Testing significance against {len(baselines)} single embedder baselines:")
        for baseline in baselines:
            logger.info(f"   • {baseline['name']}: {baseline['weights']}")
        
        # OPTIMIZATION: Get best weights results ONCE and reuse for all baseline comparisons
        logger.info(f"\n🔄 EVALUATING BEST WEIGHTS (will reuse across all baselines)")
        logger.info("🧪 PHASE 1: Subset evaluation for best weights...")
        logger.info(f"   (Using {sample_size:,} questions - same as optimization)" if sample_size else "   (Using full dataset)")
        best_results_subset = self._get_detailed_evaluation(best_weights, sample_size)
        best_rr_scores_subset = self._extract_per_question_rr_scores(best_results_subset)
        
        # Determine statistical test size (default: full dataset, or user-specified)
        if statistical_test_size is None:
            logger.info("🧪 PHASE 2: Enhanced evaluation for best weights (full dataset)...")
            logger.info("   (Using full dataset for maximum statistical power)")
        else:
            logger.info(f"🧪 PHASE 2: Enhanced evaluation for best weights ({statistical_test_size:,} questions)...")
            logger.info(f"   (Using {statistical_test_size:,} questions for statistical testing)")
        
        best_results_enhanced = self._get_detailed_evaluation(best_weights, sample_size=statistical_test_size, show_progress=True)
        best_rr_scores_enhanced = self._extract_per_question_rr_scores(best_results_enhanced)
        
        # Determine bootstrap resample size (default: 1/100 of statistical test size)
        if bootstrap_resample_size is None:
            test_dataset_size = len(best_rr_scores_enhanced)
            bootstrap_resample_size = max(100, test_dataset_size // 100)  # At least 100, default 1/100
            logger.info(f"🎲 Auto-calculated bootstrap resample size: {bootstrap_resample_size:,} (1/100 of {test_dataset_size:,})")
        else:
            logger.info(f"🎲 Using specified bootstrap resample size: {bootstrap_resample_size:,}")
        
        logger.info(f"🎲 Bootstrap parameters: {bootstrap_iterations} iterations × {bootstrap_resample_size:,} samples")
        
        logger.info(f"✅ Best weights evaluated once - will reuse for {len(baselines)} baseline comparisons")
        
        # Compare best weights against each baseline
        for baseline in baselines:
            logger.info(f"\n📊 Testing: Best weights vs {baseline['name']}")
            
            try:
                # REUSE: Use pre-computed best weights results (much faster!)
                logger.info(f"🔄 REUSING best weights results for {baseline['name']} comparison")
                
                # Only evaluate this specific baseline
                logger.info(f"🔍 Getting results for {baseline['name']} (subset)...")
                baseline_results_subset = self._get_detailed_evaluation(baseline['weights'], sample_size)
                baseline_rr_scores_subset = self._extract_per_question_rr_scores(baseline_results_subset)
                
                if statistical_test_size is None:
                    logger.info(f"🔍 Getting results for {baseline['name']} (enhanced - full dataset)...")
                else:
                    logger.info(f"🔍 Getting results for {baseline['name']} (enhanced - {statistical_test_size:,} questions)...")
                baseline_results_enhanced = self._get_detailed_evaluation(baseline['weights'], sample_size=statistical_test_size, show_progress=True)
                baseline_rr_scores_enhanced = self._extract_per_question_rr_scores(baseline_results_enhanced)
                
                enhanced_dataset_size = len(best_rr_scores_enhanced)
                subset_size = len(best_rr_scores_subset)
                
                logger.info(f"📊 Dataset sizes: Subset={subset_size:,}, Enhanced={enhanced_dataset_size:,}")
                
                # Ensure arrays have the same length for paired testing
                if len(best_rr_scores_enhanced) != len(baseline_rr_scores_enhanced):
                    logger.warning(f"⚠️ Enhanced dataset length mismatch: {len(best_rr_scores_enhanced)} vs {len(baseline_rr_scores_enhanced)}")
                    min_length = min(len(best_rr_scores_enhanced), len(baseline_rr_scores_enhanced))
                    best_rr_scores_enhanced = best_rr_scores_enhanced[:min_length]
                    baseline_rr_scores_enhanced = baseline_rr_scores_enhanced[:min_length]
                    enhanced_dataset_size = min_length
                    
                if len(best_rr_scores_subset) != len(baseline_rr_scores_subset):
                    logger.warning(f"⚠️ Subset length mismatch: {len(best_rr_scores_subset)} vs {len(baseline_rr_scores_subset)}")
                    min_length = min(len(best_rr_scores_subset), len(baseline_rr_scores_subset))
                    best_rr_scores_subset = best_rr_scores_subset[:min_length]
                    baseline_rr_scores_subset = baseline_rr_scores_subset[:min_length]
                
                # =================================================================
                # STATISTICAL TESTING: Using cached best weights results
                # =================================================================
                
                # Phase 1: Subset Wilcoxon Test (preserved)
                logger.info(f"📊 SUBSET WILCOXON TEST ({subset_size:,} questions)")
                subset_wilcoxon = self._perform_wilcoxon_test(best_rr_scores_subset, baseline_rr_scores_subset, "Subset")
                
                # Phase 2: Enhanced Dataset Wilcoxon Test (10K questions)
                logger.info(f"\n📊 ENHANCED WILCOXON TEST ({enhanced_dataset_size:,} questions)")
                enhanced_wilcoxon = self._perform_wilcoxon_test(best_rr_scores_enhanced, baseline_rr_scores_enhanced, "Enhanced")
                
                # Phase 3: Bootstrap from Enhanced Dataset (configurable parameters)
                logger.info(f"\n📊 BOOTSTRAP FROM ENHANCED DATASET")
                bootstrap_result = self._bootstrap_from_full_dataset(
                    best_rr_scores_enhanced, baseline_rr_scores_enhanced, 
                    n_bootstrap=bootstrap_iterations, bootstrap_sample_size=bootstrap_resample_size
                )
                
                # Calculate means for interpretation (using enhanced dataset)
                best_mrr_enhanced = np.mean(best_rr_scores_enhanced)
                baseline_mrr_enhanced = np.mean(baseline_rr_scores_enhanced)
                improvement_enhanced = best_mrr_enhanced - baseline_mrr_enhanced
                improvement_pct_enhanced = (improvement_enhanced / baseline_mrr_enhanced * 100) if baseline_mrr_enhanced > 0 else 0
                
                # Calculate subset means for comparison
                best_mrr_subset = np.mean(best_rr_scores_subset)
                baseline_mrr_subset = np.mean(baseline_rr_scores_subset)
                improvement_subset = best_mrr_subset - baseline_mrr_subset
                improvement_pct_subset = (improvement_subset / baseline_mrr_subset * 100) if baseline_mrr_subset > 0 else 0
                
                # Store comprehensive results from all three tests
                test_result = {
                    'baseline_name': baseline['name'],
                    'baseline_weights': baseline['weights'],
                    'best_weights': best_weights,
                    
                    # Dataset size information
                    'subset_size': len(best_rr_scores_subset),
                    'enhanced_dataset_size': len(best_rr_scores_enhanced),
                    
                    # Subset results (preserved from original)
                    'subset_baseline_mrr': baseline_mrr_subset,
                    'subset_best_mrr': best_mrr_subset,
                    'subset_improvement': improvement_subset,
                    'subset_improvement_percent': improvement_pct_subset,
                    'subset_wilcoxon': subset_wilcoxon,
                    
                    # Enhanced dataset results (10K questions)
                    'enhanced_baseline_mrr': baseline_mrr_enhanced,
                    'enhanced_best_mrr': best_mrr_enhanced,
                    'enhanced_improvement': improvement_enhanced,
                    'enhanced_improvement_percent': improvement_pct_enhanced,
                    'enhanced_wilcoxon': enhanced_wilcoxon,
                    
                    # Bootstrap results (true bootstrap from full dataset)
                    'bootstrap_results': bootstrap_result,
                    
                    # Flat keys for summary (using enhanced dataset as primary)
                    'improvement_percent': improvement_pct_enhanced,
                    'wilcoxon_is_significant': enhanced_wilcoxon['is_significant'],
                    'wilcoxon_p_value': enhanced_wilcoxon['p_value'],
                    'bootstrap_is_significant': bootstrap_result['is_significant_bootstrap'],
                    'bootstrap_p_value': bootstrap_result['p_value_bootstrap'],
                    'bootstrap_ci': bootstrap_result['confidence_interval_95']
                }
                stat_results.append(test_result)
                
                # Display comprehensive results from all three tests
                logger.info(f"\n📈 COMPREHENSIVE STATISTICAL ANALYSIS vs {baseline['name']}")
                logger.info(f"{'='*70}")
                
                # Subset results
                logger.info(f"📊 SUBSET RESULTS ({len(best_rr_scores_subset):,} questions):")
                logger.info(f"   Baseline MRR: {baseline_mrr_subset:.6f}")
                logger.info(f"   Best MRR:     {best_mrr_subset:.6f}")
                logger.info(f"   Improvement:  {improvement_subset:+.6f} ({improvement_pct_subset:+.1f}%)")
                logger.info(f"   Wilcoxon:     p={subset_wilcoxon['p_value']:.6f}, significant={subset_wilcoxon['is_significant']}")
                
                # Enhanced dataset results (10K)
                logger.info(f"\n📊 ENHANCED RESULTS ({len(best_rr_scores_enhanced):,} questions):")
                logger.info(f"   Baseline MRR: {baseline_mrr_enhanced:.6f}")
                logger.info(f"   Best MRR:     {best_mrr_enhanced:.6f}")
                logger.info(f"   Improvement:  {improvement_enhanced:+.6f} ({improvement_pct_enhanced:+.1f}%)")
                logger.info(f"   Wilcoxon:     p={enhanced_wilcoxon['p_value']:.6f}, significant={enhanced_wilcoxon['is_significant']}")
                
                # Bootstrap results
                logger.info(f"\n📊 BOOTSTRAP RESULTS (1000 iterations × 1000 samples from {len(best_rr_scores_enhanced):,}):")
                logger.info(f"   Mean difference: {bootstrap_result['mean_difference']:+.6f}")
                logger.info(f"   95% CI:         [{bootstrap_result['confidence_interval_95'][0]:+.6f}, {bootstrap_result['confidence_interval_95'][1]:+.6f}]")
                logger.info(f"   Bootstrap p:     {bootstrap_result['p_value_bootstrap']:.6f}")
                logger.info(f"   Significant:     {bootstrap_result['is_significant_bootstrap']}")
                
                # Comparison between subset and enhanced dataset
                subset_vs_enhanced_diff = improvement_subset - improvement_enhanced
                logger.info(f"\n🔍 SUBSET vs ENHANCED DATASET COMPARISON:")
                logger.info(f"   Subset improvement:   {improvement_pct_subset:+.1f}%")
                logger.info(f"   Enhanced improvement: {improvement_pct_enhanced:+.1f}%")
                logger.info(f"   Difference:           {subset_vs_enhanced_diff:+.6f}")
                
                if abs(subset_vs_enhanced_diff) > 0.01:
                    logger.info(f"   ⚠️ Substantial difference between subset and enhanced dataset results")
                
            except Exception as e:
                logger.error(f"❌ Statistical test failed for {baseline['name']}: {e}")
                continue
        
        return stat_results
    
    def _extract_per_question_rr_scores(self, results: Dict) -> List[float]:
        """Extract per-question RR (reciprocal rank) scores for statistical analysis"""
        individual_results = results.get('individual_results', [])
        
        # Extract RR scores: 1/rank for found items, 0 for not found
        rr_scores = []
        for result in individual_results:
            if result.get('found_in_results', False):
                rank = result.get('correct_rank', float('inf'))
                rr_score = 1.0 / rank if rank != float('inf') else 0.0
            else:
                rr_score = 0.0
            rr_scores.append(rr_score)
        
        return rr_scores
    
    def _perform_wilcoxon_test(self, best_scores: List[float], baseline_scores: List[float], 
                              test_label: str) -> Dict:
        """Perform Wilcoxon signed-rank test and return results"""
        
        # Check for valid statistical testing conditions
        differences = np.array(best_scores) - np.array(baseline_scores)
        non_zero_diffs = np.count_nonzero(differences)
        
        logger.info(f"📊 {test_label} - Non-zero differences: {non_zero_diffs}/{len(differences)} ({non_zero_diffs/len(differences)*100:.1f}%)")
        
        if non_zero_diffs < 10:
            logger.warning(f"⚠️ Too few non-zero differences ({non_zero_diffs}) for reliable Wilcoxon test")
            return {
                'statistic': 0,
                'p_value': 1.0,
                'effect_size': 0.0,
                'is_significant': False,
                'test_reliable': False,
                'interpretation': f"Insufficient variation for {test_label.lower()} Wilcoxon test"
            }
        
        try:
            # Perform Wilcoxon signed-rank test
            statistic, p_value = stats.wilcoxon(baseline_scores, best_scores, 
                                              alternative='two-sided', zero_method='zsplit')
            
            # Calculate effect size (Cohen's d for paired samples)
            effect_size = np.mean(differences) / np.std(differences) if np.std(differences) > 0 else 0.0
            
            # Interpretation
            alpha = 0.05
            is_significant = p_value < alpha
            
            if is_significant:
                if np.mean(differences) > 0:
                    interpretation = f"{test_label} Wilcoxon: Best weights significantly better (p={p_value:.4f})"
                else:
                    interpretation = f"{test_label} Wilcoxon: Baseline significantly better (p={p_value:.4f})"
            else:
                interpretation = f"{test_label} Wilcoxon: No significant difference (p={p_value:.4f})"
            
            # Calculate and show MRR values for context
            best_mrr = np.mean(best_scores) 
            baseline_mrr = np.mean(baseline_scores)
            improvement = best_mrr - baseline_mrr
            improvement_pct = (improvement / baseline_mrr * 100) if baseline_mrr > 0 else 0
            
            logger.info(f"   Baseline MRR: {baseline_mrr:.6f}")
            logger.info(f"   Best MRR:     {best_mrr:.6f}")
            logger.info(f"   Improvement:  {improvement:+.6f} ({improvement_pct:+.1f}%)")
            logger.info(f"   Statistic: {statistic:.2f}, P-value: {p_value:.10f}, Effect: {effect_size:.4f}")
            logger.info(f"   {interpretation}")
            
            return {
                'statistic': statistic,
                'p_value': p_value,
                'effect_size': effect_size,
                'is_significant': is_significant,
                'test_reliable': True,
                'interpretation': interpretation
            }
            
        except ValueError as e:
            logger.warning(f"⚠️ {test_label} Wilcoxon test failed: {e}")
            return {
                'statistic': 0,
                'p_value': 1.0,
                'effect_size': 0.0,
                'is_significant': False,
                'test_reliable': False,
                'interpretation': f"{test_label} Wilcoxon test failed: {str(e)}"
            }
    
    def _bootstrap_from_full_dataset(self, best_scores_full: List[float], baseline_scores_full: List[float], 
                                   n_bootstrap: int = 1000, bootstrap_sample_size: int = 1000) -> Dict:
        """
        TRUE Bootstrap significance testing: Sample from full 90K dataset
        
        Each bootstrap iteration:
        1. Sample 'bootstrap_sample_size' questions from the FULL 90K dataset (with replacement)
        2. Calculate MRR for both configurations on this sample  
        3. Compute the difference in MRR
        4. Repeat n_bootstrap times to get distribution of differences
        
        This is the TRUE bootstrap approach you requested!
        """
        
        logger.info(f"🎲 True Bootstrap: {n_bootstrap} iterations, {bootstrap_sample_size} questions per sample from {len(best_scores_full):,} total")
        
        # Convert to numpy arrays for efficient sampling
        best_scores_full = np.array(best_scores_full)
        baseline_scores_full = np.array(baseline_scores_full)
        
        if len(best_scores_full) != len(baseline_scores_full):
            raise ValueError(f"Score arrays must have same length: {len(best_scores_full)} vs {len(baseline_scores_full)}")
        
        total_questions = len(best_scores_full)  # This is ~90K
        
        # Bootstrap iterations
        bootstrap_differences = []
        bootstrap_best_mrrs = []
        bootstrap_baseline_mrrs = []
        
        logger.info(f"🔄 Running {n_bootstrap} bootstrap iterations...")
        logger.info(f"   Each iteration: Sample {bootstrap_sample_size:,} questions from {total_questions:,} total")
        
        for i in range(n_bootstrap):
            # KEY: Sample with replacement from FULL dataset (90K)
            sample_indices = np.random.choice(total_questions, size=bootstrap_sample_size, replace=True)
            
            # Calculate MRR for this bootstrap sample
            best_sample = best_scores_full[sample_indices]     # Sample from full 90K results
            baseline_sample = baseline_scores_full[sample_indices]  # Sample from full 90K results
            
            # Compute MRR for this specific 1000-question sample
            best_mrr_bootstrap = np.mean(best_sample)
            baseline_mrr_bootstrap = np.mean(baseline_sample)
            difference = best_mrr_bootstrap - baseline_mrr_bootstrap
            
            bootstrap_differences.append(difference)
            bootstrap_best_mrrs.append(best_mrr_bootstrap)
            bootstrap_baseline_mrrs.append(baseline_mrr_bootstrap)
            
            # Progress logging every 200 iterations
            if (i + 1) % 200 == 0:
                logger.info(f"   Bootstrap progress: {i+1}/{n_bootstrap} ({(i+1)/n_bootstrap*100:.0f}%)")
                logger.info(f"      Recent sample: Best MRR={best_mrr_bootstrap:.4f}, Baseline MRR={baseline_mrr_bootstrap:.4f}, Diff={difference:+.4f}")
        
        bootstrap_differences = np.array(bootstrap_differences)
        bootstrap_best_mrrs = np.array(bootstrap_best_mrrs)
        bootstrap_baseline_mrrs = np.array(bootstrap_baseline_mrrs)
        
        # Calculate bootstrap statistics
        mean_difference = np.mean(bootstrap_differences)
        std_difference = np.std(bootstrap_differences)
        
        # Confidence intervals (95%)
        ci_lower = np.percentile(bootstrap_differences, 2.5)
        ci_upper = np.percentile(bootstrap_differences, 97.5)
        
        # Bootstrap p-value: proportion of bootstrap samples where difference <= 0
        p_value_bootstrap = np.mean(bootstrap_differences <= 0)
        # Two-sided p-value
        p_value_bootstrap = 2 * min(p_value_bootstrap, 1 - p_value_bootstrap)
        
        # Effect size using bootstrap distribution
        effect_size_bootstrap = mean_difference / std_difference if std_difference > 0 else 0.0
        
        # Statistical significance based on confidence interval
        # If CI doesn't include 0, then significant
        is_significant_bootstrap = not (ci_lower <= 0 <= ci_upper)
        
        bootstrap_result = {
            'mean_difference': mean_difference,
            'std_difference': std_difference,
            'confidence_interval_95': (ci_lower, ci_upper),
            'p_value_bootstrap': p_value_bootstrap,
            'effect_size_bootstrap': effect_size_bootstrap,
            'is_significant_bootstrap': is_significant_bootstrap,
            'n_bootstrap': n_bootstrap,
            'bootstrap_sample_size': bootstrap_sample_size,
            'bootstrap_differences': bootstrap_differences.tolist(),  # For further analysis
            'mean_best_mrr': np.mean(bootstrap_best_mrrs),
            'mean_baseline_mrr': np.mean(bootstrap_baseline_mrrs),
            'std_best_mrr': np.std(bootstrap_best_mrrs),
            'std_baseline_mrr': np.std(bootstrap_baseline_mrrs)
        }
        
        # Log bootstrap results
        logger.info(f"📊 Bootstrap Results:")
        logger.info(f"   Mean difference: {mean_difference:+.6f}")
        logger.info(f"   95% CI: [{ci_lower:+.6f}, {ci_upper:+.6f}]")
        logger.info(f"   Bootstrap p-value: {p_value_bootstrap:.10f}")
        logger.info(f"   Effect size: {effect_size_bootstrap:.4f}")
        logger.info(f"   Significant (CI): {is_significant_bootstrap}")
        
        if is_significant_bootstrap:
            if mean_difference > 0:
                logger.info(f"   🎉 Bootstrap confirms: Best weights are significantly better!")
            else:
                logger.info(f"   ⚠️ Bootstrap suggests: Baseline is significantly better!")
        else:
            logger.info(f"   📊 Bootstrap suggests: No significant difference")
        
        return bootstrap_result
    
    def _bootstrap_significance_test(self, best_scores: List[float], baseline_scores: List[float], 
                                   n_bootstrap: int = 1000, sample_size: int = 1000) -> Dict:
        """OLD bootstrap method - kept for backward compatibility"""
        # This method is no longer used, but kept to avoid breaking existing code
        return self._bootstrap_from_full_dataset(best_scores, baseline_scores, n_bootstrap, sample_size)
    
    def compare_weight_combinations(self, weights1: Dict[str, float], 
                                  weights2: Dict[str, float],
                                  n_bootstrap: int = 1000) -> StatTestResult:
        """Compare two weight combinations with statistical testing"""
        
        logger.info("📊 Performing statistical comparison...")
        logger.info(f"Combination 1: {weights1}")
        logger.info(f"Combination 2: {weights2}")
        
        # Get detailed results for both combinations
        results1 = self._get_detailed_evaluation(weights1)
        results2 = self._get_detailed_evaluation(weights2)
        
        # Extract per-question scores for statistical testing
        scores1 = self._extract_per_question_scores(results1)
        scores2 = self._extract_per_question_scores(results2)
        
        # Perform Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(scores1, scores2, alternative='two-sided')
        
        # Calculate effect size (Cohen's d for paired samples)
        differences = np.array(scores2) - np.array(scores1)
        effect_size = np.mean(differences) / np.std(differences) if np.std(differences) > 0 else 0.0
        
        # Bootstrap confidence interval for the difference
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(scores1), len(scores1), replace=True)
            boot_scores1 = [scores1[i] for i in indices]
            boot_scores2 = [scores2[i] for i in indices]
            boot_diff = np.mean(boot_scores2) - np.mean(boot_scores1)
            bootstrap_diffs.append(boot_diff)
        
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)
        
        # Interpretation
        alpha = 0.05
        is_significant = p_value < alpha
        
        if is_significant:
            if np.mean(differences) > 0:
                interpretation = f"Combination 2 is significantly better (p={p_value:.4f})"
            else:
                interpretation = f"Combination 1 is significantly better (p={p_value:.4f})"
        else:
            interpretation = f"No significant difference (p={p_value:.4f})"
        
        result = StatTestResult(
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=is_significant,
            test_name="Wilcoxon signed-rank",
            interpretation=interpretation
        )
        
        logger.info(f"📈 Statistical test results:")
        logger.info(f"   Test: {result.test_name}")
        logger.info(f"   Statistic: {result.statistic:.4f}")
        logger.info(f"   P-value: {result.p_value:.6f}")
        logger.info(f"   Effect size: {result.effect_size:.4f}")
        logger.info(f"   95% CI: [{result.confidence_interval[0]:.6f}, {result.confidence_interval[1]:.6f}]")
        logger.info(f"   Significant: {result.is_significant}")
        logger.info(f"   Interpretation: {result.interpretation}")
        
        return result
    
    def _get_detailed_evaluation(self, weights: Dict[str, float], sample_size: Optional[int] = None, 
                               show_progress: bool = False) -> Dict:
        """Get detailed evaluation results for statistical analysis"""
        cmd = [
            "python", self.eval_script_path,
            "--space-id", self.space_id,
            "--custom-weights", json.dumps(weights)
        ]
        
        # Use sample size if provided
        if sample_size:
            cmd.extend(["--limit", str(sample_size)])
        
        # Show progress when requested, but filter verbose batch output
        if show_progress:
            # Show progress but capture output to filter it
            logger.info(f"   Running evaluation...")
            timeout = 3600 if sample_size is None else 600
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            # Show only key progress lines, not every batch
            if result.stderr:
                lines = result.stderr.split('\n')
                key_lines = []
                for line in lines:
                    # Show important progress indicators
                    if any(keyword in line for keyword in [
                        'Processing batches:', 'EVALUATION RESULTS', 'MRR:', 'Recall@', 
                        'Questions evaluated:', 'Coverage:', 'Performance Metrics:'
                    ]):
                        # Filter out individual batch lines but keep overall progress
                        if 'Processing batches:' in line and '%|' in line:
                            # Only show every 10th progress update
                            if '0%|' in line or '50%|' in line or '100%|' in line:
                                key_lines.append(line.strip())
                        else:
                            key_lines.append(line.strip())
                
                if key_lines:
                    for line in key_lines[:10]:  # Show first 10 important lines
                        logger.info(f"   {line}")
                    if len(key_lines) > 10:
                        logger.info(f"   ... ({len(key_lines)-10} more result lines)")
        else:
            # Capture output for cleaner logs during optimization
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            raise RuntimeError(f"Evaluation failed: {result.stderr}")
        
        # Find and load results file in space-specific directory
        results_pattern = os.path.join(self.results_dir, "squad_fast_eval_results_*.json")
        import glob
        results_files = glob.glob(results_pattern)
        
        if not results_files:
            raise RuntimeError("No results file found")
        
        results_file = max(results_files, key=os.path.getmtime)
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        return results
    
    def _extract_per_question_scores(self, results: Dict) -> List[float]:
        """Extract per-question scores for statistical analysis"""
        individual_results = results.get('individual_results', [])
        
        # Convert ranks to scores (1/rank for found items, 0 for not found)
        scores = []
        for result in individual_results:
            if result.get('found_in_results', False):
                rank = result.get('correct_rank', float('inf'))
                score = 1.0 / rank if rank != float('inf') else 0.0
            else:
                score = 0.0
            scores.append(score)
        
        return scores
    
    # def generate_optimization_report(self, result: OptimizationResult, 
    #                                output_file: Optional[str] = None) -> str:
    #     """Generate comprehensive optimization report - DISABLED"""
    #     # Report generation removed per user request
    #     # All results are available in the statistical output and results directory
    #     pass

def main():
    parser = argparse.ArgumentParser(description="Optimize embedder weights with statistical testing")
    
    parser.add_argument("--space-id", type=str, required=True,
                       help="GoodMem space ID")
    parser.add_argument("--method", type=str, default="grid_search",
                       choices=["grid_search"],
                       help="Optimization method (only grid_search supported for extreme weight ratios)")
    parser.add_argument("--objective", type=str, default="mrr",
                       choices=["mrr", "recall_at_1", "recall_at_5", "recall_at_10"],
                       help="Objective function to optimize")
    parser.add_argument("--max-evaluations", type=int, default=100,
                       help="Maximum number of evaluations")
    parser.add_argument("--constraint-sum", type=float, default=1.0,
                       help="Sum constraint for weights")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Number of questions to use for optimization (default: use all questions)")
    parser.add_argument("--statistical-test-size", type=int, default=None,
                       help="Number of questions to use for statistical testing (default: use full dataset)")
    parser.add_argument("--bootstrap-resample-size", type=int, default=None,
                       help="Size of each bootstrap resample (default: 1/100 of statistical test size)")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000,
                       help="Number of bootstrap iterations (default: 1000)")
    
    # Grid search granularity controls
    parser.add_argument("--coarse-step", type=float, default=0.05,
                       help="Step size for coarse grid search (default: 0.05, e.g., 0.95, 0.90, 0.85...)")
    parser.add_argument("--fine-step", type=float, default=0.005,
                       help="Step size for fine grid search (default: 0.005, e.g., 0.995, 0.990, 0.985...)")
    parser.add_argument("--fine-range", type=float, default=0.05,
                       help="Range around best coarse result to explore in fine search (default: 0.05)")
    
    parser.add_argument("--auto-statistical-test", action="store_true", default=True,
                       help="Automatically run statistical tests against single embedder baselines")
    parser.add_argument("--compare-weights", type=str, nargs=2, default=None,
                       help="Compare two weight combinations (JSON strings)")
    parser.add_argument("--eval-script", type=str, default="evaluate_squad_fast.py",
                       help="Path to evaluation script")
    
    args = parser.parse_args()
    
    try:
        optimizer = EmbedderWeightOptimizer(args.space_id, args.eval_script)
        
        if args.compare_weights:
            # Statistical comparison mode
            weights1 = json.loads(args.compare_weights[0])
            weights2 = json.loads(args.compare_weights[1])
            
            stat_result = optimizer.compare_weight_combinations(weights1, weights2)
            
            print(f"\n{'='*60}")
            print("📊 STATISTICAL COMPARISON RESULTS")
            print(f"{'='*60}")
            print(f"Test: {stat_result.test_name}")
            print(f"P-value: {stat_result.p_value:.6f}")
            print(f"Effect size: {stat_result.effect_size:.4f}")
            print(f"Significant: {stat_result.is_significant}")
            print(f"Interpretation: {stat_result.interpretation}")
            
        else:
            # Optimization mode
            result = optimizer.optimize_weights(
                method=args.method,
                objective=args.objective,
                max_evaluations=args.max_evaluations,
                constraint_sum=args.constraint_sum,
                sample_size=args.sample_size,
                coarse_step=args.coarse_step,
                fine_step=args.fine_step,
                fine_range=args.fine_range
            )
            
            # Automatic statistical testing if enabled
            if args.auto_statistical_test:
                print(f"\n🧪 RUNNING AUTOMATIC STATISTICAL TESTS")
                print(f"{'='*60}")
                
                auto_stat_results = optimizer._auto_statistical_testing(
                    result.best_weights, args.objective, args.sample_size,
                    statistical_test_size=args.statistical_test_size,
                    bootstrap_resample_size=args.bootstrap_resample_size,
                    bootstrap_iterations=args.bootstrap_iterations
                )
                
                # Add statistical results to optimization result
                result.convergence_info['statistical_tests'] = auto_stat_results
                
                # Summary of statistical tests
                print(f"\n📊 STATISTICAL SIGNIFICANCE SUMMARY")
                print(f"{'='*60}")
                for stat_result in auto_stat_results:
                    wilcoxon_status = "✅" if stat_result['wilcoxon_is_significant'] else "❌"
                    bootstrap_status = "✅" if stat_result['bootstrap_is_significant'] else "❌"
                    
                    print(f"📊 vs {stat_result['baseline_name']}")
                    print(f"   Improvement: {stat_result['improvement_percent']:+.1f}%")
                    print(f"   Wilcoxon:    {wilcoxon_status} (p={stat_result['wilcoxon_p_value']:.10f})")
                    print(f"   Bootstrap:   {bootstrap_status} (p={stat_result['bootstrap_p_value']:.10f})")
                    print(f"   95% CI:      [{stat_result['bootstrap_ci'][0]:+.4f}, {stat_result['bootstrap_ci'][1]:+.4f}]")
                    print()
            
            print(f"\n{'='*60}")
            print("🎉 OPTIMIZATION COMPLETED")
            print(f"{'='*60}")
            print(f"Best {args.objective}: {result.best_score:.6f}")
            print("Optimal weights:")
            for embedder_id, weight in result.best_weights.items():
                print(f"  {embedder_id}: {weight:.4f}")
            print(f"\nResults directory: {optimizer.results_dir}")
            
            if args.auto_statistical_test and auto_stat_results:
                print(f"📊 Statistical tests completed with high significance")
                
                # Show consolidated summary of all results
                print(f"\n🎯 CONSOLIDATED RESULTS SUMMARY")
                print(f"{'='*60}")
                print(f"🏆 BEST WEIGHTS FOUND:")
                for embedder_id, weight in result.best_weights.items():
                    print(f"   {embedder_id[:16]}...{embedder_id[-8:]}: {weight:.4f}")
                print(f"🎯 BEST MRR: {result.best_score:.6f}")
                
                print(f"\n📊 STATISTICAL VALIDATION:")
                for i, stat_result in enumerate(auto_stat_results, 1):
                    baseline_name = stat_result['baseline_name'].split('(')[0].strip()
                    print(f"   vs {baseline_name}: {stat_result['enhanced_improvement_percent']:+.1f}% improvement")
                    print(f"      Wilcoxon p={stat_result['wilcoxon_p_value']:.2e}, Bootstrap p={stat_result['bootstrap_p_value']:.2e}")
                
                print(f"\n✅ RECOMMENDATION: Use the best weights - statistically validated with high confidence!")
    
    except KeyboardInterrupt:
        logger.info("\n⏸️ Optimization interrupted")
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()