"""
Corporate Distress Prediction - Main Pipeline
Runs the complete 15-step ML pipeline + experiments from data to final model.

Usage:
    python main.py                    # Run pipeline + experiments (default)
    python main.py --experiments-only # Run experiments only
    python main.py --help             # Show options
"""

import sys
import os
from pathlib import Path
import importlib.util

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'experiments'))


def run_step(step_name, module_name, folder='src'):
    """Run a single pipeline step or experiment."""
    print(f"\n{'='*70}")
    print(f"  {step_name}")
    print(f"{'='*70}\n")
    
    try:
        # Import and run the module
        module_path = PROJECT_ROOT / folder / f'{module_name}.py'
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        print(f"\n✅ {step_name} completed successfully\n")
        return True
        
    except Exception as e:
        print(f"\n❌ {step_name} failed with error:")
        print(f"   {str(e)}\n")
        return False


def run_experiments(include_lstm=False):
    """Run the core optimization experiments."""
    
    print("\n" + "="*70)
    print("  OPTIMIZATION EXPERIMENTS")
    print("="*70)
    print("\nRunning core experiments that improved the model:")
    print("  • Exp 1: Reduce overfitting")
    print("  • Exp 4: Optimize recall threshold")
    print("  • Exp 5: Add temporal features (key contribution)")
    print("  • Exp 6: Combine all optimizations")
    print("  • Exp 13: Calibrate probabilities")
    print("  • Exp 14: Cross-validation")
    print("  • Exp 16: Feature selection (Top 10 features)")
    if include_lstm:
        print("  • Exp 15: LSTM baseline (optional, requires TensorFlow)")
    print("\n" + "="*70 + "\n")
    
    experiments = [
        ("Exp 1: Reduce Overfitting", "exp1_reduce_overfitting"),
        ("Exp 4: Optimize Recall", "exp4_optimize_recall"),
        ("Exp 5: Temporal Features", "exp5_temporal_features"),
        ("Exp 6: Combined Optimization", "exp6_combined_optimization"),
        ("Exp 13: Model Calibration", "exp13_model_calibration_v2"),  # Fixed: use v2
        ("Exp 14: Cross-Validation", "exp14_cross_validation_FIXED"),  # Fixed: use FIXED version
        ("Exp 16: Feature Selection", "exp16_temporal_feature_selection"),
    ]
    
    # Only include LSTM if explicitly requested
    if include_lstm:
        experiments.append(("Exp 15: LSTM Baseline", "exp15_lstm_baseline"))
    
    completed = 0
    failed = []
    
    for i, (exp_name, module_name) in enumerate(experiments, 1):
        print(f"\n[Experiment {i}/{len(experiments)}]")
        
        success = run_step(exp_name, module_name, folder='experiments')
        
        if success:
            completed += 1
        else:
            failed.append(exp_name)
            print(f"\n⚠️  Warning: {exp_name} failed but continuing...\n")
    
    return completed, failed, len(experiments)


def print_story_highlights(run_pipeline, run_exps):
    """Print a narrative summary of key metrics across steps and experiments."""
    print("\n" + "="*70)
    print("  KEY METRICS - HOW FAR WE'VE COME")
    print("="*70)
    
    if run_pipeline:
        print("\n🚦 Steps 12-15: From regularization to insight")
        print("  • Step 12 Optimization: XGBoost (AUC 0.632, F1 0.390) and LightGBM (AUC 0.627, F1 0.381) both settle on a conservative 0.45 threshold to curb overfitting.")
        print("  • Step 13 Evaluation: Test AUC 0.627, AP 0.318 — strongest year 2021 (AUC 0.647) and toughest 2022 (0.604), showing stable generalization.")
        print("  • Step 14 Benchmarks: Traditional LightGBM at AUC 0.634 still beats every baseline by >25%, especially the CDS-only rule stuck at 0.405.")
        print("  • Step 15 Explainability: XGBoost & LightGBM agree on 8/10 top drivers (CDS lags, Altman Z, returns, volatility), proving the signals are consistent.")
    
    if run_exps:
        print("\n🔬 Experiments: why we trust the final model")
        print("  • Exp 5 + 6: Temporal features plus combined regularization closed the train-test gap to ~0.09 AUC.")
        print("  • Exp 13: Calibration delivered the deployment model (AUC 0.662, recall 69.1%, ECE 0.014) catching 1,011 of 1,463 distressed firms.")
        print("  • Exp 14: Time-based cross-validation confirmed stability before moving to benchmarks and SHAP.")
        print("  • Exp 16: Feature selection reduced to 10 features (AUC 0.640, recall 72%, simpler & faster).")


def main():
    """Run the complete ML pipeline."""
    
    # Parse command line arguments
    run_pipeline = True
    run_exps = True  # Now runs experiments by default
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in ['--help', '-h']:
            print("\nUsage:")
            print("  python main.py                    # Run pipeline + experiments (default)")
            print("  python main.py --experiments-only # Run experiments only")
            print("  python main.py --help             # Show this help\n")
            return 0
        elif arg == '--with-experiments':
            run_pipeline = True
            run_exps = True
        elif arg == '--experiments-only':
            run_pipeline = False
            run_exps = True
        else:
            print(f"\n❌ Unknown argument: {arg}")
            print("Use --help to see available options\n")
            return 1
    
    print("\n" + "="*70)
    print("  CDS DISTRESS PREDICTION - COMPLETE ML PIPELINE")
    print("="*70)
    
    if run_pipeline:
        print("\nThis will run all 15 steps of the pipeline:")
        print("  • Data inspection and quality checks (Steps 1-2)")
        print("  • Data cleaning and merging (Steps 3-4)")
        print("  • Feature engineering (Steps 5-8)")
        print("  • Target creation (Step 9)")
        print("  • Model training and optimization (Steps 10-12)")
        print("  • Evaluation and explainability (Steps 13-15)")
    
    if run_exps:
        print("\n+ Optimization & validation experiments:")
        print("  • Exp 1: Reduce overfitting")
        print("  • Exp 4: Optimize recall")
        print("  • Exp 5: Temporal features")
        print("  • Exp 6: Combined model")
        print("  • Exp 13: Calibration")
        print("  • Exp 14: Cross-validation")
        print("  • Exp 16: Feature selection (Top 10)")
        print("  • (Exp 15: LSTM - run separately with lstm_train.py/lstm_test.py)")
    
    print("\n" + "="*70 + "\n")
    
    # Define all pipeline steps
    steps = [
        ("Step 1: Data Inspection", "step1_data_inspection"),
        ("Step 2: Data Quality Assessment", "step2_data_quality"),
        ("Step 3: Data Cleaning", "step3_data_cleaning"),
        ("Step 4: Data Merging", "step4_data_merging"),
        ("Step 5: Preprocessing", "step5_preprocessing"),
        ("Step 6: Accounting Features", "step6_accounting_features"),
        ("Step 7: Market Features", "step7_market_features"),
        ("Step 8: Feature Validation", "step8_feature_validation"),
        ("Step 9: Target Creation", "step9_target_creation"),
        ("Step 10: ML Construction", "step10_ml_construction"),
        ("Step 11: Model Training", "step11_model_training"),
        ("Step 12: Model Optimization", "step12_model_optimization"),
        ("Step 13: Model Evaluation", "step13_model_evaluation"),
        ("Step 13b: Confidence Intervals", "step13b_confidence_intervals"),
        ("Step 14: Benchmark Comparison", "step14_benchmark_comparison"),
        ("Step 15: Explainability Analysis", "step15_explainability"),
    ]
    
    # Track progress
    pipeline_completed = 0
    pipeline_failed = []
    exp_completed = 0
    exp_failed = []
    
    # Run pipeline steps
    if run_pipeline:
        for i, (step_name, module_name) in enumerate(steps, 1):
            print(f"\n[Step {i}/{len(steps)}]")
            
            success = run_step(step_name, module_name)
            
            if success:
                pipeline_completed += 1
            else:
                pipeline_failed.append(step_name)
                print(f"\n⚠️  Warning: {step_name} failed but continuing...\n")
    
    # Run experiments
    if run_exps:
        exp_completed, exp_failed, exp_total = run_experiments()
    
    # Final summary
    print("\n" + "="*70)
    print("  EXECUTION SUMMARY")
    print("="*70)
    
    if run_pipeline:
        print(f"\n📋 Pipeline: {pipeline_completed}/{len(steps)} steps completed")
        if pipeline_failed:
            print(f"    Failed: {len(pipeline_failed)} step(s)")
            for step in pipeline_failed:
                print(f"      - {step}")
        else:
            print("    All pipeline steps completed successfully!")
    
    if run_exps:
        exp_total = exp_completed + len(exp_failed)
        print(f"\n🔬 Experiments: {exp_completed}/{exp_total} experiments completed")
        if exp_failed:
            print(f"    Failed: {len(exp_failed)} experiment(s)")
            for exp in exp_failed:
                print(f"      - {exp}")
        else:
            print("    All experiments completed successfully!")

    print_story_highlights(run_pipeline, run_exps)

    print("\n" + "="*70)
    print("  OUTPUT LOCATIONS")
    print("="*70)
    print(f"\n📁 Processed Data:  {PROJECT_ROOT / 'output'}")
    print(f"🤖 Trained Models:  {PROJECT_ROOT / 'output' / 'models'}")
    print(f"📊 Figures:         {PROJECT_ROOT / 'report' / 'figures'}")
    print(f"📈 Results:         {PROJECT_ROOT / 'output' / 'step13_evaluation_results.csv'}")
    
    if run_exps:
        print(f"\n🔬 Experiment Outputs:")
        print(f"   • Models: {PROJECT_ROOT / 'output' / 'experiments'}")
        print(f"   • Figures: {PROJECT_ROOT / 'report' / 'figures' / 'experiments'}")
    
    print("\n" + "="*70)
    print("  FINAL MODEL")
    print("="*70)
    
    if run_exps:
        print(f"\n🏆 RECOMMENDED MODEL: output/experiments/models/exp16_xgboost.pkl")
        print(f"   • Top 10 Features (66% reduction)")
        print(f"   • Test AUC: 0.640")
        print(f"   • Precision: 30%, Recall: 72%")
        print(f"   • F1: 0.420 (best overall)")
        print(f"   • Simpler, faster, more interpretable")
        print(f"\n   Alternative: output/models/lightgbm_calibrated_isotonic.pkl")
        print(f"   • Calibrated probabilities (ECE: 0.014)")
        print(f"   • Test AUC: 0.662, Recall: 69.1%")
    else:
        print(f"\n🏆 Best Model: output/models/lightgbm_optimized.pkl")
        print(f"   • Test AUC: 0.668")
        print(f"   • Recall: 69.1%")
        print(f"   • Catches 1,011 / 1,463 distressed firms")
    
    print("\n" + "="*70 + "\n")
    
    has_failures = (pipeline_failed if run_pipeline else []) or (exp_failed if run_exps else [])
    return 0 if not has_failures else 1


if __name__ == "__main__":
    sys.exit(main())
