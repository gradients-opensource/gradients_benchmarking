#!/usr/bin/env python3
"""
Batch evaluation script for running multiple models through lm-evaluation-harness
Reads configuration from YAML file and outputs consolidated results
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import torch
import yaml

from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from lm_eval.utils import make_table
from lm_eval.evaluation_tracker import EvaluationTracker


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['tasks', 'models', 'eval_settings']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in config file")
    
    return config


def convert_model_args(model_args: Dict, eval_settings: Dict) -> str:
    """Convert model args dict to comma-separated string format"""
    args_list = []
    for key, value in model_args.items():
        if isinstance(value, bool):
            value = str(value).lower()
        args_list.append(f"{key}={value}")
    
    # Add trust_remote_code if specified in settings
    if eval_settings.get('trust_remote_code', False):
        args_list.append("trust_remote_code=true")
    
    return ",".join(args_list)


def evaluate_model(
    model_info: Dict,
    tasks: List[str],
    eval_settings: Dict,
    output_dir: Path
) -> Dict:
    """Evaluate a single model and return results"""
    
    model_name = model_info['name']
    logger.info(f"Starting evaluation for model: {model_name}")
    
    # Get model args
    model_args = convert_model_args(model_info.get('model_args', {}))
    
    # Override eval settings if specified for this model
    settings = eval_settings.copy()
    if 'eval_settings' in model_info:
        settings.update(model_info['eval_settings'])
    
    try:
        # Run evaluation
        start_time = time.time()
        
        # Create evaluation tracker for output
        model_output_path = str(output_dir / model_name.replace('/', '_'))
        evaluation_tracker = EvaluationTracker(output_path=model_output_path)
        
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=tasks,
            batch_size=settings.get('batch_size', 'auto'),
            device=settings.get('device', 'cuda'),
            use_cache=settings.get('use_cache', None),
            limit=settings.get('limit', None),
            log_samples=settings.get('log_samples', True),
            evaluation_tracker=evaluation_tracker,
        )
        
        eval_time = time.time() - start_time
        
        # Add metadata
        results['evaluation_metadata'] = {
            'model_name': model_name,
            'model_args': model_args,
            'evaluation_time_seconds': eval_time,
            'status': 'completed'
        }
        
        logger.info(f"Completed evaluation for {model_name} in {eval_time:.2f} seconds")
        
        # Print results table
        logger.info(f"\nResults for {model_name}:")
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))
        
        # Clear GPU memory if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"Cleared GPU cache after {model_name}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating model {model_name}: {str(e)}")
        logger.error(traceback.format_exc())
        
        return {
            'evaluation_metadata': {
                'model_name': model_name,
                'model_args': model_args,
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        }


def consolidate_results(all_results: List[Dict], config: Dict) -> Dict:
    """Consolidate all model results into a single structure"""
    
    consolidated = {
        'batch_evaluation': {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'tasks': config['tasks'],
                'eval_settings': config['eval_settings'],
                'num_models': len(config['models'])
            },
            'summary': {},
            'model_results': {}
        }
    }
    
    # Process each model's results
    for result in all_results:
        model_name = result['evaluation_metadata']['model_name']
        status = result['evaluation_metadata']['status']
        
        # Store full results
        consolidated['batch_evaluation']['model_results'][model_name] = result
        
        # Extract key metrics for summary if evaluation succeeded
        if status == 'completed' and 'results' in result:
            summary_metrics = {}
            
            # Process task-level results
            for task_name, task_results in result['results'].items():
                # Extract main metrics (acc, exact_match, etc.)
                metrics = {}
                for metric_key, value in task_results.items():
                    # Skip alias and sample count
                    if metric_key in ['alias', 'samples']:
                        continue
                    # Include main metrics and their stderr
                    if any(m in metric_key for m in ['acc', 'exact_match', 'f1', 'bleu', 'rouge']):
                        metrics[metric_key] = value
                
                if metrics:
                    summary_metrics[task_name] = metrics
            
            # Also include group results if available
            if 'groups' in result:
                for group_name, group_results in result['groups'].items():
                    group_metrics = {}
                    for metric_key, value in group_results.items():
                        if metric_key in ['alias', 'samples']:
                            continue
                        if any(m in metric_key for m in ['acc', 'exact_match', 'f1', 'bleu', 'rouge']):
                            group_metrics[metric_key] = value
                    
                    if group_metrics:
                        summary_metrics[f"group_{group_name}"] = group_metrics
            
            consolidated['batch_evaluation']['summary'][model_name] = {
                'status': 'completed',
                'metrics': summary_metrics,
                'eval_time': result['evaluation_metadata'].get('evaluation_time_seconds', 0)
            }
        else:
            consolidated['batch_evaluation']['summary'][model_name] = {
                'status': 'failed',
                'error': result['evaluation_metadata'].get('error', 'Unknown error')
            }
    
    return consolidated


def create_summary_csv(consolidated_results: Dict, output_path: Path):
    """Create a CSV summary of key metrics across all models"""
    
    summary = consolidated_results['batch_evaluation']['summary']
    
    # Collect all unique metrics across all models and tasks
    all_metrics = set()
    for model_name, model_summary in summary.items():
        if model_summary['status'] == 'completed':
            for task_name, metrics in model_summary.get('metrics', {}).items():
                for metric_name in metrics.keys():
                    all_metrics.add(f"{task_name}_{metric_name}")
    
    # Sort metrics for consistent ordering
    all_metrics = sorted(list(all_metrics))
    
    # Write CSV
    csv_path = output_path / 'summary.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['model', 'status', 'eval_time_seconds'] + all_metrics
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for model_name, model_summary in summary.items():
            row = {
                'model': model_name,
                'status': model_summary['status'],
                'eval_time_seconds': model_summary.get('eval_time', 0)
            }
            
            if model_summary['status'] == 'completed':
                for task_name, metrics in model_summary.get('metrics', {}).items():
                    for metric_name, value in metrics.items():
                        col_name = f"{task_name}_{metric_name}"
                        if col_name in fieldnames:
                            row[col_name] = value
            
            writer.writerow(row)
    
    logger.info(f"Saved CSV summary to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate multiple models using lm-evaluation-harness")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/batch_eval_config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Custom name for output files (default: batch_eval_TIMESTAMP)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous evaluation (skip completed models)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Set up output directory
    output_dir = Path(config['eval_settings']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectory for this batch run
    if args.output_name:
        batch_dir = output_dir / args.output_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = output_dir / f"batch_eval_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {batch_dir}")
    
    # Save config to output directory for reference
    config_copy_path = batch_dir / "config.yaml"
    with open(config_copy_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Check for resume
    completed_models = set()
    all_results = []
    
    if args.resume and (batch_dir / "consolidated_results.json").exists():
        logger.info("Resuming from previous evaluation...")
        with open(batch_dir / "consolidated_results.json", 'r') as f:
            previous_results = json.load(f)
            
            # Load previous results
            for model_name, model_result in previous_results['batch_evaluation']['model_results'].items():
                if model_result['evaluation_metadata']['status'] == 'completed':
                    completed_models.add(model_name)
                    all_results.append(model_result)
                    logger.info(f"Loaded previous results for: {model_name}")
    
    # Evaluate each model
    tasks = config['tasks']
    eval_settings = config['eval_settings']
    total_models = len(config['models'])
    
    for idx, model_info in enumerate(config['models'], 1):
        model_name = model_info['name']
        
        if model_name in completed_models:
            logger.info(f"[{idx}/{total_models}] Skipping {model_name} (already completed)")
            continue
        
        logger.info(f"[{idx}/{total_models}] Evaluating {model_name}")
        logger.info("=" * 80)
        
        result = evaluate_model(model_info, tasks, eval_settings, batch_dir)
        all_results.append(result)
        
        # Save intermediate results after each model
        if all_results:
            intermediate_consolidated = consolidate_results(all_results, config)
            intermediate_path = batch_dir / "consolidated_results_intermediate.json"
            with open(intermediate_path, 'w') as f:
                json.dump(intermediate_consolidated, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved intermediate results to {intermediate_path}")
        
        logger.info("=" * 80)
        logger.info("")
    
    # Consolidate all results
    logger.info("Consolidating all results...")
    consolidated = consolidate_results(all_results, config)
    
    # Save consolidated JSON
    json_path = batch_dir / "consolidated_results.json"
    with open(json_path, 'w') as f:
        json.dump(consolidated, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved consolidated results to {json_path}")
    
    # Create CSV summary
    create_summary_csv(consolidated, batch_dir)
    
    # Print final summary
    logger.info("\n" + "=" * 80)
    logger.info("BATCH EVALUATION COMPLETE")
    logger.info("=" * 80)
    
    summary = consolidated['batch_evaluation']['summary']
    completed = sum(1 for s in summary.values() if s['status'] == 'completed')
    failed = sum(1 for s in summary.values() if s['status'] == 'failed')
    
    logger.info(f"Total models evaluated: {len(summary)}")
    logger.info(f"Successful: {completed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Results saved to: {batch_dir}")
    
    # Print summary table
    print("\nSummary of Results:")
    print("-" * 80)
    for model_name, model_summary in summary.items():
        status = model_summary['status']
        if status == 'completed':
            print(f"\n{model_name}: {status}")
            # Print key metrics for each task/group
            for task_name, metrics in model_summary.get('metrics', {}).items():
                # Find the main accuracy/score metric (skip stderr values)
                main_metrics = [(k, v) for k, v in metrics.items() if 'stderr' not in k]
                if main_metrics:
                    # Print first main metric found
                    metric_name, value = main_metrics[0]
                    if isinstance(value, (int, float)):
                        print(f"  {task_name}: {metric_name} = {value:.4f}")
                    else:
                        print(f"  {task_name}: {metric_name} = {value}")
        else:
            print(f"\n{model_name}: {status}")
            print(f"  Error: {model_summary.get('error', 'Unknown error')}")
    print("-" * 80)


if __name__ == "__main__":
    main()