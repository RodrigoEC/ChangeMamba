#!/usr/bin/env python3
"""
Script to run inference multiple times and collect results into a JSON file.
"""

import sys
import json
import subprocess
import re
import shutil
from datetime import datetime
from pathlib import Path

def cleanup_results():
    """Remove generated result images to avoid interference between runs."""
    results_dir = Path('results/SYSU-CD/MambaBCD_Tiny/change_map')
    print("bora")
    if results_dir.exists():
        try:
            shutil.rmtree(results_dir)
            print(f"✓ Cleaned up results directory")
            return True
        except Exception as e:
            print(f"✗ Warning: Could not clean up results - {str(e)}")
            return False
    return True

def parse_inference_output(output_text):
    """Extract metrics from inference output."""
    metrics = {}

    # Look for INFER Summary section
    lines = output_text.split('\n')
    in_summary = False

    for line in lines:
        if 'INFER Summary' in line:
            in_summary = True
            continue

        if in_summary and line.strip():
            # Parse lines like "  Recall    : 0.7218"
            match = re.match(r'\s*(\w+)\s*:\s*([\d.]+)', line)
            if match:
                metric_name = match.group(1)
                metric_value = float(match.group(2))
                metrics[metric_name] = metric_value
            elif 'done!' in line.lower():
                break

    return metrics

def run_inference(run_num):
    """Run a single inference and return metrics."""
    print(f"\n{'='*60}")
    print(f"Running inference {run_num}/5...")
    print(f"{'='*60}")

    cmd = [
        sys.executable, 'changedetection/script/infer_MambaBCD.py',
        '--dataset', 'SYSU-CD',
        '--model_type', 'MambaBCD_Tiny',
        '--test_dataset_path', 'changedetection/datasets/data_name_list/SYSU-CD/test',
        '--test_data_list_path', 'changedetection/datasets/data_name_list/SYSU-CD/test_set.txt',
        '--cfg', 'changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml',
        '--model_checkpoint_path', '/Users/rodrigoec/Downloads/MambaBCD_Tiny_SYSU_F1_0.8316.pth',
        '--cuda', 'False'
    ]

    try:
        cleanup_results()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        output = result.stdout + result.stderr

        metrics = parse_inference_output(output)

        if metrics:
            print(f"✓ Run {run_num} completed successfully")
            print(f"  Metrics: {metrics}")

            # Clean up results images after successful run
            print("oi")

            return {
                'run': run_num,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'success': True
            }
        else:
            print(f"✗ Run {run_num}: Could not parse metrics from output")
            return {
                'run': run_num,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': 'Failed to parse metrics'
            }
    except subprocess.TimeoutExpired:
        print(f"✗ Run {run_num}: Timeout (exceeded 1 hour)")
        return {
            'run': run_num,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': 'Timeout'
        }
    except Exception as e:
        print(f"✗ Run {run_num}: Error - {str(e)}")
        return {
            'run': run_num,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': str(e)
        }

def main():
    """Run inference 5 times and save results to JSON."""
    num_runs = 5
    results = []

    print(f"Starting {num_runs} inference runs...")
    print(f"Start time: {datetime.now().isoformat()}")

    for i in range(1, num_runs + 1):
        result = run_inference(i)
        results.append(result)

    # Save to JSON
    output_file = Path('inference_results.json')
    output_data = {
        'total_runs': num_runs,
        'start_time': results[0]['timestamp'],
        'end_time': results[-1]['timestamp'],
        'results': results,
        'summary': {
            'successful_runs': sum(1 for r in results if r['success']),
            'failed_runs': sum(1 for r in results if not r['success'])
        }
    }

    # Calculate average metrics if all runs succeeded
    if output_data['summary']['successful_runs'] == num_runs:
        all_metrics = {}
        for result in results:
            for metric_name, metric_value in result['metrics'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(metric_value)

        averages = {
            name: sum(values) / len(values)
            for name, values in all_metrics.items()
        }
        output_data['average_metrics'] = averages

        print(f"\n{'='*60}")
        print("Average Metrics across all runs:")
        print(f"{'='*60}")
        for metric, value in averages.items():
            print(f"  {metric:12s}: {value:.4f}")

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file.absolute()}")
    print(f"{'='*60}")

    return output_data

if __name__ == '__main__':
    main()
