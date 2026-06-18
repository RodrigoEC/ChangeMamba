from create_image_subset import FileManager

from config import Config

from datetime import datetime
from pathlib import Path
import re
import shutil
import subprocess
import sys

class MainClient:

    def __init__(self, config: Config = Config()):
        self.config = config

    def _reset_environment(self, path):
        """Remove generated result images to avoid interference between runs."""
        results_dir = Path(path)
        # deu bom!

        if results_dir.exists():
            try:
                shutil.rmtree(results_dir)
                print(f"✓ Cleaned up results directory")
                return True
            except Exception as e:
                print(f"✗ Warning: Could not clean up results - {str(e)}")
                return False
        return True
    
    def _parse_inference_output(self, output_text):
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

    def run_inference(self, run_num, test_dataset_path, test_data_list_path):
        """Run a single inference and return metrics."""
        print(f"\n{'='*60}")
        print(f"Running inference {run_num}/{self.config.iterations}...")
        print(f"{'='*60}")

        cmd = [
            sys.executable, '../changedetection/script/infer_MambaBCD.py',
            '--dataset', self.config.dataset_name,
            '--model_type', self.config.model_name,
            '--test_dataset_path', test_dataset_path,
            '--test_data_list_path', test_data_list_path,
            '--cfg', self.config.config_path,
            '--model_checkpoint_path', self.config.model_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            output = result.stdout + result.stderr

            metrics = self._parse_inference_output(output)

            if metrics:
                print(f"✓ Run {run_num} completed successfully")
                print(f"  Metrics: {metrics}")

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


    def run(self):
        
        for _ in range(self.config.num_batches):
            self._reset_environment(self.config.batch_output_dir)
            FileManager.create_subset(
                self.config.data_path,
                self.config.batch_output_dir,
                self.config.data_list_path,
                self.config.images_per_batch,
            )

            for i in range(self.config.iterations):
                
                self._reset_environment("../results")
                self.run_inference(i + 1, self.config.batch_output_dir, self.config.data_list_path)

if __name__ == "__main__":
    config = Config(
        
    )
    client = MainClient(config=config)
    client.run()
