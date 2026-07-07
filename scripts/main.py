from file_manager import FileManager

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
        
        # python script/infer_MambaSCD.py  --dataset 'SECOND'  \
        #                          --model_type 'MambaSCD_Tiny' \
        #                          --test_dataset_path '<dataset_path>/SECOND/test' \
        #                          --test_data_list_path '<dataset_path>/SECOND/test_set.txt' \
        #                          --cfg '<project_path>/ChangeMamba/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml' \
        #                          --model_checkpoint_path '<saved_model_path>/[your_trained_model].pth'

        print(self.config)
        cmd = [
            sys.executable, f"../changedetection/script/{self.config.script_name}.py",
            '--dataset', self.config.dataset_name,
            '--model_type', self.config.model_name,
            '--test_dataset_path', test_dataset_path,
            '--test_data_list_path', test_data_list_path,
            '--cfg', self.config.config_path,
            '--model_checkpoint_path', self.config.model_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            print(result)
            output = result.stdout + result.stderr

            metrics = self._parse_inference_output(output)

            if metrics:
                print(f"✓ Run {run_num} completed successfully")
                print(f"  Metrics: {metrics}")

                return {
                    'run': run_num,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': metrics,
                    "model": self.config.model_name,
                    "dataset": self.config.dataset_name,
                    'success': True
                }
            else:
                print(f"✗ Run {run_num}: Could not parse metrics from output")
                return {
                    'run': run_num,
                    'timestamp': datetime.now().isoformat(),
                    'success': False,
                    "model": self.config.model_name,
                    "dataset": self.config.dataset_name,
                    'error': 'Failed to parse metrics'
                }
        except subprocess.TimeoutExpired:
            print(f"✗ Run {run_num}: Timeout (exceeded 1 hour)")
            return {
                'run': run_num,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                "model": self.config.model_name,
                "dataset": self.config.dataset_name,
                'error': 'Timeout'
            }
        except Exception as e:
            print(f"✗ Run {run_num}: Error - {str(e)}")
            return {
                'run': run_num,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                "model": self.config.model_name,
                "dataset": self.config.dataset_name,
                'error': str(e)
            }


    def run(self, use_batch=True):
        print(self.config)
        for batch_index in range(self.config.num_batches):
            if use_batch:
                self._reset_environment(self.config.batch_output_dir)
                FileManager.create_subset(
                    self.config.data_path,
                    self.config.batch_output_dir,
                    self.config.data_list_path,
                    self.config.images_per_batch,
                )
                
                test_dataset_path = self.config.batch_output_dir
            else:
                test_dataset_path = self.config.data_path
                
            
            print("got here")
            # def run_inference(self, run_num, test_dataset_path, test_data_list_path):

            res = []
            for i in range(self.config.iterations):
                
                self._reset_environment("../results")
                summary = self.run_inference(i + 1, test_dataset_path, self.config.data_list_path)
                summary["batch"] = batch_index
                res.append(summary)
        
            FileManager.append_results_to_json(res, "./summary.json")
            
    
            
            
if __name__ == "__main__":
    
    configs = [
        # { 
        #     "data_path": "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/SYSU-CD/test",
        #     "data_list_path": "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/SYSU-CD/test_set.txt",
        #     "dataset": "SYSU-CD",
        #     "model":  "MambaBCD_Tiny",
        #     "model_path": "/home/cilas/rodrigo/models/MambaBCD_Tiny_SYSU_F1_0.8316.pth",
        #     "config_path": "../changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml",
        #     "iterations": 1,
        #     "script_name": "infer_MambaBCD",
        # },
        # { 
        #     "data_path": "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/SYSU-CD/test",
        #     "data_list_path": "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/SYSU-CD/test_set.txt",
        #     "dataset": "SYSU-CD",
        #     "model":  "MambaBCD_Base",
        #     "model_path": "/home/cilas/rodrigo/models/MambaBCD_Base_SYSU_F1_0.8331.pth",
        #     "config_path": "../changedetection/configs/vssm1/vssm_base_224.yaml",
        #     "iterations": 1,
        #     "script_name": "infer_MambaBCD",
        # },
        # { 
        #     "data_path": "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/WHU",
        #     "data_list_path": "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/WHU/img_indexes.txt",
        #     "dataset": "WHU",
        #     "model":  "MambaBCD_Tiny",
        #     "model_path": "/home/cilas/rodrigo/models/MambaBCD_Tiny_WHU_F1_0.9409.pth",
        #     "config_path": "../changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml",
        #     "iterations": 1,
        #     "script_name": "infer_MambaBCD",
        # },
        # { 
        #     "data_path": "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/WHU",
        #     "data_list_path": "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/WHU/img_indexes.txt",
        #     "dataset": "WHU",
        #     "model":  "MambaBCD_Base",
        #     "model_path": "/home/cilas/rodrigo/models/MambaBCD_Base_WHU_F1_0.9419.pth",
        #     "config_path": "../changedetection/configs/vssm1/vssm_base_224.yaml",
        #     "iterations": 1,
        #     "script_name": "infer_MambaBCD",
        # },
        # { 
        #     "data_path": "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/LEVIR-CD+/test",
        #     "data_list_path": "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/LEVIR-CD+/test_set.txt",
        #     "dataset": "LEVIR-CD+",
        #     "model":  "MambaBCD_Tiny",
        #     "model_path": "/home/cilas/rodrigo/models/MambaBCD_Tiny_LEVIRCD+_F1_0.8803.pth",
        #     "config_path": "../changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml",
        #     "iterations": 1,
        #     "script_name": "infer_MambaBCD",
        # },
        # { 
        #     "data_path": "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/LEVIR-CD+/test",
        #     "data_list_path": "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/LEVIR-CD+/test_set.txt",
        #     "dataset": "LEVIR-CD+",
        #     "model":  "MambaBCD_Small",
        #     "model_path": "/home/cilas/rodrigo/models/MambaBCD_Small_LEVIRCD+_F1_0.8825.pth",
        #     "config_path": "../changedetection/configs/vssm1/vssm_small_224.yaml",
        #     "iterations": 1,
        #     "script_name": "infer_MambaBCD",
        # },
        # { 
        #     "data_path": "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/SECOND/test",
        #     "data_list_path": "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/SECOND/test_set.txt",
        #     "dataset": "SECOND",
        #     "model":  "MambaSCD_Base",
        #     "model_path": "/home/cilas/rodrigo/models/MambaSCD_Base_SECOND_SeK_0.2292.pth",
        #     "config_path": "../changedetection/configs/vssm1/vssm_base_224.yaml",
        #     "iterations": 1,
        #     "script_name": "infer_MambaSCD",
        # },
        # { 
        #     "data_path": "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/SECOND/test",
        #     "data_list_path": "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/SECOND/test_set.txt",
        #     "dataset": "SECOND",
        #     "model":  "MambaSCD_Tiny",
        #     "model_path": "/home/cilas/rodrigo/models/MambaSCD_Tiny_SECOND_SeK_0.2208.pth",
        #     "config_path": "../changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml",
        #     "iterations": 1,
        #     "script_name": "infer_MambaSCD",
        # },
        # { 
        #     "data_path": "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/WHU",
        #     "data_list_path": "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/WHU/img_indexes.txt",
        #     "dataset": "WHU",
        #     "model":  "MambaBCD_Small",
        #     "model_path": "/home/cilas/rodrigo/models/MambaBCD_Small_WHU_F1_0.9404.pth",
        #     "config_path": "../changedetection/configs/vssm1/vssm_small_224.yaml",
        #     "iterations": 1,
        #     "script_name": "infer_MambaBCD",
        # },
        # { 
        #     "data_path": "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/SYSU-CD/test",
        #     "data_list_path": "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/SYSU-CD/test_set.txt",
        #     "dataset": "SYSU-CD",
        #     "model":  "MambaBCD_Small",
        #     "model_path": "/home/cilas/rodrigo/models/MambaBCD_Small_SYSU_F1_0.8336.pth",
        #     "config_path": "../changedetection/configs/vssm1/vssm_small_224.yaml",
        #     "iterations": 1,
        #     "script_name": "infer_MambaBCD",
        # },
        { 
            "data_path": "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/LEVIR-CD+/test",
            "data_list_path": "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/LEVIR-CD+/test_set.txt",
            "dataset": "LEVIR-CD+",
            "model":  "MambaBCD_Base",
            "model_path": "/home/cilas/rodrigo/models/MambaBCD_Base_LEVIRCD+_F1_0.8823.pth",
            "config_path": "../changedetection/configs/vssm1/vssm_base_224.yaml",
            "iterations": 1,
            "script_name": "infer_MambaBCD",
        }
    ]
    
    for config in configs:
        
        config = Config(
            data_path = config.get("data_path"),
            data_list_path = config.get("data_list_path"),
            dataset_name = config.get("dataset"),
            model_name = config.get("model"),
            model_path = config.get("model_path"),
            config_path = config.get("config_path"),
            iterations = config.get("iterations"),
            script_name = config.get("script_name"),
        )
        client = MainClient(config=config)
    
        client.run(use_batch=False)
