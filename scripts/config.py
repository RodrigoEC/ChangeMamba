from dataclasses import dataclass


@dataclass
class Config:

    # Variables from original dataset
    data_path: str = (
        "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/SYSU-CD/test"
    )
    data_list_path: str = (
        "/home/cilas/rodrigo/ChangeMamba/changedetection/datasets/data_name_list/SYSU-CD/test_set.txt"
    )
    iterations: int = 5
    dataset_name: str = "SYSU-CD"
    model_name: str = "MambaBCD_Tiny"
    model_path: str = "/home/cilas/rodrigo/MambaBCD_Tiny_SYSU_F1_0.8316.pth"
    config_path: str = "../changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"

    # Batch creation variables
    batch_output_dir: str = "/home/cilas/rodrigo/ChangeMamba/scripts/batch"
    result_output_dir: str = "/home/cilas/rodrigo/ChangeMamba/scripts/results"
    images_per_batch: int = 500
    num_batches: int = 5
