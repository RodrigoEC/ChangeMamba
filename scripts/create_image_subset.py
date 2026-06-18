import random
import shutil
from pathlib import Path


class FileManager:
    
    
    
    
    @staticmethod
    def create_subset(source_dir, output_dir, test_list_path, num_images=500):
        """
        Create a subset of images maintaining the GT/T1/T2 directory structure.

        Args:
            source_dir: Path to the source directory containing GT, T1, T2 subdirs
            output_dir: Path where the subset directory will be created
            num_images: Number of random images to select (default: 500)
            test_list_path: Optional path to save a test list file (one image per line)
        """
        source_path = Path(source_dir)
        output_path = Path(output_dir)

        # Validate source directory
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source_path}")

        required_dirs = {'GT', 'T1', 'T2'}
        for dir_name in required_dirs:
            dir_path = source_path / dir_name
            if not dir_path.exists() or not dir_path.is_dir():
                raise FileNotFoundError(f"Required directory not found: {dir_path}")

        # Get list of images from GT directory
        gt_dir = source_path / 'GT'
        image_files = sorted([f.name for f in gt_dir.iterdir() if f.is_file()])

        if len(image_files) < num_images:
            print(f"Warning: Only {len(image_files)} images found, but {num_images} requested.")
            num_images = len(image_files)

        # Randomly select images
        selected_images = sorted(random.sample(image_files, num_images))
        print(f"Selected {len(selected_images)} random images from {len(image_files)} total")

        # Create output directory structure
        output_path.mkdir(parents=True, exist_ok=True)

        for dir_name in required_dirs:
            (output_path / dir_name).mkdir(exist_ok=True)

        # Copy selected images from all three directories
        for img_name in selected_images:
            for dir_name in required_dirs:
                src_file = source_path / dir_name / img_name
                dst_file = output_path / dir_name / img_name

                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                else:
                    print(f"Warning: File not found: {src_file}")

        print(f"Subset created at: {output_path}")
        print(f"Total images copied per directory: {len(selected_images)}")

        # Generate test list file if requested
        if test_list_path:
            test_list_file = Path(test_list_path)
            test_list_file.parent.mkdir(parents=True, exist_ok=True)

            with open(test_list_file, 'w') as f:
                for img_name in selected_images:
                    f.write(f"{img_name}\n")

            print(f"Test list created at: {test_list_file}")
            print(f"  Total entries: {len(selected_images)}")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description='Create a random subset of images maintaining GT/T1/T2 structure'
#     )
#     parser.add_argument(
#         'source_dir',
#         help='Path to source directory containing GT, T1, T2 subdirectories'
#     )
#     parser.add_argument(
#         'output_dir',
#         help='Path where the subset directory will be created'
#     )
#     parser.add_argument(
#         '-n', '--num-images',
#         type=int,
#         default=500,
#         help='Number of random images to select (default: 500)'
#     )
#     parser.add_argument(
#         '-t', '--test-list',
#         help='Optional: path to save test list file (one image name per line)'
#     )

#     args = parser.parse_args()

#     create_subset(args.source_dir, args.output_dir, args.num_images, args.test_list)
