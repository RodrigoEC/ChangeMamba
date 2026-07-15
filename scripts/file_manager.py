import json
import random
import shutil
from pathlib import Path
from PIL import Image


class FileManager:
    
    @staticmethod
    def convert_tif_to_png(dir_path: str | Path, output_dir: str | Path = None) -> list[str]:
        """
        Convert all TIF files in a directory to PNG format.

        Args:
            dir_path: Path to directory containing TIF files
            output_dir: Directory to save PNG files. If None, saves in the same directory as the TIF

        Returns:
            List of paths to created PNG files
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        if output_dir is None:
            output_dir = dir_path
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        png_files = []

        for tif_file in list(dir_path.glob("*.tif*")) + list(dir_path.glob("*.TIF*")):
            if tif_file.is_file():
                try:
                    with Image.open(tif_file) as image:
                        png_path = output_dir / f"{tif_file.stem}.png"
                        image.convert("RGB").save(png_path, "PNG")
                        png_files.append(str(png_path))
                        print(f"Converted: {tif_file.name} -> {png_path.name}")
                except Exception as e:
                    print(f"Error converting {tif_file.name}: {e}")

        return png_files

    @staticmethod
    def split_image_into_tiles(image_path: str | Path, output_dir: str | Path, tile_size: int = 256) -> list[str]:
        """
        Split an image into tiles of specified size and save them to output directory.

        Args:
            image_path: Path to the image file
            output_dir: Directory to save the tiles
            tile_size: Size of each tile in pixels (default: 256x256)

        Returns:
            List of paths to created tile images
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tile_files = []

        try:
            with Image.open(image_path) as image:
                width, height = image.size
                base_name = image_path.stem
                
                index = 0 

                for y in range(0, height, tile_size):
                    for x in range(0, width, tile_size):
                        box = (x, y, min(x + tile_size, width), min(y + tile_size, height))
                        tile = image.crop(box)

                        row = y // tile_size
                        col = x // tile_size
                        tile_name = f"{index:04d}.png"
                        tile_path = output_dir / tile_name

                        tile.save(tile_path, "PNG")
                        tile_files.append(str(tile_path))
                        print(f"Saved tile: {tile_name}")
                        index += 1

        except Exception as e:
            print(f"Error splitting image {image_path.name}: {e}")

        return tile_files


    

    
    @staticmethod
    def create_subset(source_dir, output_dir, test_list_path, script_name, num_images=500):
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

        required_dirs = {'GT_CD', 'GT_T1', 'GT_T1_COLORED', 'GT_T2', 'GT_T2_COLORED', 'T1', 'T2'}
        
        if script_name == "infer_MambaBCD":
            required_dirs = {'GT', 'T1', 'T2'}
           
        for dir_name in required_dirs:
            dir_path = source_path / dir_name
            if not dir_path.exists() or not dir_path.is_dir():
                raise FileNotFoundError(f"Required directory not found: {dir_path}")

        # Get list of images from GT directory (skip macOS AppleDouble metadata files)
        gt_dir = source_path / 'GT_CD'
        if script_name == "infer_MambaBCD":
            gt_dir = source_path / "GT"
        image_files = sorted([f.name for f in gt_dir.iterdir() if f.is_file() and not f.name.startswith('._')])

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

    @staticmethod
    def append_results_to_json(results: list[dict], output_file: str | Path):
        """
        Append a list of result objects to a JSON file.

        Args:
            results: List of result dictionaries to append
            output_file: Path to the output JSON file
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        existing_results = []
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    existing_results = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode existing JSON file {output_file}")
                existing_results = []

        existing_results.extend(results)

        with open(output_file, 'w') as f:
            json.dump(existing_results, f, indent=2)

        print(f"Appended {len(results)} results to {output_file}")


if __name__ == "__main__":
    # files = FileManager.convert_tif_to_png("/Users/rodrigoec/Downloads/WHU-test", "/Users/rodrigoec/cc/mestrado/ChangeMamba/changedetection/datasets/data_name_list/WHU")
    # print(files)
    
    FileManager.split_image_into_tiles("/Users/rodrigoec/Downloads/whu/data/2012/whole_image/test/image/2012_test.tif", "/Users/rodrigoec/cc/mestrado/ChangeMamba/changedetection/datasets/data_name_list/WHU/T1")
    FileManager.split_image_into_tiles("/Users/rodrigoec/Downloads/whu/data/2016/whole_image/test/image/2016_test.tif", "/Users/rodrigoec/cc/mestrado/ChangeMamba/changedetection/datasets/data_name_list/WHU/T2")
    FileManager.split_image_into_tiles("/Users/rodrigoec/Downloads/whu/data/change_label/test/change_label.tif", "/Users/rodrigoec/cc/mestrado/ChangeMamba/changedetection/datasets/data_name_list/WHU/GT")