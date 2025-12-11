"""
Split large COCO JSON annotation files into smaller batches.
Each batch will contain a specified number of images with their annotations.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List


def split_coco_json(
    input_json_path: str,
    output_dir: str,
    images_per_batch: int = 100,
    output_prefix: str = "batch"
) -> List[str]:
    """
    Split a large COCO JSON file into smaller batches.
    
    Args:
        input_json_path: Path to the input COCO JSON file
        output_dir: Directory to save the batch files
        images_per_batch: Number of images per batch (default: 100)
        output_prefix: Prefix for output batch files (default: "batch")
        
    Returns:
        List of paths to the created batch files
    """
    print(f"Loading COCO JSON from: {input_json_path}")
    
    # Load the input JSON
    with open(input_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Extract components
    info = coco_data.get('info', {})
    licenses = coco_data.get('licenses', [])
    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])
    categories = coco_data.get('categories', [])
    
    total_images = len(images)
    total_annotations = len(annotations)
    
    print(f"Total images: {total_images}")
    print(f"Total annotations: {total_annotations}")
    print(f"Images per batch: {images_per_batch}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create image_id to annotations mapping for faster lookup
    print("Creating image-to-annotations mapping...")
    image_annotations = {}
    for ann in annotations:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # Split into batches
    num_batches = (total_images + images_per_batch - 1) // images_per_batch
    print(f"Creating {num_batches} batch files...")
    
    batch_files = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * images_per_batch
        end_idx = min(start_idx + images_per_batch, total_images)
        
        batch_images = images[start_idx:end_idx]
        batch_image_ids = {img['id'] for img in batch_images}
        
        # Get annotations for this batch
        batch_annotations = []
        for image_id in batch_image_ids:
            if image_id in image_annotations:
                batch_annotations.extend(image_annotations[image_id])
        
        # Create batch COCO structure
        batch_coco = {
            'info': info.copy(),
            'licenses': licenses.copy(),
            'images': batch_images,
            'annotations': batch_annotations,
            'categories': categories.copy()
        }
        
        # Update info with batch details
        batch_coco['info']['batch_number'] = batch_idx + 1
        batch_coco['info']['total_batches'] = num_batches
        batch_coco['info']['images_in_batch'] = len(batch_images)
        batch_coco['info']['annotations_in_batch'] = len(batch_annotations)
        
        # Save batch file
        batch_filename = f"{output_prefix}_{batch_idx + 1:03d}_of_{num_batches:03d}.json"
        batch_filepath = output_path / batch_filename
        
        with open(batch_filepath, 'w') as f:
            json.dump(batch_coco, f, indent=2)
        
        batch_files.append(str(batch_filepath))
        
        print(f"  Batch {batch_idx + 1}/{num_batches}: {len(batch_images)} images, "
              f"{len(batch_annotations)} annotations -> {batch_filename}")
    
    print(f"\n{'='*60}")
    print(f"Splitting complete!")
    print(f"{'='*60}")
    print(f"Created {len(batch_files)} batch files in: {output_dir}")
    print(f"{'='*60}")
    
    return batch_files


def main():
    """Main function with configuration."""
    
    # ============================================================
    # CONFIGURATION - Modify these values as needed
    # ============================================================
    
    # Path to the large COCO JSON file
    INPUT_JSON = r"output\annotations.json"
    
    # Directory to save batch files
    OUTPUT_DIR = r"output\batches"
    
    # Number of images per batch
    IMAGES_PER_BATCH = 100
    
    # Prefix for output batch files
    OUTPUT_PREFIX = "annotations_batch"
    
    # ============================================================
    # END CONFIGURATION
    # ============================================================
    
    if not os.path.exists(INPUT_JSON):
        print(f"Error: Input file does not exist: {INPUT_JSON}")
        return
    
    batch_files = split_coco_json(
        input_json_path=INPUT_JSON,
        output_dir=OUTPUT_DIR,
        images_per_batch=IMAGES_PER_BATCH,
        output_prefix=OUTPUT_PREFIX
    )
    
    print(f"\nBatch files created:")
    for batch_file in batch_files:
        print(f"  - {batch_file}")


if __name__ == '__main__':
    main()
