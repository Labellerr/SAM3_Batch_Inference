"""
SAM3 Batch Inference Script
Performs inference on a folder of images using SAM3 model with multiple text prompts
and saves annotations in COCO-JSON format.

Supports multi-class detection by providing a list of text prompts.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Union
import warnings

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning, module='triton')

from sam3.sam3.model_builder import build_sam3_image_model
from sam3.sam3.model.sam3_image_processor import Sam3Processor


class SAM3BatchInference:
    """Batch inference using SAM3 model with COCO-JSON output format."""
    
    def __init__(self, model_checkpoint_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the SAM3 batch inference processor.
        
        Args:
            model_checkpoint_path: Path to SAM3 model checkpoint
            confidence_threshold: Confidence threshold for detections (0.0-1.0)
        """
        self.model_checkpoint_path = model_checkpoint_path
        self.confidence_threshold = confidence_threshold
        self.processor = None
        self._load_model()
        
    def _load_model(self):
        """Load the SAM3 model."""
        print(f"Loading SAM3 model from: {self.model_checkpoint_path}")
        model = build_sam3_image_model(checkpoint_path=self.model_checkpoint_path)
        self.processor = Sam3Processor(model, confidence_threshold=self.confidence_threshold)
        print("Model loaded successfully")
        
    def _mask_to_rle(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        Convert binary mask to RLE (Run-Length Encoding) format.
        
        Args:
            mask: Binary mask array (H, W)
            
        Returns:
            RLE dictionary with 'counts' and 'size'
        """
        # Flatten mask in Fortran order (column-major)
        pixels = mask.flatten(order='F')
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        
        return {
            'counts': runs.tolist(),
            'size': list(mask.shape)
        }
    
    def _mask_to_polygon(self, mask: np.ndarray) -> List[List[float]]:
        """
        Convert binary mask to polygon format (list of [x, y] coordinates).
        
        Args:
            mask: Binary mask array (H, W)
            
        Returns:
            List of polygons, where each polygon is a flat list [x1, y1, x2, y2, ...]
        """
        import cv2
        
        # Convert boolean mask to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for contour in contours:
            # Flatten contour and convert to list
            if len(contour) >= 3:  # Need at least 3 points for a polygon
                polygon = contour.flatten().tolist()
                polygons.append(polygon)
        
        return polygons
    
    def _calculate_bbox_area(self, bbox: List[float]) -> float:
        """Calculate area of bounding box in COCO format [x, y, width, height]."""
        return bbox[2] * bbox[3]
    
    def process_image_single_prompt(
        self, 
        image_path: str, 
        text_prompt: str, 
        image_id: int, 
        category_id: int,
        max_image_size: int = None
    ) -> Dict[str, Any]:
        """
        Process a single image with a single text prompt and return detection results.
        
        Args:
            image_path: Path to the image file
            text_prompt: Text prompt for detection
            image_id: Unique image ID for COCO format
            category_id: Category ID for this prompt
            max_image_size: Maximum dimension (width or height) for image. 
                           Larger images will be resized to fit.
            
        Returns:
            Dictionary containing image info, annotations, and category_id
        """
        import gc
        
        # Initialize variables for cleanup
        image = None
        resized_image = None
        inference_state = None
        output = None
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            original_width, original_height = image.size
            
            # Resize if image is too large
            scale_factor = 1.0
            if max_image_size is not None:
                max_dim = max(original_width, original_height)
                if max_dim > max_image_size:
                    scale_factor = max_image_size / max_dim
                    new_width = int(original_width * scale_factor)
                    new_height = int(original_height * scale_factor)
                    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
                    image.close()
                    image = resized_image
            
            # Run inference
            inference_state = self.processor.set_image(image)
            output = self.processor.set_text_prompt(state=inference_state, prompt=text_prompt)
            
            # Extract results
            masks = output["masks"]
            boxes = output["boxes"]
            scores = output["scores"]
            
            # Convert to numpy immediately to free GPU tensors
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            
            # Handle mask shape (N, 1, H, W) -> (N, H, W)
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)
            
            # Scale boxes back to original size if image was resized
            if scale_factor != 1.0:
                boxes = boxes / scale_factor
                # Resize masks back to original size
                import cv2
                resized_masks = []
                for mask in masks:
                    resized_mask = cv2.resize(
                        mask.astype(np.uint8), 
                        (original_width, original_height), 
                        interpolation=cv2.INTER_NEAREST
                    )
                    resized_masks.append(resized_mask.astype(bool))
                masks = np.array(resized_masks)
            
            return {
                'image_info': {
                    'id': image_id,
                    'file_name': os.path.basename(image_path),
                    'width': original_width,
                    'height': original_height,
                    'date_captured': datetime.now().isoformat()
                },
                'masks': masks,
                'boxes': boxes,
                'scores': scores,
                'category_id': category_id
            }
            
        finally:
            # Cleanup to prevent memory accumulation
            if image is not None:
                image.close()
            if inference_state is not None:
                del inference_state
            if output is not None:
                del output
            gc.collect()
            torch.cuda.empty_cache()
    
    def process_folder(
        self,
        input_folder: str,
        text_prompts: Union[str, List[str]],
        output_json_path: str,
        segmentation_format: str = 'polygon',
        image_extensions: List[str] = None,
        max_image_size: int = 1024
    ):
        """
        Process all images in a folder with multiple text prompts and save annotations in COCO-JSON format.
        
        Uses prompt-sequential processing strategy: processes one prompt at a time across all images
        to optimize memory usage when dealing with large datasets.
        
        Args:
            input_folder: Path to folder containing images
            text_prompts: Single text prompt string or list of text prompts for multi-class detection
            output_json_path: Path to save COCO-JSON output
            segmentation_format: 'rle' or 'polygon' for segmentation encoding
            image_extensions: List of image file extensions to process
            max_image_size: Maximum dimension for images (default 1024). Larger images will be 
                           resized to prevent GPU memory issues. Set to None to disable.
        """
        # Normalize text_prompts to list
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # Find all image files
        input_path = Path(input_folder)
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(input_path.glob(f'*{ext}')))
            image_files.extend(list(input_path.glob(f'*{ext.upper()}')))
        
        image_files = sorted(set(image_files))  # Remove duplicates and sort
        
        if not image_files:
            print(f"No images found in {input_folder}")
            return
        
        print(f"Found {len(image_files)} images to process")
        print(f"Text prompts ({len(text_prompts)}): {text_prompts}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Segmentation format: {segmentation_format}")
        print(f"Processing strategy: Prompt-sequential (memory optimized)")
        
        # Build categories from text prompts (0-indexed for compatibility)
        categories = []
        for idx, prompt in enumerate(text_prompts):
            categories.append({
                'id': idx,
                'name': prompt,
                'supercategory': 'object'
            })
        
        # Initialize COCO format structure
        coco_output = {
            'info': {
                'description': f'SAM3 Multi-Class Inference Results',
                'version': '1.0',
                'year': datetime.now().year,
                'contributor': 'SAM3 Batch Inference',
                'date_created': datetime.now().isoformat(),
                'text_prompts': text_prompts,
                'confidence_threshold': self.confidence_threshold
            },
            'licenses': [],
            'images': [],
            'annotations': [],
            'categories': categories
        }
        
        # First pass: collect image info (only once per image)
        image_info_dict = {}
        for image_id, image_path in enumerate(image_files):
            image = Image.open(str(image_path))
            width, height = image.size
            image_info = {
                'id': image_id,
                'file_name': os.path.basename(str(image_path)),
                'width': width,
                'height': height,
                'date_captured': datetime.now().isoformat()
            }
            image_info_dict[image_id] = image_info
            coco_output['images'].append(image_info)
            image.close()
        
        annotation_id = 0
        
        # Process prompts sequentially (one prompt across all images at a time)
        for category_id, text_prompt in enumerate(text_prompts):
            print(f"\n{'='*60}")
            print(f"Processing prompt {category_id + 1}/{len(text_prompts)}: '{text_prompt}'")
            print(f"{'='*60}")
            
            # Process each image for this prompt
            for image_id, image_path in enumerate(tqdm(image_files, desc=f"[{text_prompt}]")):
                try:
                    result = self.process_image_single_prompt(
                        str(image_path), 
                        text_prompt, 
                        image_id, 
                        category_id,
                        max_image_size=max_image_size
                    )
                    
                    # Add annotations
                    masks = result['masks']
                    boxes = result['boxes']
                    scores = result['scores']
                    
                    for i in range(len(boxes)):
                        # Convert box from [x1, y1, x2, y2] to COCO format [x, y, width, height]
                        x1, y1, x2, y2 = boxes[i]
                        bbox_coco = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                        
                        # Get mask
                        mask = masks[i].astype(bool)
                        
                        # Convert segmentation based on format
                        if segmentation_format == 'rle':
                            segmentation = self._mask_to_rle(mask)
                        else:  # polygon
                            segmentation = self._mask_to_polygon(mask)
                        
                        # Calculate area
                        area = float(np.sum(mask))
                        
                        annotation = {
                            'id': annotation_id,
                            'image_id': image_id,
                            'category_id': category_id,
                            'segmentation': segmentation,
                            'area': area,
                            'bbox': bbox_coco,
                            'iscrowd': 0,
                            'score': float(scores[i])
                        }
                        
                        coco_output['annotations'].append(annotation)
                        annotation_id += 1
                        
                except Exception as e:
                    print(f"\nError processing {image_path} with prompt '{text_prompt}': {str(e)}")
                    continue
            
            # Clear GPU cache after each prompt to free memory
            torch.cuda.empty_cache()
        
        # Save COCO JSON
        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_json_path, 'w') as f:
            json.dump(coco_output, f, indent=2)
        
        # Print summary
        total_images = len(coco_output['images'])
        total_annotations = len(coco_output['annotations'])
        avg_detections = total_annotations / total_images if total_images > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"{'='*60}")
        print(f"Total images processed: {total_images}")
        print(f"Total categories: {len(text_prompts)}")
        print(f"Total annotations: {total_annotations}")
        print(f"Average detections per image: {avg_detections:.2f}")
        
        # Per-category summary
        print(f"\nPer-category breakdown:")
        for cat in categories:
            cat_annotations = [a for a in coco_output['annotations'] if a['category_id'] == cat['id']]
            print(f"  - {cat['name']}: {len(cat_annotations)} detections")
        
        print(f"\nOutput saved to: {output_json_path}")
        print(f"{'='*60}")


def run_batch_inference(INPUT_FOLDER,# Path to folder containing input images
         TEXT_PROMPTS,
         MODEL_CHECKPOINT,
         CONFIDENCE_THRESHOLD=0.4,
         OUTPUT_JSON =None,
        ):
    """Main function to run batch inference with manual configuration."""
    
    # ============================================================
    # CONFIGURATION - Modify these values as needed
    # ============================================================
    if OUTPUT_JSON is None:
        input_folder_name = Path(INPUT_FOLDER).name
        OUTPUT_JSON = f"SAM3_Results/{input_folder_name}/annotations.json"
    
    # Segmentation format: 'rle' (Run-Length Encoding) or 'polygon'
    SEGMENTATION_FORMAT = 'polygon'  # Much more compact!
    
    # Image file extensions to process
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    # Maximum image dimension (width or height) in pixels
    # Large images will be resized to prevent GPU memory issues
    # Set to None to disable resizing (may cause out-of-memory errors)
    MAX_IMAGE_SIZE = 1024  # Recommended for 8GB VRAM
    
    # ============================================================
    # END CONFIGURATION
    # ============================================================
    
    # Validate inputs
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder does not exist: {INPUT_FOLDER}")
        return
    
    if not os.path.exists(MODEL_CHECKPOINT):
        print(f"Error: Model checkpoint does not exist: {MODEL_CHECKPOINT}")
        return
    
    # Initialize processor
    processor = SAM3BatchInference(
        model_checkpoint_path=MODEL_CHECKPOINT,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    
    # Process folder with multiple text prompts
    processor.process_folder(
        input_folder=INPUT_FOLDER,
        text_prompts=TEXT_PROMPTS,
        output_json_path=OUTPUT_JSON,
        segmentation_format=SEGMENTATION_FORMAT,
        image_extensions=IMAGE_EXTENSIONS,
        max_image_size=MAX_IMAGE_SIZE
    )
    
    # Clear GPU cache
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
