"""
Batch upload pre-annotations from multiple COCO JSON files.
Uploads annotation files one by one with progress tracking and error handling.

Usage:
    # From notebook or another script:
    from batch_upload_preannot import upload_preannotations
    
    result = upload_preannotations(
        project_id='your_project_id',
        annotation_format='coco_json',
        batch_annotation_dir='path/to/annotations'
    )
"""

from labellerr.core import LabellerrClient
from labellerr.core.projects import LabellerrProject
from labellerr.core.exceptions import LabellerrError
from dotenv import load_dotenv
import os
import time
from pathlib import Path
from typing import List, Optional


def _get_client(env_path: str = '.env') -> LabellerrClient:
    """
    Initialize and return a LabellerrClient using credentials from .env file.
    
    Args:
        env_path: Path to .env file containing credentials
        
    Returns:
        Initialized LabellerrClient instance
    """
    load_dotenv(env_path)
    
    client_id = os.getenv("CLIENT_ID")
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")
    
    if not all([client_id, api_key, api_secret]):
        raise ValueError("Missing credentials. Ensure CLIENT_ID, API_KEY, and API_SECRET are set in .env file")
    
    return LabellerrClient(
        api_key=api_key,
        api_secret=api_secret,
        client_id=client_id
    )


def _upload_single_batch(
    project: LabellerrProject,
    annotation_file: str,
    annotation_format: str,
    batch_number: int,
    total_batches: int
) -> bool:
    """
    Upload a single batch annotation file.
    
    Args:
        project: LabellerrProject instance
        annotation_file: Path to the annotation file
        annotation_format: Format of the annotation file
        batch_number: Current batch number
        total_batches: Total number of batches
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\n[{batch_number}/{total_batches}] Uploading: {os.path.basename(annotation_file)}")
        
        # Execute synchronous upload
        result = project.upload_preannotations(
            annotation_format=annotation_format,
            annotation_file=annotation_file,
            _async=False  # Blocks until completion
        )
        
        # Validate completion status
        if result['response']['status'] == 'completed':
            metadata = result['response'].get('metadata', {})
            print(f"  ✓ Success! Metadata: {metadata}")
            return True
        else:
            print(f"  ✗ Upload completed with status: {result['response']['status']}")
            return False
            
    except LabellerrError as e:
        print(f"  ✗ Upload failed: {str(e)}")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error: {str(e)}")
        return False


def upload_preannotations(
    project_id: str,
    annotation_format: str,
    batch_annotation_dir: str,
    delay_between_uploads: int = 2,
    env_path: str = '.env'
) -> dict:
    """
    Upload all batch annotation files from a directory to a Labellerr project.
    
    This function can be imported and used in notebooks or other scripts.
    
    Args:
        project_id: Labellerr project ID to upload annotations to
        annotation_format: Format of annotation files (e.g., 'coco_json', 'yolo', 'pascal_voc')
        batch_annotation_dir: Directory containing annotation files to upload
        delay_between_uploads: Seconds to wait between uploads (default: 2)
        env_path: Path to .env file with credentials (default: '.env')
        
    Returns:
        Dictionary with upload statistics:
        {
            'total_batches': int,
            'successful_uploads': int,
            'failed_uploads': int,
            'failed_files': List[str],
            'elapsed_time': float
        }
        
    Example:
        >>> from batch_upload_preannot import upload_preannotations
        >>> result = upload_preannotations(
        ...     project_id='my_project_id',
        ...     annotation_format='coco_json',
        ...     batch_annotation_dir='output/annotations'
        ... )
        >>> print(f"Uploaded {result['successful_uploads']} files")
    """
    # Initialize client
    try:
        client = _get_client(env_path)
    except ValueError as e:
        print(f"Error: {str(e)}")
        return {}
    
    # Find all JSON files in the batch directory
    batch_path = Path(batch_annotation_dir)
    if not batch_path.exists():
        print(f"Error: Batch directory does not exist: {batch_annotation_dir}")
        return {}
    
    batch_files = sorted(list(batch_path.glob("*.json")))
    
    if not batch_files:
        print(f"No JSON files found in: {batch_annotation_dir}")
        return {}
    
    total_batches = len(batch_files)
    print(f"{'='*60}")
    print(f"Batch Upload Started")
    print(f"{'='*60}")
    print(f"Project ID: {project_id}")
    print(f"Annotation Format: {annotation_format}")
    print(f"Total batch files: {total_batches}")
    print(f"Delay between uploads: {delay_between_uploads}s")
    print(f"{'='*60}")
    
    # Get project instance
    try:
        project = LabellerrProject(client=client, project_id=project_id)
    except LabellerrError as e:
        print(f"Failed to initialize project: {str(e)}")
        return {}
    
    # Upload each batch
    successful_uploads = 0
    failed_uploads = 0
    failed_files = []
    
    start_time = time.time()
    
    for idx, batch_file in enumerate(batch_files, start=1):
        success = _upload_single_batch(
            project=project,
            annotation_file=str(batch_file),
            annotation_format=annotation_format,
            batch_number=idx,
            total_batches=total_batches
        )
        
        if success:
            successful_uploads += 1
        else:
            failed_uploads += 1
            failed_files.append(str(batch_file))
        
        # Wait between uploads (except for the last one)
        if idx < total_batches:
            time.sleep(delay_between_uploads)
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Batch Upload Complete!")
    print(f"{'='*60}")
    print(f"Total batches: {total_batches}")
    print(f"Successful uploads: {successful_uploads}")
    print(f"Failed uploads: {failed_uploads}")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"{'='*60}")
    
    if failed_files:
        print(f"\nFailed files:")
        for failed_file in failed_files:
            print(f"  - {failed_file}")
    
    return {
        'total_batches': total_batches,
        'successful_uploads': successful_uploads,
        'failed_uploads': failed_uploads,
        'failed_files': failed_files,
        'elapsed_time': elapsed_time
    }


def main():
    """Main function with configuration for standalone execution."""
    
    # ============================================================
    # CONFIGURATION - Modify these values as needed
    # ============================================================
    
    # Labellerr project ID
    PROJECT_ID = 'lynelle_additional_gibbon_85467'
    
    # Annotation format (coco_json, yolo, pascal_voc, etc.)
    ANNOTATION_FORMAT = 'coco_json'
    
    # Directory containing batch annotation files
    BATCH_DIR = r"output"
    
    # Delay between uploads in seconds (to avoid rate limiting)
    DELAY_BETWEEN_UPLOADS = 2
    
    # ============================================================
    # END CONFIGURATION
    # ============================================================
    
    upload_preannotations(
        project_id=PROJECT_ID,
        annotation_format=ANNOTATION_FORMAT,
        batch_annotation_dir=BATCH_DIR,
        delay_between_uploads=DELAY_BETWEEN_UPLOADS
    )


if __name__ == "__main__":
    main()

