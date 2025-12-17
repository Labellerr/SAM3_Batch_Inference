# SAM3 Batch Inference

[![Labellerr](https://img.shields.io/badge/Labellerr-Website-green.svg)](https://www.labellerr.com/?utm_source=githubY&utm_medium=social&utm_campaign=github_clicks)
[![Labellerr Blog](https://img.shields.io/badge/Labellerr-BLOG-black.svg)](https://www.labellerr.com/blog)
[![Youtube](https://img.shields.io/badge/Labellerr-YouTube-b31b1b.svg)](https://www.youtube.com/@Labellerr)

Perform batch inference on images using **SAM3 (Segment Anything Model v3)** with text prompts and export annotations in **COCO-JSON format**. Seamlessly upload results to [Labellerr](https://www.labellerr.com) for review and refinement.

---

## 🌟 Features

- **Text-Prompt Based Segmentation**: Detect and segment objects using natural language prompts
- **Multi-Class Detection**: Supports multiple text prompts for detecting various object categories
- **COCO-JSON Output**: Export annotations in industry-standard COCO format
- **Labellerr Integration**: Upload pre-annotations directly to Labellerr projects for review
- **Memory Efficient**: Prompt-sequential processing strategy for efficient GPU memory usage
- **Batch Processing**: Process entire folders of images with progress tracking

---

## 📋 Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- Hugging Face account with SAM3 access

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Labellerr/SAM3_batch_inference.git
cd SAM3_batch_inference
```

### Step 2: Clone SAM3 Repository

```bash
git clone https://github.com/facebookresearch/sam3.git
```

### Step 3: Install SAM3 Package with Notebook Dependencies

```bash
cd sam3
pip install -e ".[notebooks]"
```

This will install:

- The `sam3` package in editable mode
- All required dependencies including `einops`, `opencv-python`, `pycocotools`, etc.

### Step 4: Install Other Requirements

```bash
cd ..
pip install -r requirements.txt
```

### Step 5: Install Labellerr SDK (Optional - for annotation upload)

```bash
pip install git+https://github.com/Labellerr/SDKPython.git
```

### Why Install SAM3 as a Package?

When you run scripts directly:

- Python adds the current directory to `sys.path`
- The `sam3` folder is accessible as a local module

When you import from a notebook:

- The notebook's working directory might be different
- Python needs the `sam3` package to be properly installed
- Installing with `pip install -e .` registers the package with Python

---

## 📥 Download SAM3 Model

SAM3 is developed by Meta and requires access permission.

1. Visit the official [SAM3 repository on Hugging Face](https://huggingface.co/facebook/sam3)
2. Request access (log in to Hugging Face and fill out the access form)
3. Once approved, download the model checkpoint (e.g., `sam3.pt`) to the `model/` folder

---

## 📖 Usage

### Using Jupyter Notebook

Open `Labellerr_SAM3_Batch_Inference.ipynb` and follow the step-by-step instructions.

### Using Python Script

```python
from sam3_batch_inference import run_batch_inference

# Configuration
INPUT_FOLDER = "flower_sample_img"
MODEL_CHECKPOINT = "model/sam3.pt"
TEXT_PROMPTS = [
    "Red Flower", 
    "Yellow Flower", 
    "White Flower",
    "Violet Flower"
]

# Run batch inference
run_batch_inference(
    INPUT_FOLDER,
    TEXT_PROMPTS,
    MODEL_CHECKPOINT,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `INPUT_FOLDER` | `str` | Path to folder containing input images |
| `TEXT_PROMPTS` | `List[str]` | List of text prompts for object detection |
| `MODEL_CHECKPOINT` | `str` | Path to SAM3 model checkpoint file (.pt) |
| `CONFIDENCE_THRESHOLD` | `float` | Detection confidence threshold (default: 0.4) |
| `OUTPUT_JSON` | `str` | Custom output path (default: `SAM3_Results/{input_folder}/`) |

---

## 📤 Review Annotations on Labellerr

The notebook provides a complete workflow to upload your SAM3 annotations to Labellerr for review and refinement.

### Step 1: Set Up Environment Variables

Create a `.env` file in the project root with your Labellerr credentials:

```env
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
CLIENT_ID = "your_client_id"
```

> **Note:** Get these credentials from the [Labellerr API tab](https://docs.labellerr.com/sdk/getting-started#getting-started) in your dashboard.
>
> **Important:** The notebook includes automatic validation to ensure all required environment variables (`CLIENT_ID`, `API_KEY`, `API_SECRET`) are properly loaded from the `.env` file. If any variable is missing or empty, you'll receive a clear error message indicating which credentials need to be configured.

### Step 2: Configure Workflow Parameters

Set up your dataset, template, and project names:

```python
# Path to SAM3 results folder
ANNOTATION_DIR = "SAM3_Results/sample_img"

# Labellerr resource names
DATASET_NAME = "My_Labellerr_Dataset"
TEMPLATE_NAME = "SDK SAM3 Batch Inference Template"
PROJECT_NAME = "SDK SAM3 Review Project"
```

### Step 3: Create Annotation Template

The notebook automatically creates an annotation template based on your `TEXT_PROMPTS`:

```python
from template_helper import create_questions_from_prompts
from labellerr.core.annotation_templates import create_template

# Creates annotation questions from your text prompts
QUESTIONS = create_questions_from_prompts(TEXT_PROMPTS)

template = create_template(
    client=CLIENT,
    params=CreateTemplateParams(
        template_name=TEMPLATE_NAME,
        data_type=DatasetDataType.image,
        questions=QUESTIONS
    )
)
```

### Step 4: Create Dataset from Local Images

Upload your images to Labellerr as a dataset:

```python
from labellerr.core.datasets import create_dataset_from_local

dataset = create_dataset_from_local(
    client=CLIENT,
    dataset_config=DatasetConfig(
        dataset_name=DATASET_NAME,
        data_type="image"
    ),
    folder_to_upload=INPUT_FOLDER,
)

# Wait for dataset processing
dataset.status()
print("Dataset ID:", dataset.dataset_id)
```

### Step 5: Create Labellerr Project

Create a project that combines your dataset and annotation template:

```python
from labellerr.core.projects import create_project

project = create_project(
    client=CLIENT,
    params=CreateProjectParams(
        project_name=PROJECT_NAME,
        data_type=DatasetDataType.image,
        rotations=RotationConfig(
            annotation_rotation_count=1,
            review_rotation_count=1,
            client_review_rotation_count=1
        )
    ),
    datasets=[dataset],
    annotation_template=template
)

print("Created project with ID:", project.project_id)
```

### Step 6: Upload SAM3 Pre-Annotations

Upload the SAM3 annotations to your Labellerr project:

```python
from batch_upload_preannot import upload_preannotations

result = upload_preannotations(
    project_id=project.project_id,
    annotation_format='coco_json',
    batch_annotation_dir=ANNOTATION_DIR
)
```

The upload process will display:

- Total batch files to upload
- Progress for each file
- Success/failure status
- Upload statistics (total batches, successful uploads, failed uploads, total time)

### Step 7: Review on Labellerr UI

Once pre-annotations are uploaded, visit the [Labellerr platform](https://www.labellerr.com) to:

- Review SAM3 annotations
- Refine segmentation masks
- Fix any detection errors
- Export corrected annotations

---

## 📁 Project Structure

```text
SAM3_batch_inference/
├── model/                          # Store SAM3 model checkpoint here
│   └── sam3.pt
├── sam3/                           # SAM3 repository (cloned)
├── flower_sample_img/              # Sample input images
├── SAM3_Results/                   # Output annotations
├── sam3_batch_inference.py         # Main inference script
├── batch_upload_preannot.py        # Labellerr upload script
├── template_helper.py              # Helper for creating annotation templates
├── Labellerr_SAM3_Batch_Inference.ipynb  # Jupyter notebook
├── requirements.txt                # Python dependencies
├── INSTALLATION_GUIDE.md           # Detailed installation guide
├── .env                            # Credentials (create this)
└── README.md                       # This file
```

---

## 📊 Output Format

Annotations are saved in COCO-JSON format:

```json
{
  "images": [
    {"id": 0, "file_name": "image.jpg", "width": 1920, "height": 1080}
  ],
  "categories": [
    {"id": 0, "name": "Red Flower"},
    {"id": 1, "name": "Yellow Flower"}
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 0,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "bbox": [x, y, width, height],
      "area": 1234.5
    }
  ]
}
```

---

## 🔧 Troubleshooting

### CUDA Out of Memory

- The script uses prompt-sequential processing to minimize memory usage
- For large images, the script automatically resizes them (default max: 1024px)
- Close other GPU applications before running

### Import Errors

- Ensure SAM3 is installed as a package: `pip install -e ".[notebooks]"`
- Verify you're in the correct virtual environment

### Model Not Found

- Ensure the model checkpoint is downloaded to the `model/` folder
- Verify the path in `MODEL_CHECKPOINT` is correct

### Environment Variable Validation Errors

If you see an error like `ValueError: Missing required environment variables: CLIENT_ID, API_KEY, API_SECRET`:

- Verify the `.env` file exists in the same directory as the notebook
- Check that variable names in `.env` match exactly: `CLIENT_ID`, `API_KEY`, `API_SECRET`
- Ensure there are no extra spaces around the `=` sign in the `.env` file
- Confirm the values are enclosed in quotes (e.g., `API_KEY = "your_key_here"`)
- Try restarting the Jupyter kernel after creating/modifying the `.env` file

---

## 📄 License

This project uses the SAM3 model which is subject to Meta's licensing terms. Please review the [SAM3 repository](https://github.com/facebookresearch/sam3) for licensing details.

---

## 🔗 Resources

- [Labellerr Platform](https://www.labellerr.com)
- [Labellerr Blog](https://www.labellerr.com/blog)
- [SAM3 on Hugging Face](https://huggingface.co/facebook/sam3)
- [SAM3 GitHub Repository](https://github.com/facebookresearch/sam3)
