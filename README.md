# VisionMetrics: Vision-Language Model for Product Metric Extraction and Text Transcription

## Project Overview

**VisionMetrics** is a machine learning solution designed to extract product metrics and transcribe textual content directly from images. This capability is crucial in industries such as healthcare, e-commerce, and content moderation, where precise information like weight, volume, dimensions, and other critical data are necessary for efficient product categorization, inventory management, and customer satisfaction. As digital marketplaces grow, products may lack detailed textual descriptions, making it essential to extract key details directly from images. VLMetrics leverages **Vision-Language Models (VLMs)** to address this challenge and automate both product information extraction and text transcription from images.

---

## Directory Structure

```bash
.
├── dataset
│   ├── sample_test_out.csv     # Sample output for test data
│   ├── test.csv                # Test data without ground truth
│   ├── train.csv               # Training data with labels
├── output
│   ├── test_out.csv            # Final output predictions file
├── src
│   ├── __init__.py             # Init file for src module
│   ├── bunny-llama-3-vision-language-model.py   # Vision-Language Model script
│   ├── constants.py            # Constants, such as allowed units for extraction
│   ├── image_processing.py     # Image preprocessing and augmentation script
│   ├── model.py                # Core model implementation
│   ├── sanity.py               # Sanity checker to ensure correct output format
│   ├── utils.py                # Utility functions (e.g., downloading images)
│   └── main.py                 # Main script to train and test the model
├── requirements.txt            # Required dependencies for the project
└── README.md                   # Project description and instructions
```

---

## Problem Statement

The goal of this project is twofold:
1. **Extract product metrics** from product images, such as weight, dimensions, volume, and other critical product details.
2. **Transcribe text** from images, ensuring that the extracted text matches the exact format and order of the original content.

These capabilities are crucial for industries like e-commerce, healthcare, and logistics, where accurate textual and metric extraction from images can significantly enhance automation workflows.

The model must predict the entity value in the format: `"x unit"` (e.g., "2 kilogram" or "3 centimetre") and transcribe text content with perfect accuracy, maintaining the format and order as shown in the image.

---

## Data Description

### Files:

- **train.csv**: Contains labeled training data with columns:
  - `index`: Unique identifier for the data sample.
  - `image_link`: Public URL of the product image.
  - `group_id`: Category code of the product.
  - `entity_name`: The entity being measured (e.g., item_weight, height).
  - `entity_value`: The actual value for the entity (e.g., "34 gram").

- **test.csv**: Contains test data without the entity value. Your task is to predict the entity value and transcribe text from images.

- **sample_test_out.csv**: A sample output file to demonstrate the expected format of predictions.

- **test_out.csv**: Your final predictions will be saved in this file, following the same format as `sample_test_out.csv`.

---

## Technical Approach

1. **Vision-Language Model (VLM)**:
   - The model used is **BAAI/Bunny-Llama-3-8B-V**, a state-of-the-art Vision-Language Model designed to process both visual and textual inputs.
   - **Quantization**: 4-bit quantization (NF4) is applied for memory-efficient inference on **CUDA-enabled devices**.
   - **Torch dtype**: The model operates using `torch.float16` for efficient tensor operations on GPU.

2. **Image Processing**:
   - Images are downloaded and processed to ensure they are ready for multimodal analysis.
   - The model processes image tensors directly, ensuring compatibility with the vision-language pipeline.
   - Images are preprocessed for noise reduction, contrast enhancement, and image cropping.

3. **Feature Extraction**:
   - **Text transcription**: The model is instructed via a system prompt to transcribe text exactly as it appears in the image, maintaining the order and format of the original content.
   - **Metric extraction**: The model identifies and extracts key product attributes, such as weight, volume, and dimensions, using Optical Character Recognition (OCR) techniques and deep learning.

4. **Entity Recognition and Formatting**:
   - The model formats the extracted values in the correct format: `"x unit"`.
   - Predicted entity values are validated against predefined mappings in the `constants.py` file to ensure correct units and formatting.

5. **Prediction Output**:
   - The predicted values and transcriptions are written to `test_out.csv` in the format: `index,prediction`, where the prediction is either the entity value or transcribed text.

---

## Model Training and Inference

The model is trained using supervised learning on the labeled `train.csv` dataset. The training process involves feature extraction from images and subsequent classification of extracted features into the correct entity categories. Additionally, the model is trained to transcribe text accurately.

### Running Inference:

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <https://github.com/Callmeamann/VisionMetrics-VLM_based_Product_Metric_Extraction.git>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Preprocess the images and train the model:
   ```bash
   python src/main.py
   ```

4. Run inference on test images and generate predictions:
   ```bash
   python src/main.py --test
   ```

5. Validate the output using the sanity checker:
   ```bash
   python src/sanity.py --output output/test_out.csv
   ```

---

## Evaluation Criteria

The project is evaluated based on two key metrics:
1. **F1 Score for Product Metrics Extraction**:
   - Measures the accuracy of the entity value extraction and formatting.
   - True positives (TP), false positives (FP), and false negatives (FN) are used to calculate the F1 score.

2. **Text Transcription Accuracy**:
   - The model is evaluated on how accurately it transcribes the text from images, maintaining the correct order and format.

---

## Appendix

### Allowed Units for Extraction

The `constants.py` file contains a predefined set of allowed units. Ensure that your predictions only include these units:

```python
entity_unit_map = {
  "width": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
  "depth": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
  "height": {"centimetre", "foot", "millimetre", "metre", "inch", "yard"},
  "item_weight": {"milligram", "kilogram", "microgram", "gram", "ounce", "ton", "pound"},
  "maximum_weight_recommendation": {"milligram", "kilogram", "microgram", "gram", "ounce", "ton", "pound"},
  "voltage": {"millivolt", "kilovolt", "volt"},
  "wattage": {"kilowatt", "watt"},
  "item_volume": {"microlitre", "litre", "millilitre", "gallon", "cubic inch", "cup", "fluid ounce"}
}
```

---

## Authors

- **Aman Gusain**
