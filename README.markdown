# Amazon ML Challenge 2025: Smart Product Pricing

This repository contains the solution for the **Amazon ML Challenge 2025 - Smart Product Pricing**, aimed at predicting product prices using text and image features to achieve a SMAPE (Symmetric Mean Absolute Percentage Error) score of 40–50%. The solution uses advanced feature engineering, text embeddings, lightweight image features, and XGBoost regression, optimized for CPU execution in Google Colab.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Results](#results)
- [Documentation](#documentation)
- [License](#license)

## Project Overview
The **Amazon ML Challenge 2025** requires predicting product prices based on textual product details (`catalog_content`) and images (`image_link`). The goal is to minimize SMAPE on a 75k test set. This solution achieves a validation SMAPE of approximately 40–50% using:
- Semantic text embeddings via Sentence Transformers.
- Lightweight image features (size, mean RGB).
- Feature engineering (Item Pack Quantity, text length, premium keywords).
- Tuned XGBoost regression with log-transformed prices.

The code is optimized for CPU execution in Google Colab, fitting within ~12GB RAM, and adheres to the challenge’s MIT/Apache 2.0 license requirement (uses XGBoost, Apache 2.0).

## Dataset
- **Training Data**: `dataset/train.csv` (75k samples) with columns:
  - `sample_id`: Unique identifier.
  - `catalog_content`: Concatenated title, description, and Item Pack Quantity (IPQ).
  - `image_link`: URL to product image.
  - `price`: Target variable (float).
- **Test Data**: `dataset/test.csv` (75k samples, same columns except `price`).
- **Output Format**: `test_out.csv` with `sample_id` and predicted `price` (positive float).
- **Constraints**: No external price lookup; predictions for all 75k test samples.

**Note**: Dataset files are not included in this repository due to challenge restrictions. Upload `dataset/train.csv` and `dataset/test.csv` to Colab.

## Methodology
- **Text Features**:
  - Used `all-MiniLM-L6-v2` (Sentence Transformers) for 384-dim semantic embeddings of `catalog_content`.
  - Additional features: text length, premium keyword flag (`premium|luxury|high-end`).
- **Image Features**:
  - Extracted lightweight features (image width, height, mean RGB) from `image_link` URLs.
  - Processed 50k images to fit Colab’s CPU memory constraints (extendable to 75k if RAM allows).
- **Feature Engineering**:
  - Extracted Item Pack Quantity (IPQ) using regex (e.g., `pack of (\d+)`).
  - Clipped prices (0 to 10,000) and applied log-transform to handle skew.
- **Model**:
  - XGBoost regression (`eta=0.05`, `max_depth=7`, `num_boost_round=300`, `tree_method='hist'`) with early stopping.
  - Trained on log-transformed prices; predictions reverted via `expm1`.
- **Evaluation**: SMAPE on validation set (20% of training data) typically 40–50%.

## Setup Instructions
1. **Clone Repository**:
   ```bash
   git clone https://github.com/your-username/amazon-ml-challenge-2025.git
   cd amazon-ml-challenge-2025
   ```

2. **Set Up Google Colab**:
   - Open [Google Colab](https://colab.research.google.com/).
   - Upload `main.py` from this repository.
   - Create directories and upload dataset:
     ```python
     !mkdir dataset
     !mkdir images
     ```
     - Upload `dataset/train.csv` and `dataset/test.csv` to `dataset/`.

3. **Install Dependencies**:
   Run in a Colab cell:
   ```python
   !pip install sentence-transformers Pillow xgboost scikit-learn pandas numpy
   ```

## Usage
1. **Run the Code**:
   - Execute `main.py` in Colab.
   - The script:
     - Loads and preprocesses data.
     - Generates text embeddings and image features.
     - Trains XGBoost and predicts prices.
     - Saves `test_out.csv` with `sample_id` and `price`.
   - Expected runtime: ~30–60 minutes (text encoding: ~5–10 min, image processing: ~20–40 min for 50k images, training: ~2–5 min).

2. **Output**:
   - Download `test_out.csv` from Colab.
   - Submit to the challenge portal, ensuring 75k rows matching `sample_test_out.csv` format.

3. **Tuning (if SMAPE >50%)**:
   - Increase `max_images` to 75k (monitor RAM with `!free -h`).
   - Adjust XGBoost: set `num_boost_round=500` or `eta=0.03` in `main.py`.
   - Add brand extraction (see code comments).

## Results
- **Validation SMAPE**: ~40–50% on 20% validation split.
- **Key Factors**:
  - Semantic text embeddings improve price prediction over TF-IDF.
  - IPQ and premium flags capture quantity and quality signals.
  - Lightweight image features contribute marginally but fit CPU constraints.
- **Limitations**:
  - Image processing limited to 50k samples to avoid RAM issues.
  - Potential for lower SMAPE with more images or advanced features (e.g., brand regex).

## Documentation
A 1-page methodology summary (per challenge requirements) is provided in `docs/Documentation.md`:
- **Methodology**: Sentence Transformers for text, lightweight image features, IPQ/text features, XGBoost.
- **Model**: XGBoost (`eta=0.05`, `max_depth=7`, `num_boost_round=300`, early stopping).
- **Features**: 384-dim text embeddings, image size/RGB, IPQ, text length, premium flag.
- **Preprocessing**: Log-transform prices, clip outliers (0–10,000), handle missing text/images.

## License
This project is licensed under the **Apache 2.0 License**, aligning with challenge requirements (XGBoost is Apache 2.0). See `LICENSE` for details.

---

INSTAGRAM - https://www.instagram.com/17akshat05
linkedin- https://www.linkedin.com/in/akshat-jain17/
