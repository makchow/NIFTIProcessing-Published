# NIFTIprocessing

**NIFTIprocessing** is a modular Python toolkit designed for preprocessing NIFTI medical images and preparing them for machine learning workflows. It includes image registration, segmentation, patient data abstraction, radiomics feature extraction, and a training pipeline.

---

## 📦 Overview

This repository provides a complete pipeline to:

- Organize and preprocess raw NIFTI patient scans
- Register and segment brain images using an atlas
- Extract radiomics features from whole CT and segmented regions
- Perform robust left-right region feature comparison
- Structure data for downstream ML processing
- Train ML models using structured imaging datasets

---

## 🧪 NIFTIprocessing.py

`NIFTIprocessing.py` implements the preprocessing pipeline for raw NCCT images of the MMH cohort. It includes conversion, cleaning, registration to atlas space, and anatomical segmentation using a Talairach-based atlas.

---

### 🧠 Overview

The script performs:

1. **NCCT Preparation**

   * Conversion from DICOM (optional)
   * FOV cropping
   * Skull stripping (FSL BET)

2. **Atlas Registration**

   * Rigid registration to `talairach.nii` via FSL FLIRT
   * Applies transformation and saves `SUBJID_reg.nii.gz`

3. **Atlas-Based Segmentation**

   * Maps Talairach labels from `tlabels.csv`
   * Extracts and saves anatomical region masks and their CT segmentations
   * Outputs `SUBJID_tlabels_regions.txt` for downstream reference

---

### 📁 Folder Structure

The default output structure for each patient is:

```
processedNIFTIs/SUBJ_ID/
├── SUBJID_reg.nii.gz                  # Skull stripped, registered NCCT
├── SUBJID_tlabels_regions.txt         # List of saved regions
├── registration/                      # Intermediate FSL registration outputs
├── masks/                             # Binary masks per anatomical region
│   ├── SUBJID_atlas.nii.gz
│   └── SUBJID_{region}_mask.nii.gz
└── segmentation_output/              # Segmented region CT volumes
    └── SUBJID_{region}_region.nii.gz
```

---

### ⚙️ Configuration

The script expects this structure under `BASE_FOLDER` to process images (OS-dependent):

* `preNIFTIs/` — Raw NCCT files as `.nii.gz`, named by subject ID
* `mmhcohort.csv` — Cohort file with a `subj_id` column
* `talairach.nii` — Atlas image (in NIfTI format)
* `tlabels.csv` — CSV with `index,label` mapping for atlas regions

You can change the OS-specific paths directly inside the `main()` function.

---

### 🧼 Utility Functions

* `dcm_nii_convert(folder)`
  Converts DICOMs in a folder to compressed NIfTI using `dcm2niix`.

* `clean_mask(mask, min_size=50)`
  Removes small components in a binary mask using connected-component analysis.

---

### 🚀 Usage

Run the full pipeline with:

```bash
python NIFTIprocessing.py
```

The script will:

* Detect your platform and set paths
* Load patient IDs from `mmhcohort.csv`
* Register and segment each subject
* Save outputs to `processedNIFTIs/`

---

## NIFTIPatient.py

`NIFTIPatient.py` is the core module for patient-level data abstraction and feature engineering in the NIFTIprocessing pipeline. It provides classes and logic to:

- Load and organize patient CT data and segmented brain regions
- Extract radiomics and statistical features from both the whole scan and each anatomical region
- Pair left/right brain regions for robust comparative analysis
- Merge all features into a flat, ML-ready dictionary
- Provide static evaluation utilities for patient and region assessment

This module is designed to be the main interface for downstream ML workflows, enabling both batch and single-patient processing.

---

### 🧬 Core Data Classes

The `PatientNIFTI` and `RegionNIFTI` classes are the core abstractions for representing and processing patient imaging data in the NIFTIprocessing pipeline.

#### `PatientNIFTI`
Represents a complete patient brain imaging study with full CT data and segmented anatomical regions.

**Constructor:**
```python
PatientNIFTI(subj_id: str, patient_folder: str, metadata=None)
```

**Key Attributes:**
- `subj_id`: Subject/patient identifier
- `patient_folder`: Path to processed patient data folder
- `metadata`: Dictionary of patient metadata (from cohort CSV)
- `full_ct_data`: Numpy array of full CT image voxels
- `atlas_data`: Numpy array of brain atlas for the patient
- `regions`: Dictionary of RegionNIFTI objects (key: region name, value: RegionNIFTI object)
- `left_right_pairs`: Dictionary of paired left/right regions
- `features`: Flattened dictionary containing all extracted features and metadata

**Key Capabilities:**
- Automatically instantiates a `RegionNIFTI` object for each anatomical region
- Extracts and merges features from the whole CT, each region, and left-right region comparisons into a single flat dictionary (`.features`) suitable for ML workflows
- Provides methods to pair left/right regions and merge region-level features into the patient-level feature set

#### `RegionNIFTI`
Represents a single segmented anatomical brain region (e.g., left hippocampus, right amygdala).

**Constructor:**
```python
RegionNIFTI(parent_NIFTI: PatientNIFTI, region_name: str, 
            region_mask_fname: str, region_ct_fname: str)
```

**Key Attributes:**
- `region_name`: Name of the anatomical region
- `region_ct_data`: Numpy array of region-specific CT data
- `region_mask_data`: Numpy array of binary mask for the region
- `hemisphere`: "left", "right", or None
- `base_region`: Region name without hemisphere prefix (e.g., "hippocampus")
- `features`: Dictionary of region-specific features
- `voxel_volume`: Volume of a single voxel (mm³)

**Key Capabilities:**
- Loads region CT and mask from NIfTI files
- Determines hemisphere and base region name for robust pairing and comparison
- Computes region-specific features (e.g., radiomics, volume, HU statistics)
- Stores all computed features with keys prefixed by the region name for clarity

### Features Extracted:

After running the `Evaluator` (either directly or via the batch processing utilities), all features are automatically extracted and merged into the patient object:

**Whole CT (PatientNIFTI):**
- Radiomics features extracted from the full registered CT using PyRadiomics
- Metadata from the cohort CSV (e.g., age, sex, clinical variables)

**Per Region (RegionNIFTI):**
- Radiomics features extracted from each segmented anatomical region
- Region-specific statistics (e.g., volume, HU statistics)

**Left-Right Comparison (PatientNIFTI):**
- For each base region, computes the difference and ratio of features between left and right hemispheres
- Example: `frontallobe_Mean_RL_diff`, `hippocampus_volume_RL_ratio`

**Radiomics Feature Categories:**
PyRadiomics extracts the following feature types from both whole CT and region masks:
- **First Order Statistics**: mean, standard deviation, min, max, median, energy, entropy
- **Shape-based (3D)**: volume, sphericity, compactness, surface area, elongation
- **Gray Level Co-occurrence Matrix (GLCM)**: contrast, dissimilarity, homogeneity, correlation, energy
- **Gray Level Run Length Matrix (GLRLM)**: short run emphasis, long run emphasis, run length nonuniformity
- **Gray Level Size Zone Matrix (GLSZM)**: small area emphasis, large area emphasis, size zone nonuniformity
- **Neighboring Gray Tone Difference Matrix (NGTDM)**: coarseness, contrast, busyness
- **Gray Level Dependence Matrix (GLDM)**: small dependence emphasis, large dependence emphasis

**Laterality Indices (for paired left/right regions):**
- `RL_diff`: Right - Left difference
- `RL_ratio`: Right / Left ratio
- `RL_absdiff`: Absolute difference

All of these features are merged into the `.features` dictionary of the `PatientNIFTI` object, making them immediately available for downstream analysis or machine learning. You can access them as follows:


### Example:

```python
import NIFTIpatient

# Load a single patient (with optional metadata)
patient = PatientNIFTI(subj_id="MMH001", patient_folder="./processedNIFTIs/MMH001", metadata=meta)

# Generate evaluations for full CT and it's segmented regions
Evaluator.evaluate(patient)

# Access the full feature dictionary (ready for ML)
feature_dict = patient.get_features()

# Access a specific region's features
region = patient.get_region("leftcerebrum_frontallobe")
region_features = region.get_features()
```

---

### 📈 Evaluator

The `Evaluator` class is a static utility providing methods to compute radiomics features and regional comparisons.

**Key Static Methods:**

| Method | Purpose |
|--------|---------|
| `calc_radiomics_features(nifti_obj, radiomics_features=None)` | Extracts radiomics features using PyRadiomics library |
| `evaluate_patient(patient)` | Computes patient-level radiomics features |
| `evaluate_region(region)` | Computes region-level radiomics features |
| `compare_left_right_regions(patient, feature_names=None)` | Computes laterality indices (L-R difference, ratio, absolute difference) |
| `evaluate(patient)` | Runs all evaluations and aggregates results |

**Feature Preprocessing:**
All evaluations automatically:
- Exclude features with `'diagnostics'` in the key
- Strip the `original_` prefix from PyRadiomics feature names
- Support filtering to specific features via optional parameters
- Merge results into the main `PatientNIFTI.features` dictionary for model use

---

#### **Radiomics Feature Extraction**

The pipeline uses **PyRadiomics** to extract quantitative features from both the whole CT and each segmented region.

**Single Patient Processing:**
```python
from NIFTIpatient import Evaluator

# Extract all available radiomics features
features = Evaluator.calc_radiomics_features(patient)

# Extract only selected features
selected = ["original_firstorder_Mean", "original_shape_VoxelVolume"]
features = Evaluator.calc_radiomics_features(patient, radiomics_features=selected)
```

**Feature Naming Convention:**
- Whole CT features are prefixed with the CT identifier
- Region features are prefixed with the region name (e.g., `leftcerebrum_frontallobe_firstorder_Mean`)
- Left-right comparisons are suffixed with `_RL_diff` or `_RL_ratio`

For complete radiomics feature documentation, see the [PyRadiomics documentation](https://pyradiomics.readthedocs.io/en/latest/features.html) or the `radiomics_docs/` folder in this repository.

**Example:**

```python
from NIFTIpatient import Evaluator

# Extract all features (default)
features = Evaluator.calc_radiomics_features(patient)
# Example: 'original_firstorder_Mean' from PyRadiomics becomes 'full_ct_firstorder_Mean' in the patient feature dict

# Extract only selected features
selected = ["original_firstorder_Mean", "original_shape_VoxelVolume"]
features = Evaluator.calc_radiomics_features(patient, radiomics_features=selected)
# After adding to the patient, keys will be 'firstorder_Mean', 'shape_VoxelVolume', etc.

# Extract region features
region = patient.get_region("leftcerebrum_frontallobe")
region_features = Evaluator.calc_radiomics_features(region, radiomics_features=selected)
# When added to the region, keys are prefixed: e.g., 'leftcerebrum_frontallobe_firstorder_Mean'

# Access the region's feature dictionary
region_features_dict = region.get_features()
# Example key: 'leftcerebrum_frontallobe_firstorder_Mean'

# Comparing left/right regions
Evaluator.compare_left_right_regions(patient, feature_names=selected)
# When added to the patient, keys are suffixed: e.g., 'frontallobe_Mean_RL_diff', 'frontallobe_VoxelVolume_RL_ratio'
# Where 'frontallobe' is the base region name (without 'left' or 'right') and '_RL_diff'/'_RL_ratio' indicate the comparison type
```

---

#### **Left-Right Region Pairing**

The toolkit robustly pairs left/right brain regions using `base_region` and `hemisphere` attributes:

**Automatic Pairing:**
- Regions are paired based on their base name (e.g., `left_hippocampus` paired with `right_hippocampus`)
- Supports flexible region naming conventions with left/right/center designations

**Laterality Computation:**
- **Difference** (Right - Left): Stored as `{region}_feature_RL_diff`
- **Ratio** (Right / Left): Stored as `{region}_feature_RL_ratio`
- Restricted to features present in both left and right regions
- Optionally filter to specific features via `feature_names` parameter

#### Example:

```python
from NIFTIpatient import Evaluator
# Compare left/right regions for all features
Evaluator.compare_left_right_regions(patient)
# Compare only selected features
Evaluator.compare_left_right_regions(patient, feature_names=["Mean", "VoxelVolume"])
```

---

### 🗃️ Batch Processing

`NIFTIPatient.py` includes utility functions to streamline batch processing of multiple patients. These functions handle saving/loading patient features to/from CSV files, enabling easy integration with machine learning workflows.
- `generate_all_patients(base_folder, cohort_csv, processed_folder)`
  - Loads all patients from a cohort CSV and their corresponding processed folders, returning a list of `PatientNIFTI` objects.
  - Each patient object is automatically evaluated and ready for ML use.
  - Returns a list of `PatientNIFTI` objects.
- `save_all_patients(patient_list, csv_fname)`
  - Saves a list of `PatientNIFTI` objects to a flat CSV file using each patient’s `.get_feature_vector()` output.
  - Ensures the output directory exists before saving.

- `load_all_patients(csv_fname, exclude_pred, ground_truth)`
  - Loads patient features from a CSV and splits them into predictors and ground truth columns.
  - Excludes columns such as IDs or metadata (`exclude_pred`)
  - Extracts target labels for ML training (`ground_truth`)
  - Returns a tuple: `(X: predictors, y: labels)`

Each `PatientNIFTI` object is evaluated, saved, and easily reloadable for downstream ML workflows.

#### Example:

```python
from NIFTIpatient import generate_all_patients, save_all_patients, load_all_patients
patients = generate_all_patients(base_folder="./", cohort_csv="mmhcohort.csv", processed_folder="processedNIFTIs/")
save_all_patients(patients, csv_fname="all_patient_features.csv")
X, y = load_all_patients("all_patient_features.csv", exclude_pred=["subj_id", "age"], ground_truth="outcome")
```

---

### 🔑 Key Dependencies

- **nibabel**: NIfTI file I/O and manipulation
- **numpy**: Numerical computations and array operations
- **pandas**: Data manipulation and CSV handling
- **radiomics**: PyRadiomics feature extraction from medical images
- **tqdm**: Progress bars for batch processing

---

### ⚠️ Important Notes

**Region Naming Convention:**
- Regions should be prefixed with "left_" or "right_" to enable automatic pairing
- Example: `left_hippocampus`, `right_hippocampus`

**Data Format:**
- Atlas and region masks must be in NIfTI format (.nii.gz)
- Patient metadata is loaded from cohort CSV with matching subject IDs
- Non-numeric radiomics outputs are silently skipped

**Batch Processing:**
- All patients are processed sequentially with progress tracking
- Failed patients are logged but don't halt the pipeline
- Missing regions for a patient are handled gracefully

---

### 🚀 CLI Entrypoint

Run the `main()` function to:

* Load platform-specific paths
* Validate file structure
* Load and evaluate all patients
* Print patient features to stdout and save to CSV

---

## NIFTItraining.py

`NIFTItraining.py` is a comprehensive machine learning training and evaluation pipeline for neuroimaging radiomics features. It handles model selection, hyperparameter tuning, performance evaluation, and multi-seed robustness assessment.

---

### 🎯 Core ML Pipelines

The module provides specialized pipeline implementations for multiple algorithms, all inheriting from a base class that handles preprocessing and metrics computation.

#### **BaseMLPipeline**
Base class for all ML pipeline implementations.

**Constructor:**
```python
BaseMLPipeline(n_features=None, estimator=None, random_state=None, ord_categories=None)
```

**Preprocessing Pipeline:**
1. **Imputation**: Median imputation for missing values
2. **Scaling**: StandardScaler normalization
3. **Variance Thresholding**: Removes features with variance < 1%
4. **Feature Selection**: SelectKBest with f_classif scoring
5. **Classifier**: Model-specific estimator

**Key Methods:**
- `search_fit()`: Performs hyperparameter tuning (GridSearchCV or RandomizedSearchCV)
- `get_preprocessed_feature_names()`: Extracts feature names after preprocessing
- `save()`: Saves trained model and comprehensive metrics report

**Metrics Computed:**
- Accuracy, Precision, Recall, Specificity, F1 Score
- AUC (Area Under Curve)
- Confusion Matrix
- Classification Report (per-class metrics)

---

#### **Specialized Pipelines**

The module includes optimized implementations for:

**RandomForestMLPipeline**
- Tree-based feature importance extraction
- Handles non-linear relationships
- Parallel execution support

**LogisticRegressionMLPipeline**
- Linear model with interpretable coefficients
- Fast training for preliminary analysis
- Coefficient-based feature importance

**XGBoostMLPipeline**
- Gradient boosting for improved accuracy
- Better handling of imbalanced classes
- Feature importance from tree structure

**XGBoostRFMLPipeline**
- Combines XGBoost with random forest base learners
- Balances bias and variance
- Subsample parameter for regularization

Each pipeline has a default hyperparameter grid that can be customized via the `param_grid` parameter.

---

### 📊 Nested Cross-Validation

The training pipeline implements **nested cross-validation** for unbiased model evaluation:

**Outer Loop:**
- Stratified K-fold splitting (default: 5 folds)
- Evaluates generalization to unseen data
- Prevents data leakage to hyperparameter tuning

**Inner Loop:**
- Hyperparameter search on training folds
- GridSearchCV (exhaustive, thorough) or RandomizedSearchCV (fast, sampling-based)
- Returns best hyperparameters for each outer fold

**Returns:**
- Test scores for each outer fold
- Best hyperparameters found at each fold
- ROC curve data (FPR, TPR, AUC) for visualization

---

### 📈 Multi-Seed Evaluation

For robust and reproducible results, the pipeline supports running experiments across multiple random seeds:

**Key Functions:**

**`nested_cv_evaluate(pipeline, X, y, outer_cv=3, inner_cv=3, search_type="random")`**
- Performs nested CV for a single pipeline
- Returns test scores, best parameters, and ROC data
- Supports flexible CV fold configuration

**`aggregate_seed_metrics(aggregate_records, ml_dir)`**
- Aggregates performance metrics across multiple seeds
- Computes mean, standard deviation, and 95% confidence intervals
- Outputs raw metrics, summary statistics, and human-readable reports

**`aggregate_feature_usage(feature_usage, ml_dir, top_n=None)`**
- Aggregates feature importance/selection across seeds
- Ranks features by consistency across runs
- Identifies reproducible, stable features
- Generates per-model summaries

---

### ⚙️ Feature Preprocessing & Selection

**Automated Preprocessing:**
1. **Missing Value Handling**: Median imputation (handles -1 sentinel values)
2. **Scaling**: StandardScaler for consistent feature scales across models
3. **Variance Filtering**: Removes constant/near-constant features
4. **Feature Selection**: SelectKBest with f_classif (tuned via hyperparameters)

**Hyperparameter Tuning Options:**
- **GridSearchCV**: Exhaustive search over specified parameter grid (slower, thorough)
- **RandomizedSearchCV**: Random sampling from distributions (faster, good for large grids)
- **Stratified K-Fold**: Maintains class distribution in CV splits

---

### 📁 Output Structure

Results are organized by model type and random seed:

```
trainingML/
├── patient_features.csv                    # Input feature matrix
├── aggregate_seed_metrics_raw.csv          # Per-seed metrics
├── aggregate_seed_metrics_summary.csv      # Summary with 95% CI
├── aggregate_seed_metrics_summary.txt      # Human-readable summary
├── feature_usage_across_models.txt         # Feature importance aggregation
│
├── rf/                                     # Random Forest results
│   ├── seed_1292/
│   │   ├── rf_seed_1292_model.pkl
│   │   └── rf_seed_1292_metrics.txt
│   └── feature_usage_summary.csv
│
├── lr/                                     # Logistic Regression results
├── xgb/                                    # XGBoost results
└── xgbrf/                                  # XGBoost RF results
```

---

### 🔑 Key Features & Best Practices

**Nested Cross-Validation**
- Provides unbiased estimate of model performance
- Outer loop evaluates generalization; inner loop tunes hyperparameters
- Prevents overfitting to validation set during tuning

**Feature Preprocessing**
- Median imputation handles missing data robustly
- StandardScaler required for Logistic Regression
- Variance filtering removes uninformative features
- Feature selection (k-best) is tuned during hyperparameter search

**ROC Curves**
- Generated for each outer CV fold
- Aggregated across seeds for model comparison
- AUC reported as single scalar metric

**Feature Importance Aggregation**
- Tree models: Extracted from feature_importances_
- Linear models: Absolute values of coefficients
- Across seeds: Ranked by frequency and consistency
- Identifies reproducible, generalizable features

---

### 📋 Configuration Parameters

**Input/Output:**
- `BASE_FOLDER`: Project root
- `FEATURES_CSV`: Path to input feature matrix
- `ML_DIR`: Output directory for results

**Cross-Validation:**
- `OUTER_CV`: Number of outer loop folds (default: 5)
- `INNER_CV`: Number of inner loop folds (default: 5)
- `search_type`: "grid" or "random" for inner loop search

**Reproducibility:**
- `SEEDS`: List of random states for multi-seed evaluation
- `random_state`: Set in each pipeline for deterministic results

**Data Filtering:**
- `EXCLUDE_COL`: Columns to exclude (e.g., IDs, metadata)
- `EXCLUDE_COL_MATCH`: Partial strings to exclude columns

**Model-Specific Grids:**
- Customize hyperparameter ranges per model type
- Default grids provided for each pipeline

---

### 🔗 Integration with NIFTIpatient

Feature loading from the NIFTIpatient pipeline:

```python
from NIFTItraining import RandomForestMLPipeline, nested_cv_evaluate
import NIFTIpatient

# Load features prepared by NIFTIpatient
X, y = NIFTIpatient.load_all_patients(
    csv_fname="trainingML/patient_features.csv",
    exclude_col=["subj_id", "mrn"],
    ground_truth=["pts"]
)

# Replace sentinel values
X = X.replace(-1, np.nan)
y = y.values.ravel()

# Run nested CV
pipeline = RandomForestMLPipeline(random_state=42)
test_scores, best_params, roc_data = nested_cv_evaluate(
    pipeline, X, y, outer_cv=5, inner_cv=5
)
```

---

### 📦 Dependencies

- **scikit-learn**: ML pipeline, cross-validation, metrics, feature selection
- **xgboost**: Gradient boosting models
- **pandas**: Data manipulation and result aggregation
- **numpy**: Numerical computations
- **matplotlib**: ROC curve visualization
- **NIFTIpatient**: Feature loading and preprocessing

---

### ⚠️ Key Considerations

**Binary Classification:**
- Optimized for binary classification (e.g., PTS labels)
- Metrics include class-specific measures (sensitivity, specificity)

**Class Imbalance:**
- Not explicitly handled with class weights (can be added if needed)
- Stratified CV maintains class distribution

**Hyperparameter Grids:**
- Default grids provided; customize for your data
- GridSearchCV exhaustive; RandomizedSearchCV faster
- Larger grids increase training time quadratically

**Feature Scaling:**
- Required for Logistic Regression and distance-based metrics
- Already handled in pipeline
- Important for interpreting coefficients

**Reproducibility:**
- Set `random_state` consistently across runs
- Multiple seeds recommended for robust conclusions
- Document all configuration parameters

---

### 🚀 CLI Entrypoint

Run the complete multi-seed pipeline:

```bash
python NIFTItraining.py
```

---