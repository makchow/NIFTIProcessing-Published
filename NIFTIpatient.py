import nibabel as nib
import numpy as np
import os
from glob import glob
import pandas as pd
import radiomics.featureextractor
import logging
import traceback
from tqdm import tqdm
import random


class PatientNIFTI:
    def __init__(self, subj_id: str, patient_folder: str, metadata=None):
        """
        Represents a full patient NIFTI image and its segmented anatomical regions.

        Args:
            subj_id (str): Subject ID.
            patient_folder (str): Path to the processed patient folder which expects folders (masks, segmentation_output) and subj_reg.nii
            metadata (dict): Optional metadata for the patient (from cohort CSV).
            
        Returns:
            None
        """
        self.subj_id = subj_id
        self.patient_folder = patient_folder
        self.metadata = metadata if metadata else {}

        # CT image attributes
        self.full_ct_data = None
        self.atlas_data = None
        self.clean_flat_ct_data = None
        self.region_labels = None
        self.affine = None
        self.header = None
        self.voxel_volume = None

        # Region dict
        self.regions = {}  # key: region name, value: RegionNIFTI object

        # Dict of tuple (R,L) pairs of RegionNIFTI objects
        # key: region base name without left_/right_ prefix, value: (left_region_obj, right_region_obj)
        self.left_right_pairs = {}

        # All features flattened to one dict => main input for ML models
        # Includes metadata, full ct and region ct feats + evals
        self.features = {}

        # Calculated Attributes for Full CT (Will be added to features dict)
        # --- Add more as needed ---

        # Initialize by loading data and extracting features
        self.load_ct()
        # self.apply_mask()
        self.extract_voxel_volume()
        self.import_regions()
        self.extract_all_features()
        self.build_left_right_pairs()

    def load_ct(self):
        """
        Load the full CT image from NIfTI file and store its data, affine, and header.
        
        Args:
            None
            
        Returns:
            None
        """
        try:
            # Import CT
            ct_img_fname = os.path.join(
                self.patient_folder, f"{self.subj_id}_reg.nii.gz"
            )
            ct_img = nib.load(ct_img_fname)
            self.full_ct_data = ct_img.get_fdata()
            self.affine = ct_img.affine
            self.header = ct_img.header

            # Import the atlas
            atlas_fname = os.path.join(
                self.patient_folder, "masks", f"{self.subj_id}_atlas.nii.gz"
            )
            atlas_img = nib.load(atlas_fname)
            self.atlas_data = atlas_img.get_fdata()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load CT image from {self.patient_folder}: {e}"
            )

    def apply_mask(self):
        """
        Applies mask onto the CT and returns the numpy array to self.clean_flat_ct_data
        
        Args:
            None
            
        Returns:
            None
        """
        masked_data = np.where(self.atlas_data > 0, self.full_ct_data, 0)
        self.clean_flat_ct_data = masked_data[masked_data > 0]

    def extract_voxel_volume(self):
        """
        Compute the volume of a single voxel (in mm³) from the CT header's zooms.
        
        Args:
            None
            
        Returns:
            None
        """
        self.voxel_volume = np.prod(self.header.get_zooms())

    def extract_all_features(self):
        """
        Add metadata and calculate full CT features
        For each segmented region:
            - Create a RegionNIFTI object.
            - Extract region-level features.
            - Add the features to the global dictionary.
            
        Args:
            None
            
        Returns:
            None
        """
        # Add metadata to features dict
        self.features.update(self.metadata)

        # Add calculated attributes here (if any)

        self.merge_region_features()

    def import_regions(self):
        """
        Instantiate and imports RegionNIFTI objects according the provided labels.
        Creates a dictionary in PatientNIFTI => key: region_name, value: RegionNIFTI
        
        Args:
            None
            
        Returns:
            None
        """

        def read_region_names(region_labels_fname):
            '''
            Returns a cleaned list of region names
            
            Args:
                region_labels_fname (str): Path to the region labels text file.
                
            Returns:
                region_labels (list): List of cleaned region names.
            '''
            with open(region_labels_fname, "r") as f:
                region_labels = [line.strip().lower().replace(".", "_") for line in f]
            return region_labels

        # Import region names
        region_labels_fname = os.path.join(
            self.patient_folder, f"{self.subj_id}_tlabels_regions.txt"
        )
        self.region_labels = read_region_names(region_labels_fname)

        # Add region features
        for region in self.region_labels:
            region_mask_fname = os.path.join(
                self.patient_folder, "masks", f"{self.subj_id}_{region}_mask.nii.gz"
            )
            region_ct_fname = os.path.join(
                self.patient_folder,
                "segmentation_output",
                f"{self.subj_id}_{region}_region.nii.gz",
            )
            region_obj = RegionNIFTI(
                parent_NIFTI=self,
                region_name=region,
                region_mask_fname=region_mask_fname,
                region_ct_fname=region_ct_fname,
            )
            self.regions[region] = region_obj
            self.features.update(region_obj.get_features())

    def build_left_right_pairs(self):
        """
        Populates self.left_right_pairs with keys as base region names (without left_/right_ prefix)
        and values as (left_region_obj, right_region_obj) tuples, only if both exist.
        
        Args:
            None
            
        Returns:
            None
        """
        self.left_right_pairs = {}
        # Collect left and right regions by base_region
        left_regions = [r for r in self.regions.values() if r.hemisphere == "left"]
        right_regions = [r for r in self.regions.values() if r.hemisphere == "right"]
        # Build a dict for quick lookup
        left_dict = {r.base_region: r for r in left_regions}
        right_dict = {r.base_region: r for r in right_regions}
        # Pair only if both left and right exist for the same base_region
        for base_region in left_dict:
            if base_region in right_dict:
                self.left_right_pairs[base_region] = (
                    left_dict[base_region],
                    right_dict[base_region],
                )

    def add_eval(self, key, value, prefix="full_ct_"):
        """
        Adds a key, value evaluation to the feature dict
        
        Args:
            key (str): Feature name.
            value (any): Feature value.
            prefix (str): Prefix to add to the key (default: 'full_ct_').
            
        Returns:
            None
        """
        self.features[f"{prefix}{key}"] = value

    def merge_region_features(self):
        """
        Update the patient feature dict after evaluations are added to each region.
        Does NOT overwrite existing keys.
        
        Args:
            None
            
        Returns:
            None
        """
        for region_obj in self.regions.values():
            for k, v in region_obj.get_features().items():
                if k not in self.features:
                    self.features[k] = v

    def get_left_right_pairs(self):
        """
        Returns the dictionary of left-right region pairs.
        
        Args:
            None
            
        Returns:
            None
        """
        return self.left_right_pairs

    def get_base_region_names(self):
        """
        Returns the list of base region names without left/right prefix.
        
        Args:
            None
            
        Returns:
            None
        """
        return self.left_right_pairs.keys()

    def get_region(self, region_name):
        """
        Returns the RegionNIFTI object for the given region name.
        
        Args:
            region_name (str): Name of the region to retrieve.
            
        Returns:
            RegionNIFTI or None: The RegionNIFTI object if found, else None.
        """
        return self.regions.get(region_name, None)

    def get_features(self):
        """
        Get the features for this patient.
        Combines region features with metadata and patient ID.
        
        Args:
            None
            
        Returns:
            dict: Flattened features dictionary.
        """
        return self.features.copy()


class RegionNIFTI:
    def __init__(
        self,
        parent_NIFTI: PatientNIFTI,
        region_name: str,
        region_mask_fname: str,
        region_ct_fname: str,
    ):
        """
        Represents a segmented anatomical region of a CT image.

        Args:
            parent_NIFTI (PatientNIFTI): The parent patient object.
            region_name (str): The name of the anatomical region.
            region_mask_path (str): File path to the binary mask NIfTI file for this region.
            region_ct_path (str): File path to the NIfTI file for this region.
            
        Returns:
            None
        """
        self.subj_id = parent_NIFTI.subj_id
        self.region_name = region_name
        self.region_mask_fname = region_mask_fname
        self.region_ct_fname = region_ct_fname

        # CT image data
        self.region_ct_data = None
        self.region_mask_data = None
        self.clean_flat_region_ct_data = None
        self.affine = None
        self.header = None
        self.voxel_volume = None

        # Location attributes
        self.base_region = None
        self.hemisphere = None

        # Features
        self.features = {}

        # Calculated Attributes for this region (Will be added to features dict)
        # --- Add more as needed ---

        self.load_ct()
        # self.apply_mask()
        self.extract_voxel_volume()
        self.determine_location()
        self.extract_region_features()

    def load_ct(self):
        """
        Load the region NIfTI file and store its data, affine, and header.
        
        Args:
            None
            
        Returns:
            None
        """
        try:
            # Import CT
            ct_img = nib.load(self.region_ct_fname)
            self.region_ct_data = ct_img.get_fdata()
            self.affine = ct_img.affine
            self.header = ct_img.header

            # Import mask
            mask_img = nib.load(self.region_mask_fname)
            self.region_mask_data = mask_img.get_fdata()
        except Exception as e:
            raise RuntimeError(f"Failed to load CT image from {self.ct_path}: {e}")

    def apply_mask(self):
        """
        Applies mask onto the CT and returns the numpy array to self.clean_region_ct_data
        
        Args:
            None
            
        Returns:
            None
        """
        masked_data = np.where(self.region_mask_data > 0, self.region_ct_data, 0)
        self.clean_flat_region_ct_data = masked_data[masked_data > 0]

    def extract_voxel_volume(self):
        """
        Calculate the volume of a single voxel using NIfTI header.
        
        Args:
            None
            
        Returns:
            None
        """
        self.voxel_volume = np.prod(self.header.get_zooms())

    def extract_region_features(self):
        """
        Compute region-specific features:
        
        Args:
            None
            
        Returns:
            None
        """
        # Add calculated attributes here (if any)
        pass

    def determine_location(self):
        """
        Determine the base region name and hemisphere from the region name.
        
        Args:
            None
            
        Returns:
            None
        """
        if self.region_name.startswith("left"):
            self.hemisphere = "left"
            self.base_region = self.region_name.replace("left", "", 1)
        elif self.region_name.startswith("right"):
            self.hemisphere = "right"
            self.base_region = self.region_name.replace("right", "", 1)
        else:
            self.hemisphere = None
            self.base_region = self.region_name

    def add_eval(self, key, value):
        """
        Adds evaluations to the feature set
        
        Args:
            key (str): Feature name.
            value (any): Feature value.
            
        Returns:
            None
        """
        self.features[f"{self.region_name}_{key}"] = value

    def get_features(self):
        """
        Dynamically collect all numeric region features (excluding non-numeric/internal fields).
        
        Args:
            None
            
        Returns:
            dict: Dictionary with keys prefixed by the region name.
        """
        return self.features.copy()


class Evaluator:
    """
    Evaluator provides a collection of static methods to compute and assign
    clinical and imaging-derived features to a PatientNIFTI object and its
    associated brain regions. It supports both patient-level evaluations
    and region-level assessments.
    """

    @staticmethod
    def calc_radiomics_features(nifti_obj, radiomics_features=None):
        """
        Runs the radiomics feature extractor and returns a dictionary of features.
        If radiomics_features is provided, only those features are returned (if present),
        with diagnostics features excluded and 'original_' prefix removed from keys.
        If None, all features are returned, but diagnostics features are excluded and 'original_' prefix is removed
        
        Args:
            nifti_obj (PatientNIFTI or RegionNIFTI): The NIFTI object to extract features from.
            radiomics_features (list, optional): List of specific feature names to extract. Defaults to None.
        
        Returns:
            filtered_features (dict): Dictionary of extracted radiomics features.
        """
        # Suppress radiomics logging
        logger = logging.getLogger("radiomics")
        logger.setLevel(logging.ERROR)

        extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()

        # Provide paths based on object type
        if isinstance(nifti_obj, PatientNIFTI):
            ct_img_path = os.path.join(
                nifti_obj.patient_folder, f"{nifti_obj.subj_id}_reg.nii.gz"
            )
            atlas_path = os.path.join(
                nifti_obj.patient_folder, "masks", f"{nifti_obj.subj_id}_atlas.nii.gz"
            )
            features = extractor.execute(ct_img_path, atlas_path)
        elif isinstance(nifti_obj, RegionNIFTI):
            ct_img_path = nifti_obj.region_ct_fname
            mask_path = nifti_obj.region_mask_fname
            features = extractor.execute(ct_img_path, mask_path)

        # If a list is provided, filter to only those features (before renaming)
        if radiomics_features is not None:
            features = {k: features[k] for k in radiomics_features if k in features}

        # Remove diagnostics features and strip 'original_' prefix from all keys
        filtered_features = {}
        for k, v in features.items():
            if "diagnostics" in k:
                continue
            new_key = k
            if new_key.startswith("original_"):
                new_key = new_key[len("original_") :]
            filtered_features[new_key] = v
        return filtered_features

    @staticmethod
    def evaluate_patient(patient: PatientNIFTI):
        """
        Run patient-level evaluations and add them to the patient object.
        
        Args:
            patient (PatientNIFTI): The patient object to evaluate.
        
        Returns: 
            None
        """
        radiomics_features = None  # Can specify a list of features to compute
        radiomics_dict = Evaluator.calc_radiomics_features(patient, radiomics_features)
        for feature, value in radiomics_dict.items():
            try:
                patient.add_eval(feature, float(value))
            except Exception:
                pass  # skip non-numeric or missing values

    @staticmethod
    def evaluate_region(patient_region: RegionNIFTI):
        """
        Run region-level evaluations and store in each RegionNIFTI object.
        
        Args:
            patient_region (RegionNIFTI): The region object to evaluate.
        
        Returns:
            None
        """
        # Radiomics features
        radiomics_features = None  # Can specify a list of features to compute
        radiomics_dict = Evaluator.calc_radiomics_features(
            patient_region, radiomics_features
        )
        for feature, value in radiomics_dict.items():
            try:
                patient_region.add_eval(feature, float(value))
            except Exception:
                pass  # skip non-numeric or missing values

    @staticmethod
    def compare_left_right_regions(patient: PatientNIFTI, feature_names=None):
        """
        Compares left vs right regions and adds evals to the patient.
        For each base_region, compares features present in both left and right regions.
        Adds difference and ratio as new features to the patient.
        If feature_names is provided, only those features are compared; otherwise, all features are compared.
        feature_names should be the suffix without the left_/right_ prefix.
        
        Args:
            patient (PatientNIFTI): The patient object containing regions to compare.
            feature_names (list, optional): List of feature names (suffixes) to compare. Defaults to None.
        
        Returns:
            None
        """
        for base_region, (
            left_region,
            right_region,
        ) in patient.get_left_right_pairs().items():
            left_feats = left_region.get_features()
            right_feats = right_region.get_features()
            # Find features present in both left and right (by suffix)
            for lf in left_feats:
                # Remove the region prefix to get the feature name
                if not lf.startswith(left_region.region_name + "_"):
                    continue
                feature_name = lf[len(left_region.region_name) + 1 :]
                # Only compare if feature_names is None or feature_name is in feature_names
                if feature_names is not None and feature_name not in feature_names:
                    continue
                rf = f"{right_region.region_name}_{feature_name}"
                if rf in right_feats:
                    left_value = left_feats[lf]
                    right_value = right_feats[rf]
                    if left_value is not None and right_value is not None:
                        diff = right_value - left_value
                        abs_diff = abs(diff)
                        ratio = right_value / left_value if left_value != 0 else None
                        patient.add_eval(
                            f"{base_region}_{feature_name}_RL_diff", diff, prefix=""
                        )
                        patient.add_eval(
                            f"{base_region}_{feature_name}_RL_ratio", ratio, prefix=""
                        )
                        patient.add_eval(
                            f"{base_region}_{feature_name}_RL_absdiff",
                            abs_diff,
                            prefix="",
                        )

    def compare_region_to_full(patient: PatientNIFTI, region_name: str):
        """
        Compares a specific region to the full CT image and adds evaluations to the patient.
        Computes the difference and ratio of the region's features compared to the full CT features.
        
        Args:
            patient (PatientNIFTI): The patient object containing the region to compare.
            region_name (str): The name of the region to compare.
        
        Returns:
            None
        """
        # TODO is bugged and doesn't do anything
        if region_name not in patient.regions:
            raise ValueError(
                f"Region '{region_name}' not found in patient {patient.subj_id}"
            )

        region_obj = patient.regions[region_name]
        full_ct_features = patient.features
        region_features = region_obj.get_features()

        for feature, value in region_features.items():
            # TODO doesn't match feature name
            if feature in full_ct_features:
                full_value = full_ct_features[feature]
                diff = value - full_value
                abs_diff = abs(diff)
                ratio = value / full_value if full_value != 0 else None
                patient.add_eval(f"{region_name}_{feature}_diff", diff, prefix="")
                patient.add_eval(f"{region_name}_{feature}_ratio", ratio, prefix="")
                patient.add_eval(
                    f"{region_name}_{feature}_absdiff", abs_diff, prefix=""
                )

    def evaluate_bilat_regions(patient: PatientNIFTI):
        """
        Evaluates a specific region as a single entity, aggregating its features.
        This is useful for cases where the region is treated as a whole rather than split into left/right.
        
        Args:
            patient (PatientNIFTI): The patient object containing the regions to evaluate.
            
        Returns:
            None
        """
        pass

    @staticmethod
    def evaluate(patient: PatientNIFTI):
        """
        Run all evaluations (patient and region level).
        
        Args:
            patient (PatientNIFTI): The patient object to evaluate.
            
        Returns:
            None
        """
        tqdm.write(f"\nEvaluating {patient.subj_id}...")
        Evaluator.evaluate_patient(patient)
        for region_obj in tqdm(
            patient.regions.values(),
            total=len(patient.regions.values()),
            desc=f"Evaluating {patient.subj_id} Regions",
            leave=False,
            dynamic_ncols=True,
        ):
            Evaluator.evaluate_region(region_obj)

            # TODO not used yet since I doubt its usefulness, but could be useful in the future.
            # Evaluator.compare_region_to_full(patient, region_obj.region_name)
        patient.merge_region_features()
        Evaluator.compare_left_right_regions(patient)


def generate_all_patients(base_folder: str, cohort_csv: str, processed_folder: str, excluded_subjs: list[int] = None) -> list[PatientNIFTI]:
    """
    Generate all patients listed in the cohort CSV, build PatientNIFTI objects,
    compute features, and return them as a list.

    Args:
        base_folder (str): Root directory containing the CSV.
        cohort_csv (str): Filename of the cohort CSV.
        processed_folder (str): Path to processed NIfTI files.

    Returns:
        patients (list): List of PatientNIFTI objects.
    """
    df_cohort = pd.read_csv(os.path.join(base_folder, cohort_csv))
    patients = []

    for _, row in tqdm(
        df_cohort.iterrows(), total=len(df_cohort), desc="Generating Patients"
    ):
        subj_id = str(row["subj_id"])

        # Skip excluded subjects
        if int(subj_id) in excluded_subjs:
            tqdm.write(f"Excluding patient {subj_id}...")
            continue

        patient_folder = os.path.join(processed_folder, subj_id)
        metadata = row.to_dict()

        try:
            patient = PatientNIFTI(
                subj_id=subj_id,
                patient_folder=patient_folder,
                metadata=metadata,
            )

            # Evaluate more complicated calcs after initialization and add to feature dict
            Evaluator.evaluate(patient)

            patients.append(patient)
            tqdm.write(f"Loaded patient: {subj_id}")
        except Exception as e:
            tqdm.write(f"Failed to load patient {subj_id}: {e}")
            traceback.print_exc()

    return patients


def save_all_patients(patient_list: list, csv_fname: str) -> None:
    """
    Saves the featuress of a list of PatientNIFTI objects to a CSV file.

    Args:
        patient_list (List): List of PatientNIFTI objects that implement `.get_features()`.
        csv_fname (str): Full path to the output CSV file.
    
    Returns:
        None
    
    Notes:
        - Ensures the directory for the file exists before saving.
        - Assumes each patient object has a `get_features()` method returning a flat dict of features.
    """
    os.makedirs(os.path.dirname(csv_fname), exist_ok=True)

    # Extract features into DataFrame
    df = pd.DataFrame([p.get_features() for p in patient_list])

    # Save to CSV
    df.to_csv(csv_fname, index=False)
    print(f"[INFO] Saved patient features to: {csv_fname}")


def load_all_patients(
    csv_fname: str,
    exclude_col: list[str],
    ground_truth: list[str],
    exclude_col_match: list[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads patient features from a CSV and separates predictors and target labels.

    Args:
        csv_fname (str): Path to the CSV file containing patient features.
        exclude_col (List[str]): List of columns to exclude from predictors (e.g., patient_id, metadata).
        exclude_col_match (List[str]): List of strings; any column containing one of these will be excluded.
        ground_truth (List[str]): List of column(s) to treat as ground truth labels for ML training.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (predictors, ground_truth) where:
            - predictors contains predictor features (excluding columns in `exclude_col` and `ground_truth`)
            - ground_truth contains the target label columns

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        KeyError: If any of the ground truth columns are missing from the CSV.
    """
    if not os.path.exists(csv_fname):
        raise FileNotFoundError(f"[ERROR] CSV file not found: {csv_fname}")

    df = pd.read_csv(csv_fname)

    # Check for missing ground truth columns
    missing_gt = [col for col in ground_truth if col not in df.columns]
    if missing_gt:
        raise KeyError(f"[ERROR] Missing ground truth columns in CSV: {missing_gt}")

    # Filter out only existing predictor columns to exclude
    exclude_col = [col for col in exclude_col if col in df.columns]
    # Exclude columns matching any string in exclude_col_match
    if exclude_col_match:
        exclude_col += [
            col for col in df.columns if any(s in col for s in exclude_col_match)
        ]
    # Deduplicate exclude_col
    exclude_col = list(set(exclude_col))

    # Split predictors and labels
    predictors = df.drop(columns=exclude_col + ground_truth)
    ground_truth = df[ground_truth]

    print(f"[INFO] Loaded data from: {csv_fname}")
    print(
        f"[INFO] Predictors shape: {predictors.shape}, Ground truth shape: {ground_truth.shape}"
    )

    return predictors, ground_truth


def main():
    """
    Main entry point.
    Sets platform-specific paths, validates required input, loads all patients,
    and prints their features.
    
    Args:
        None
        
    Returns:
        None
    """
    BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_FOLDER = os.path.join(BASE_FOLDER, "processedNIFTIs/")

    INPUT_CSV_FILE = "mhhcohort.csv"

    EXCLUDED_SUBJECTS = [10006, 10007, 10024, 10032, 10035, 10094]  # List of subject IDs to exclude
    # Patients w/o a processed NIFTI folder will automatically be excluded

    # Validate required files
    required_files = [os.path.join(BASE_FOLDER, INPUT_CSV_FILE)]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    # Load patients and extract features
    patients = generate_all_patients(
        base_folder=BASE_FOLDER,
        cohort_csv=INPUT_CSV_FILE,
        processed_folder=PROCESSED_FOLDER,
        excluded_subjs=EXCLUDED_SUBJECTS,
    )

    # Save patient features to CSV
    CSV_FNAME = os.path.join(BASE_FOLDER, "trainingML", "patient_features.csv")
    save_all_patients(patients, csv_fname=CSV_FNAME)

    # Load patient features from CSV
    # exclude_col = ["subj_id", "mrn"]
    # ground_truth = ["pts"]
    # load_all_patients(CSV_FNAME, exclude_col, ground_truth)

    # Print extracted features summary for each patient
    for p in patients:
        print(f"\nSubject ID: {p.subj_id}")
        print(f"PTS label: {p.features['pts']}")
        print(f"Total Features Extracted: {len(p.features)}")
        # Group features by type (e.g., region, patient, RL comparison)
        region_feats = [k for k in p.features if any(h in k for h in ["left", "right"])]
        rl_feats = [
            k for k in p.features if k.endswith("_RL_diff") or k.endswith("_RL_ratio")
        ]
        patient_feats = [
            k
            for k in p.features
            if k not in region_feats
            and k not in rl_feats
            and not "subj_id" in k
            and not "mrn" in k
        ]
        print(f"  Patient-level features: {len(patient_feats)}")
        print(f"  Region-level features: {len(region_feats)}")
        print(f"  Left-Right comparison features: {len(rl_feats)}")

        # Show a sample of each type
        def print_sample(feat_list, label, n=5, seed=42):
            print(f"    Sample {label}:")
            if len(feat_list) == 0:
                print("      (none)")
                return
            random.seed(seed)
            sample_keys = random.sample(feat_list, min(n, len(feat_list)))
            for k in sample_keys:
                v = p.features[k]
                if isinstance(v, float):
                    print(f"      {k}: {round(v, 3)}")
                else:
                    print(f"      {k}: {v}")

        print_sample(patient_feats, "patient-level features")
        print_sample(region_feats, "region-level features")
        print_sample(rl_feats, "left-right comparison features")
        # Optionally, print all features for a single patient for inspection
        # for k, v in p.features.items():
        #     print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
