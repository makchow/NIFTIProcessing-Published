#
# Process NCCT images by cleaning, registering, and segmenting
# For the MHH Cohort Data
# Authors: Jerome Jeevarajan and Mark Chao
#

import os
import sys
import time
import pandas as pd
import numpy as np
from csv import writer
import nibabel as nib
import fsl
import fsl.wrappers
import fsl.data.image
import fsl.utils.image.roi
import scipy.ndimage
from tqdm import tqdm


def dcm_nii_convert(folder: str):
    """
    Converts DICOM files in the given folder to NIfTI format using dcm2niix.

    Args:
        folder (str): Path to folder containing DICOM images.
        
    Returns:
        None
    """
    file_conversion = "dcm2niix -z y " + folder
    os.system(file_conversion)


def clean_mask(mask: np.ndarray, min_size=50):
    """
    Removes small connected components from a binary mask using connected component analysis.

    Args:
        mask (np.ndarray): Binary image mask (e.g., 0/1).
        min_size (int): Minimum size of a region to retain (default: 50 voxels).

    Returns:
        cleaned_mask (np.ndarray): Cleaned binary mask.
    """
    labeled_array, num_features = scipy.ndimage.label(mask)
    component_sizes = np.bincount(labeled_array.ravel())

    too_small = component_sizes < min_size
    too_small_mask = too_small[labeled_array]
    cleaned_mask = mask.copy()
    cleaned_mask[too_small_mask] = 0

    # Optional: Hole filling could be added here using scipy or FSL
    return cleaned_mask


def atlas_segmentation(
    reg_patient_file: str,
    atlas_file: str,
    labels_csv: str,
    seg_folder: str,
    masks_folder: str,
    subj_id: str,
):
    """
    Segments registered patient brain into labeled anatomical regions using an atlas.

    Args:
        reg_patient_file (str): Path to registered patient NIfTI file.
        atlas_file (str): Path to Talairach atlas NIfTI file.
        labels_csv (str): CSV file mapping atlas indices to labels.
        seg_folder (str): Path to folder in which isolated segments are saved
        masks_folder (str): Path to folder in which masks to segment the CT are saved
        subj_id (str): Patient subject ID.
        
    Returns:
        List[str]: List of normalized anatomical region labels.
    """
    
    tqdm.write(
        "Starting Atlas segmentation to produce the patient-specific atlas masks"
    )
    st = time.time()

    # Load atlas image and label metadata
    atlas_img = nib.load(atlas_file)
    atlas_data = atlas_img.get_fdata()
    atlas_affine = atlas_img.affine
    df = pd.read_csv(labels_csv)
    index_to_label = dict(zip(df["index"], df["label"]))

    # Make atlas mask and save to masks folder
    atlas_mask_img_fname = f"{subj_id}_atlas.nii.gz"
    atlas_mask_data = ((atlas_data != 0) & ~np.isnan(atlas_data)).astype(np.uint8)
    atlas_mask_image = nib.Nifti1Image(atlas_mask_data, atlas_affine)
    nib.save(atlas_mask_image, os.path.join(masks_folder, atlas_mask_img_fname))

    # Normalization function for matching label strings
    def normalize_label(s):
        """
        Normalize label strings by converting to lowercase and removing spaces/underscores.
        
        Args:
            s (str): Input label string.
            
        Returns:
            normalized_labels (str): Normalized label string.
        """
        
        return s.lower().replace(" ", "").replace("_", "").strip()

    normalized_index_to_label = {
        idx: normalize_label(lbl) for idx, lbl in index_to_label.items()
    }

    # Broad anatomical groups for segmentation
    # Commented groups exist in the patient, but have very little or odd data
    broad_groups = [
        "leftcerebrum.frontallobe",
        # "leftcerebrum.frontal-temporalspace",
        "leftcerebrum.limbiclobe",
        "leftcerebrum.occipitallobe",
        "leftcerebrum.parietallobe",
        "leftcerebrum.sub-lobar",
        "leftcerebrum.temporallobe",
        "rightcerebrum.frontallobe",
        # "rightcerebrum.frontal-temporalspace",
        "rightcerebrum.limbiclobe",
        # "rightcerebrum.midbrain",
        "rightcerebrum.occipitallobe",
        "rightcerebrum.parietallobe",
        "rightcerebrum.sub-lobar",
        "rightcerebrum.temporallobe",
        "leftcerebellum.anteriorlobe",
        "leftcerebellum.posteriorlobe",
        # "leftcerebellum.sub-lobar",
        "rightcerebellum.anteriorlobe",
        "rightcerebellum.posteriorlobe",
        # "rightcerebellum.sub-lobar",
        # "leftbrainstem.medulla",
        "leftbrainstem.midbrain",
        "leftbrainstem.pons",
        # "rightbrainstem.medulla",
        "rightbrainstem.midbrain",
        "rightbrainstem.pons",
        # "inter-hemispheric.frontallobe",
        # "inter-hemispheric.limbiclobe",
        # "inter-hemispheric.occipitallobe",
        # "inter-hemispheric.sub-lobar",
    ]

    normalized_labels = []

    # Process each anatomical group
    for group in tqdm(
        broad_groups,
        total=len(broad_groups),
        desc="Segmenting Atlas",
        leave=False,
        dynamic_ncols=True,
    ):
        group_norm = normalize_label(group)
        normalized_labels.append(group_norm.replace(".", "_"))
        matching_indices = [
            idx for idx, lbl in normalized_index_to_label.items() if group_norm in lbl
        ]

        # Create binary mask for this group in atlas space
        mask_data = np.isin(atlas_data, matching_indices).astype(np.uint8)

        # Load registered patient image
        patient_img = nib.load(reg_patient_file)
        patient_data = patient_img.get_fdata()
        patient_affine = patient_img.affine

        binary_mask = (mask_data > 0.5).astype(np.uint8)
        binary_mask_img = nib.Nifti1Image(binary_mask, atlas_affine)

        # Apply mask to isolate region
        region_data = np.where(binary_mask > 0, patient_data, 0)
        region_img = nib.Nifti1Image(region_data, patient_affine)

        # Save segmented region and mask
        region_fname = f"{subj_id}_{group_norm.replace('.', '_')}_region.nii.gz"
        binary_mask_fname = f"{subj_id}_{group_norm.replace('.', '_')}_mask.nii.gz"
        nib.save(region_img, os.path.join(seg_folder, region_fname))
        nib.save(binary_mask_img, os.path.join(masks_folder, binary_mask_fname))

        tqdm.write(f"Saved region CT: {region_fname}")

    # Save subjid_tlabels_regions.txt (list of region labels) to patient folder
    subj_folder, seg_output = os.path.split(seg_folder)
    np.savetxt(
        os.path.join(subj_folder, f"{subj_id}_tlabels_regions.txt"),
        np.array(normalized_labels),
        fmt="%s",
        delimiter="\t",
    )

    tqdm.write(
        f"Segmentation completed in {time.strftime('%H:%M:%S', time.gmtime(time.time() - st))}"
    )

    return normalized_labels


def registration(output_folder: str, ncct_file: str, atlas_file: str, subj_id: str):
    """
    Preprocess and register a patient's NCCT image to the atlas space.

    Args:
        output_folder (str): Base output directory.
        ncct_file (str): Path to raw NCCT file.
        atlas_file (str): Path to Talairach atlas file.
        subj_id (str): Subject ID.

    Returns:
        output_reg (str): Path to registered patient image.
    """
    # Create structured directories
    subj_folder = os.path.join(output_folder, subj_id)
    reg_dir = os.path.join(subj_folder, "registration")
    os.makedirs(reg_dir, exist_ok=True)

    # Load image and get shape
    nib_file = nib.load(ncct_file)
    img = fsl.data.image.Image(ncct_file)
    dimensions = nib_file.shape

    # Set FOV
    if len(dimensions) > 3:
        new_fov = [(0, dimensions[0]), (0, dimensions[1]), (0, dimensions[2]), (0, 1)]
    else:
        new_fov = [(0, dimensions[0]), (0, dimensions[1]), (0, dimensions[2])]
    tqdm.write(f"New FOV: {new_fov}")

    # File paths
    def reg_file(name: str) -> str:
        """
        Helper to generate registration file paths.
        
        Args:
            name (str): Descriptive name for the file.
        
        Returns:
            str: Full path to the registration file.
        """
        return os.path.join(reg_dir, f"{subj_id}_{name}.nii.gz")

    output_fov = reg_file("fov")
    output_fov_thr = reg_file("fov_thr")
    output_sm = reg_file("sm")
    output_sm_thr = reg_file("sm_thr")
    output_ss = reg_file("ss")
    output_ss_mask = reg_file("ss_mask")
    output_ss_mask_fill = reg_file("ss_mask_fill")
    output_prereg = reg_file("prereg")
    output_reg = os.path.join(subj_folder, f"{subj_id}_reg.nii.gz")
    pat_to_atlas_tmatrix = os.path.join(reg_dir, f"{subj_id}_to_atlas_tmatrix")

    # Preprocessing
    tqdm.write("Preprocessing...")
    roi_img = fsl.utils.image.roi.roi(img, new_fov)
    nib.save(roi_img.nibImage, output_fov)
    fsl.wrappers.fslmaths(output_fov).thr(0).uthr(100).run(output_fov_thr)

    fsl.wrappers.fslmaths(output_fov_thr).smooth(0.1).run(output_sm)
    fsl.wrappers.fslmaths(output_sm).thr(0).uthr(100).run(output_sm_thr)

    # Skull stripping
    tqdm.write("Brain Extraction...")
    fsl.wrappers.bet(output_sm_thr, output_ss, f=0.1, m=1)
    fsl.wrappers.fslmaths(output_ss_mask).fillh().run(output_ss_mask_fill)
    fsl.wrappers.fslmaths(output_sm_thr).mul(output_ss_mask_fill).run(output_ss)

    # Registration
    tqdm.write("Registering to atlas...")
    fsl.wrappers.flirt(output_ss, atlas_file, omat=pat_to_atlas_tmatrix)

    # Apply transformon original image
    fsl.wrappers.fslmaths(output_fov_thr).mul(output_ss_mask_fill).run(output_prereg)
    fsl.wrappers.applyxfm(
        output_prereg,
        atlas_file,
        mat=pat_to_atlas_tmatrix,
        out=output_reg,
        interp="spline",
    )

    tqdm.write(f"Saved registered CT: {output_reg}")

    return output_reg


def process(
    base_folder: str,
    input_folder: str,
    output_folder: str,
    atlas_file: str,
    input_csv_file: str,
    labels_csv_file: str,
):
    """
    Full pipeline: load subject list, register each NCCT, and segment into regions.

    Args:
        base_folder (str): Root directory of project.
        input_folder (str): Folder with raw NCCT files.
        output_folder (str): Destination for processed data.
        atlas_file (str): Path to atlas image.
        input_csv_file (str): CSV with subject IDs.
        labels_csv_file (str): CSV mapping labels in the atlas.
    
    Returns:
        None
    """
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    subj_df = pd.read_csv(os.path.join(base_folder, input_csv_file))

    os.makedirs(output_folder, exist_ok=True)

    for _, row in tqdm(
        subj_df.iterrows(),
        total=len(subj_df),
        desc="Processing Patients",
        dynamic_ncols=True,
    ):
        subj_id = str(row["subj_id"])
        ncct_file = os.path.join(input_folder, f"{subj_id}.nii.gz")
        tqdm.write(f"\n--- Processing subject: {subj_id} ---\n")

        subj_output_dir = os.path.join(output_folder, subj_id)
        segmentation_output = os.path.join(subj_output_dir, "segmentation_output")
        masks_output = os.path.join(subj_output_dir, "masks")
        registration_output = os.path.join(subj_output_dir, "registration")

        if os.path.exists(subj_output_dir):
            tqdm.write(f"Output directory for {subj_id} already exists. Skipping...")
            continue
        elif not os.path.exists(ncct_file):
            tqdm.write(f"NCCT file for {subj_id} does not exist: {ncct_file}")
            continue
        else:
            # Create necessary directories
            os.makedirs(subj_output_dir, exist_ok=True)
            os.makedirs(segmentation_output, exist_ok=True)
            os.makedirs(masks_output, exist_ok=True)
            os.makedirs(registration_output, exist_ok=True)

        # Step 1: Register patient CT to atlas
        registered_file = registration(output_folder, ncct_file, atlas_file, subj_id)

        # Step 2: Segment brain regions
        atlas_segmentation(
            registered_file,
            atlas_file,
            labels_csv_file,
            segmentation_output,
            masks_output,
            subj_id,
        )


def main():
    """
    Main entry point for processing NIfTI images.
    
    Args:
        None
        
    Returns:
        None
    """
    # REQUIRED files in BASE_FOLDER:
    # Atlas - talairach.nii
    # Labels CSV - tlabels.csv
    # Cohort CSV - mmhcohort.csv

    # Within the base folder, there are two other folders that are specified:
    # BASE_FOLDER - Contains necessary files for the script to run
    # --> preNIFTs/ - All patient NIFTI files named SUBJ_ID.nii.gz
    # --> processedNIFTIs/ - Processed NIFTI output in the structure of:
    #   --> SUBJ_ID/ - subject ID which is assigned to the patient and includes:
    #       --> SUBJID_reg.nii - Final brain extracted, skull stripped, registered patient CT
    #       --> SUBJID_tlabels_regions.txt - List of anatomical region labels used for segmentation
    #       --> registration/ - Files generated used to aid in registration
    #       --> masks/ - Has masks used for to isolate the segments and full ct
    #       --> segmentation_output/ - Has patient segments according to the atlas labeling

    # Change BASE_FOLDER if a different directory structure is used
    BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
    INPUT_FOLDER = os.path.join(BASE_FOLDER, "preNIFTIs")
    OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "processedNIFTIs/")

    ATLAS_FILE = "talairach.nii"
    INPUT_CSV_FILE = "mhhcohort.csv"
    LABELS_CSV_FILE = "tlabels.csv"

    # Validate required files
    required_files = [ATLAS_FILE, INPUT_CSV_FILE, LABELS_CSV_FILE, INPUT_FOLDER]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    process(
        base_folder=BASE_FOLDER,
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        atlas_file=ATLAS_FILE,
        input_csv_file=INPUT_CSV_FILE,
        labels_csv_file=LABELS_CSV_FILE,
    )


if __name__ == "__main__":
    main()
