import pandas as pd
import os

# Mapping from numeric value to risk meaning for each field
risk_label_map = {
    "tobacco_use": {
        3: "High Risk",
        2: "Medium Risk",
        1: "Low Risk",
        0: "Not At Risk",
        -1: "Missing",
    },
    "alcohol_use": {1: "Alcohol Misuse", 0: "Not At Risk", -1: "Missing"},
    "financial_resource_strain": {1: "At risk", 0: "Not at risk", -1: "Missing"},
    "food_insecurity": {
        1: "Food Insecurity Present",
        0: "No Food Insecurity",
        -1: "Missing",
    },
    "transportation_needs": {
        1: "Unmet Transportation Needs",
        0: "No Transportation Needs",
        -1: "Missing",
    },
    "physical_activity": {
        2: "Inactive",
        1: "Insufficiently Active",
        0: "Sufficiently Active",
        -1: "Missing",
    },
    "stress": {
        1: "Stress Concern Present",
        0: "No Stress Concern Present",
        -1: "Missing",
    },
    "social_connections": {
        2: "Socially Isolated",
        1: "Moderately Isolated",
        0: "Moderately Integrated",
        -1: "Missing",
    },
    "intimate_partner_violence": {0: "Patient Declined", -1: "Missing"},
    "depression": {
        1: "Moderate depression",
        0: "None or minimal depression",
        -1: "Missing",
    },
    "housing_stability": {1: "At Risk", 0: "Not At Risk", -1: "Missing"},
    "utilities": {1: "At Risk", 0: "Not At Risk", -1: "Missing"},
    "health_literacy": {
        1: "Inadequate Health Literacy",
        0: "Not At Risk",
        -1: "Missing",
    },
}

# Output CSV path
OUTPUT_CSV = "cohort_stats_by_pte.csv"


def cohort_stats_table(df, outcome_col="pts"):
    # Split by outcome
    pte_mask = df[outcome_col] == 1
    nopte_mask = df[outcome_col] == 0
    df_pte = df[pte_mask]
    df_nopte = df[nopte_mask]

    rows = []
    # Age at CT (always include row)
    age_col = "age_at_ct"
    if age_col in df.columns:
        df[age_col] = pd.to_numeric(df[age_col], errors="coerce")

        def age_stats(subdf):
            mean = subdf[age_col].mean()
            std = subdf[age_col].std()
            return f"{mean:.1f} ± {std:.1f}"

        rows.append(["Age at CT (mean ± SD)", age_stats(df_pte), age_stats(df_nopte)])
    else:
        rows.append(["Age at CT (mean ± SD)", "nan ± nan", "nan ± nan"])
    # Sex
    if "sex" in df.columns:
        male_label = "Male"
        female_label = "Female"
        male_pte = (df_pte["sex"] == 1).sum()
        male_nopte = (df_nopte["sex"] == 1).sum()
        female_pte = (df_pte["sex"] == 0).sum()
        female_nopte = (df_nopte["sex"] == 0).sum()
        rows.append([male_label, male_pte, male_nopte])
        rows.append([female_label, female_pte, female_nopte])
    # Social determinants
    for field, label_map in risk_label_map.items():
        if field in df.columns:
            for val, label in label_map.items():
                pte_count = (df_pte[field] == val).sum()
                nopte_count = (df_nopte[field] == val).sum()
                rows.append([f"{field}: {label}", pte_count, nopte_count])
    # Create DataFrame and write CSV
    out_df = pd.DataFrame(rows, columns=["Characteristic", "PTE", "No PTE"])
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Cohort statistics table written to {OUTPUT_CSV}")
    print(out_df)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "../trainingML/patient_features.csv")
    df = pd.read_csv(csv_path, sep=None, engine="python")
    # Ensure age is numeric
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    if "pts" not in df.columns:
        print("Error: 'pts' column (PTE outcome) not found in CSV.")
        return
    cohort_stats_table(df, outcome_col="pts")


if __name__ == "__main__":
    main()
