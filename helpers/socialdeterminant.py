import os

def is_valid(line):
    """Check if line contains a valid social determinant field."""
    social_det = [
        "Tobacco Use",
        "Alcohol Use",
        "Financial Resource Strain",
        "Food Insecurity",
        "Transportation Needs",
        "Physical Activity",
        "Stress",
        "Social Connections",
        "Intimate Partner Violence",
        "Depression",
        "Housing Stability",
        "Utilities",
        "Health Literacy",
    ]

    return any(det in line for det in social_det)


def get_value(line):
    """Extract risk level from a line and return numeric value and the mapping used."""
    # Define risk patterns and their corresponding values
    risk_mappings = {
        # tobacco_use	
        "High Risk": 3,
        "Medium Risk": 2,
        "Low Risk": 1,
        "Not At Risk": 0,
        
        # alcohol_use	
        "Alcohol Misuse": 1,
        "Not At Risk": 0,
        
        # financial_resource_strain	
        "At risk": 1,
        "Not at risk": 0,
        
        # food_insecurity	
        "Food Insecurity Present": 1,
        "No Food Insecurity": 0,
        
        # transportation_needs	
        "Unmet Transportation Needs": 1,
        "No Transportation Needs": 0,
        
        # physical_activity
        "Inactive": 2,
        "Insufficiently Active": 1,
        "Sufficiently Active": 0,
        
        # stress	
        "Stress Concern Present": 1,
        "No Stress Concern Present": 0,
        
        # social_connections	
        "Socially Isolated": 2,
        "Moderately Isolated": 1,
        "Moderately Integrated": 0,
        
        # intimate_partner_violence	
        "Patient Declined": 0,
        
        # depression	
        "Moderate depression": 1,
        "None or minimal depression": 0,
        
        # housing_stability	
        "At Risk": 1,
        "Not At Risk": 0,
        
        # utilities	
        "At Risk": 1,
        "Not At Risk": 0,
        
        # health_literacy
        "Inadequate Health Literacy": 1,
        "Not At Risk": 0,

        # Missing response
        "Patient Unable To Answer": -1,
        "Unknown": -1,
        "Not on file": -1,
        "Not on File": -1,
    }

    # Check each risk pattern in the line
    for risk_pattern, value in risk_mappings.items():
        if risk_pattern in line:
            return value, risk_pattern

    # If no pattern matched, throw an error to identify unmapped risk patterns
    raise ValueError(
        f"Unknown risk mapping found in line: '{line.strip()}'. Please add this pattern to risk_mappings dictionary."
    )


def rewrite_file(sd_dict):
    """Create header and value lines for CSV output."""
    # Define the standard field order
    standard_fields = [
        "tobacco_use",
        "alcohol_use",
        "financial_resource_strain",
        "food_insecurity",
        "transportation_needs",
        "physical_activity",
        "stress",
        "social_connections",
        "intimate_partner_violence",
        "depression",
        "housing_stability",
        "utilities",
        "health_literacy",
    ]

    # Create mapping from display names to standard field names
    field_mapping = {
        "Tobacco Use": "tobacco_use",
        "Alcohol Use": "alcohol_use",
        "Financial Resource Strain": "financial_resource_strain",
        "Food Insecurity": "food_insecurity",
        "Transportation Needs": "transportation_needs",
        "Physical Activity": "physical_activity",
        "Stress": "stress",
        "Social Connections": "social_connections",
        "Intimate Partner Violence": "intimate_partner_violence",
        "Depression": "depression",
        "Housing Stability": "housing_stability",
        "Utilities": "utilities",
        "Health Literacy": "health_literacy",
    }

    # Create ordered values based on standard field order
    ordered_values = []
    for field in standard_fields:
        # Find the corresponding key in sd_dict
        value = -1  # default
        for key, val in sd_dict.items():
            if field_mapping.get(key) == field:
                value = val
                break
        ordered_values.append(str(value))

    header_line = "\t".join(standard_fields)
    value_line = "\t".join(ordered_values)

    return header_line, value_line


def main():
    BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
    txt_file = os.path.join(BASE_FOLDER, "sd.txt")
    with open(txt_file, "r") as file:
        sd_dict = {}
        sd_mapping_dict = {}  # Track the mappings used
        sd_key = str()
        get_flag = False
        lines = file.readlines()
        for line in lines:
            if is_valid(line) and get_flag == False:
                print(f"Valid line: {line}")
                sd_key = line.strip()
                get_flag = True
                continue
            elif get_flag:
                get_flag = False
                value, mapping = get_value(line)
                sd_dict[sd_key] = value
                sd_mapping_dict[sd_key] = mapping
                continue
    # Print results in a more readable format
    print("\nSocial Determinants Results:")
    print("=" * 40)
    for key, value in sd_dict.items():
        risk_text = {
            3: "High Risk",
            2: "Medium Risk",
            1: "Low Risk",
            0: "Not at risk",
            -1: "Not on file",
        }
        print(
            f"{key}: {value} ({risk_text[value]}) - Mapped from: '{sd_mapping_dict[key]}'"
        )

    # Generate CSV output and write to file
    header_line, value_line = rewrite_file(sd_dict)

    with open(txt_file, "w") as file:
        file.write(header_line + "\n")
        file.write(value_line + "\n")


if __name__ == "__main__":
    main()
