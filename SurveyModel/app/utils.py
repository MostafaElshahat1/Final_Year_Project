import pandas as pd

def preprocess_input(raw_data: dict):
    """
    UPGRADED PREPROCESSOR:
    1. Converts string-based numbers from backend to floats safely.
    2. Calculates Trauma_x_Bullying interaction.
    3. Formats for the XGBoost Pipeline.
    """
    # Define columns needed for math
    bl_cols = [f"BL_{i}" for i in range(1, 11)]
    ace_cols = [f"ACES_{i}" for i in range(1, 11)]
    # Add other numeric fields that might come as strings
    other_nums = ["Age", "Parents_Home", "Parents_Dead", "Fathers_Education", 
                  "Mothers_Education", "Co_Curricular", "Percieved_Academic_Abilities"]
    
    # 1. SAFETY CONVERSION: Loop through all potential numeric fields
    for col in bl_cols + ace_cols + other_nums:
        if col in raw_data:
            try:
                # This turns "3" into 3.0 so the model can read it
                raw_data[col] = float(raw_data[col])
            except (ValueError, TypeError):
                # If backend sends "N/A" or something weird, we use 0
                raw_data[col] = 0.0
    
    # 2. Calculate the interaction feature (Same as your old code)
    bl_total = sum([raw_data.get(col, 0) for col in bl_cols])
    ace_total = sum([raw_data.get(col, 0) for col in ace_cols])
    raw_data["Trauma_x_Bullying"] = bl_total * ace_total
    
    # 3. Create DataFrame
    df = pd.DataFrame([raw_data])
    
    return df