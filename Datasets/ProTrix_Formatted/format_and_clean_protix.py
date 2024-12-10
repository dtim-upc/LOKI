import subprocess
import os

# Define the paths for programs
program_1 = "convert_data_single.py"
program_2 = "Remove_Non_English.py"

# Step 1: Execute the first program (convert_data_single.py)
print("Running Step 1: Converting and formatting the data...")
try:
    subprocess.run(["python", program_1], check=True)
    print("Step 1 completed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error during Step 1 execution: {e}")
    exit(1)

# Step 2: Ensure the output of Step 1 is correctly saved
if not os.path.exists("output/formatted_data.json"):
    print("Error: Output file from Step 1 (formatted_data.json) not found.")
    exit(1)

# Move Step 1 output to Step 2 input location
# print("Preparing data for Step 2...")
# os.rename("output/formatted_data.json", "formatted_data.json")

# Step 3: Execute the second program (Remove_Non_English.py)
print("\nRunning Step 2: Removing non-English characters...")
try:
    subprocess.run(["python", program_2], check=True)
    print("Step 2 completed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error during Step 2 execution: {e}")
    exit(1)

print("Both programs executed successfully. Cleaned data is ready in 'formatted_data_cleaned.json'.")
