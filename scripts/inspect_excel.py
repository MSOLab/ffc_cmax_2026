import openpyxl
import pandas as pd
from pathlib import Path

def get_excel_sheet_names(file_path):
    try:
        workbook = openpyxl.load_workbook(file_path, read_only=True)
        return workbook.sheetnames
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"

# --- Start of analysis ---
excel_file = Path('20250715_dispatching.xlsx')
sheet_names = get_excel_sheet_names(excel_file)
print(f"Sheet names in {excel_file.name}: {sheet_names}")

# --- Read and display the '100dashboard' sheet ---
if isinstance(sheet_names, list) and '100dashboard' in sheet_names:
    # Read with header on the second row (index 1) and multi-index columns
    df_dashboard = pd.read_excel(excel_file, sheet_name='100dashboard', header=[0, 1])
    print("\n--- Contents of '100dashboard' sheet ---")
    print(df_dashboard)
else:
    print(f"\n'100dashboard' sheet not found in {sheet_names}.")