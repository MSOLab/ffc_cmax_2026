import argparse
import logging
from pathlib import Path

import pandas as pd

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_excel_to_csv(excel_file_path: Path):
    """
    Reads an Excel file and saves each sheet as a separate CSV file.

    The output CSV files are named using the convention:
    {excel_filename_stem}_{sheet_name}.csv

    Args:
        excel_file_path (Path): The path to the input Excel file.
    """
    if not excel_file_path.exists():
        logging.error(f"Error: The file '{excel_file_path}' does not exist.")
        return

    try:
        # Create an ExcelFile object to access sheet names
        xls = pd.ExcelFile(excel_file_path)
    except Exception as e:
        logging.error(f"Error reading Excel file: {e}", exc_info=True)
        return

    excel_filename_stem = excel_file_path.stem
    output_dir = excel_file_path.parent

    logging.info(f"Processing '{excel_file_path.name}'...")
    logging.info(f"Found sheets: {xls.sheet_names}")

    for sheet_name in xls.sheet_names:
        try:
            # Read the specific sheet into a DataFrame
            df = pd.read_excel(xls, sheet_name=sheet_name)

            # Sanitize sheet name for use in filename
            sanitized_sheet_name = "".join(
                c if c.isalnum() else "_" for c in sheet_name
            )

            # Construct the output CSV filename
            output_csv_name = f"{excel_filename_stem}_{sanitized_sheet_name}.csv"
            output_csv_path = output_dir / output_csv_name

            # Save the DataFrame to a CSV file
            df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            logging.info(f"Successfully created '{output_csv_path}'")

        except Exception as e:
            logging.error(f"Error processing sheet '{sheet_name}': {e}", exc_info=True)

    logging.info("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split an Excel file into multiple CSV files, one for each sheet."
    )
    parser.add_argument(
        "excel_file",
        type=str,
        help="Path to the Excel file to be split.",
    )
    args = parser.parse_args()

    excel_file = Path(args.excel_file)
    split_excel_to_csv(excel_file)
