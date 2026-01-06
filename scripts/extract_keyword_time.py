import csv
import glob
import os

import yaml

scenario_name = "dispatch_cjims103_shdlb_baseCp"
results = []
base_dir = os.path.join("pra_100s", scenario_name)
file_pattern = os.path.join(base_dir, "*", "results", "*_obj_log.yaml")
file_paths = glob.glob(file_pattern)

for yaml_file in file_paths:
    try:
        instance_id = os.path.basename(yaml_file).split("_obj_log.yaml")[0]
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if (
            data
            and "obj_value" in data
            and data.get("obj_value")
            and "notes" in data["obj_value"]
        ):
            notes = data["obj_value"]["notes"]
            if isinstance(notes, dict):
                for time, memo in notes.items():
                    if isinstance(memo, str) and "neh_cp" in memo:
                        results.append([instance_id, time])
                        break
    except Exception as e:
        print(f"Error processing file {yaml_file}: {e}")

results.sort(key=lambda x: int(x[0]))

output_csv_file = f"{scenario_name}_initialization_times.csv"

with open(output_csv_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["instanceId", "initialization_time"])
    writer.writerows(results)

print(f"Successfully created {output_csv_file}")
if not results:
    print("Warning: No matching data was found.")
else:
    print(f"Successfully found and wrote {len(results)} rows.")
