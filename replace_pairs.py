import os

# List of (old, new) string pairs to replace
replace_pairs = [
    ("djq1", "djq1"),
    ("djq2", "djq2"),
    ("dj_cds", "dj_cds"),
    ("dj_tp", "dj_tp"),
    ("dj_gupta", "dj_gupta"),
    ("dj_palmer", "dj_palmer"),
    ("dsq1", "dsq1"),
    ("dsq2", "dsq2"),
    ("ds_cds", "ds_cds"),
    ("ds_tp", "ds_tp"),
    ("ds_gupta", "ds_gupta"),
    ("ds_palmer", "ds_palmer"),
]


def should_skip_dir(dirname):
    return dirname.startswith(".")


def replace_in_file(filepath, pairs):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        original = content
        for old, new in pairs:
            content = content.replace(old, new)
        if content != original:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Updated: {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")


for root, dirs, files in os.walk("."):
    # Skip dot-directories
    dirs[:] = [d for d in dirs if not should_skip_dir(d)]
    for file in files:
        if should_skip_dir(file):
            continue
        path = os.path.join(root, file)
        replace_in_file(path, replace_pairs)
