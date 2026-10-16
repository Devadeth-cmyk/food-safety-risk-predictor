
from pathlib import Path
import gzip
import shutil

# Change this to the actual name of your original Chicago CSV.
input_file = Path("Food_Inspections_20261015.csv")
output_file = Path("chicago_inspections.csv.gz")

if not input_file.exists():
    raise FileNotFoundError(
        f"Original CSV not found: {input_file.resolve()}"
    )

original_size = input_file.stat().st_size

with input_file.open("rb") as source:
    with gzip.open(output_file, "wb", compresslevel=6) as compressed:
        shutil.copyfileobj(source, compressed)

compressed_size = output_file.stat().st_size

print("Compression completed successfully!")
print(f"Original file:   {input_file.resolve()}")
print(f"Compressed file: {output_file.resolve()}")
print(f"Original size:   {original_size / (1024**2):.2f} MB")
print(f"Compressed size: {compressed_size / (1024**2):.2f} MB")
print(
    f"Space saved:     "
    f"{(1 - compressed_size / original_size) * 100:.2f}%"
)
