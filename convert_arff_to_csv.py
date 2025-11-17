import pandas as pd

arff_file = "chronic_kidney_disease_full.arff"
csv_file = "data/ckd_data.csv"

# Step 1 — Read file
lines = []
with open(arff_file, "r") as f:
    for line in f:
        lines.append(line.strip())

# Step 2 — Extract attributes
columns = []
data_started = False
data_rows = []

for line in lines:
    if line.lower().startswith("@attribute"):
        parts = line.split()
        col_name = parts[1].strip().replace("'", "")
        columns.append(col_name)

    elif line.lower().startswith("@data"):
        data_started = True

    elif data_started and line and not line.startswith("%"):
        # Clean row
        row = (
            line.replace("?", "NaN")
                .replace(" ", "")
                .replace("\t", "")
                .split(",")
        )

        # If row has more values than columns → trim
        if len(row) > len(columns):
            row = row[:len(columns)]

        # If row has fewer values → pad with NaN
        if len(row) < len(columns):
            row += ["NaN"] * (len(columns) - len(row))

        data_rows.append(row)

# Step 3 — Build DataFrame
df = pd.DataFrame(data_rows, columns=columns)

# Step 4 — Clean string values
for col in df.columns:
    df[col] = df[col].astype(str).str.replace("b'", "").str.replace("'", "")

# Step 5 — Save CSV
df.to_csv(csv_file, index=False)

print("✔ Conversion complete!")
print("✔ Saved as:", csv_file)
print("✔ Shape:", df.shape)
