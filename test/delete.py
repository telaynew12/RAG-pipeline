import os

# ✅ Define paths (change these as needed)
input_path = r"C:\Users\telay\Downloads\PMS and PEP (1)\PMS and PEP\pms_prod.sql"
output_path = r"C:\Users\telay\Downloads\PMS and PEP (1)\PMS and PEP\pms_prod_cleaned_final.sql"

print("🔄 Cleaning started...")

# ✅ Read input file
with open(input_path, "r", encoding="utf-8", errors="ignore") as infile:
    lines = infile.readlines()

cleaned_lines = []
removed = 0
total = len(lines)

for i, line in enumerate(lines, start=1):
    # Skip empty or whitespace-only lines
    if not line.strip():
        cleaned_lines.append(line)
        continue

    # Split line by comma to find second column (type)
    # Note: assumes line format like: ('id', 'type', ...)
    parts = line.split(",")
    
    if len(parts) > 1:
        # Clean quotes and whitespace from second field
        second_field = parts[1].strip().strip("'").strip('"')
        # Check if second field starts with "App\"
        if second_field.startswith("App\\"):
            removed += 1
            # Skip this line (remove it)
            continue
    
    # If not removed, keep the line
    cleaned_lines.append(line)
    
    # Optional: print progress every 100000 lines
    if i % 100000 == 0 or i == total:
        print(f"  → Processed {i:,}/{total:,} lines...")

# Write cleaned lines to output file
with open(output_path, "w", encoding="utf-8") as outfile:
    outfile.writelines(cleaned_lines)

print("\n✅ Cleaning complete.")
print(f"📄 Total lines processed : {total}")
print(f"🗑️  Lines removed        : {removed}")
print(f"✏️  Lines written        : {len(cleaned_lines)}")
print(f"📁 Cleaned file saved to : {output_path}")
