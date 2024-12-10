import os
import json

# Define input and output paths
input_file = "input/TrixInstruct.json" 
output_file = "output/formatted_data.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Initialize counters
total_tables = 0
total_sentences = 0
failed_entries = 0
skipped_entries = 0
output_data_list = []

# Load the input JSON file
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Helper function to convert table text to list of lists
def parse_table(table_text):
    rows = table_text.splitlines()
    table = [row.split(" | ") for row in rows]
    return table

# Process the JSON data
for entry in data:
    try:
        instruction = entry.get("instruction", "")
        
        # Skip entries with IDs starting with "nt-" and those with ## Table since we only need tables with texts
        if str(entry["id"]).startswith("nt-") and "## Table" in instruction:
            skipped_entries += 1
            continue

        # Extract table context, table title, and caption
        table_context = ""
        table_title = ""
        caption = ""
        table = []

        if "## Table Context" in instruction:
            table_section = instruction.split("## Table Context\n", 1)[-1].split("\n##", 1)[0].strip()
            lines = table_section.split("\n")
            if "Page Title:" in lines[0]:
                table_title = lines[0].replace("Page Title: ", "").strip()
                if "Caption:" in lines[1]:
                    caption = lines[1].replace("Caption: ", "").strip()
                    table_context = "\n".join(lines[2:]).strip()
                else:
                    table_context = "\n".join(lines[1:]).strip()
            else:
                table_context = table_section

            table = parse_table(table_context)

        # Extract sentence context
        sentence_context = ""
        if "## Sentence Context" in instruction:
            sentence_context = instruction.split("## Sentence Context\n", 1)[-1].split("\n##", 1)[0].strip()

        # Skip entries without a sentence context, as we need tables with corresponding text
        if not sentence_context:
            skipped_entries += 1
            continue

        # Convert sentence context into a list of sentences, splitting by newline
        sentence_context_list = sentence_context.split("\n")

        # Count the extracted contexts
        if table:
            total_tables += 1
        if sentence_context_list:
            total_sentences += len(sentence_context_list)

        # Construct the output structure
        output_data = {
            "id": total_tables,  # Assign a new sequential ID based on total_tables
            "original_id": entry["id"],  # Preserve the original ID
            "table_title": table_title.strip() if table_title else "",
            "caption": caption.strip() if caption else "",
            "table": table,
            "sentence_context": sentence_context_list
        }

        output_data_list.append(output_data)

    except Exception as e:
        failed_entries += 1
        print(f"Failed to parse entry with original_id {entry['id']}: {e}")

# Write all entries to a single output JSON file
with open(output_file, "w", encoding="utf-8") as out_f:
    json.dump(output_data_list, out_f, indent=4, ensure_ascii=False)

# Print summary
print(f"Total entries processed: {len(data)}")
print(f"Total tables extracted: {total_tables}")
print(f"Total sentence contexts extracted: {total_sentences}")
print(f"Total skipped entries: {skipped_entries}")
print(f"Total failed entries: {failed_entries}")

# Calculate and print the total number of rows in all tables
total_rows = sum(len(entry['table']) for entry in output_data_list)
print(f"Total rows in all tables: {total_rows}")
