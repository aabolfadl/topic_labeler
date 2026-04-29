import csv
import json

# Read the input CSV
with open("columns.csv", "r", encoding="utf-8") as infile:
    reader = csv.DictReader(infile)

    # Prepare output data
    output_rows = []

    for row in reader:
        research_field = row["research_field"]
        response_str = row["response"]

        # Clean up the response string - remove newlines and extra spaces
        response_str = response_str.replace("\n", "").replace("\r", "")

        # Parse the JSON array from the response column
        try:
            topics = json.loads(response_str)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for {research_field}: {e}")
            print(f"Problematic string: {response_str[:200]}...")
            continue

        # Create a row for each topic
        for topic in topics:
            # Debug: print what keys are available
            if "topic_label" not in topic:
                print(f"Available keys in topic: {topic.keys()}")
                print(f"Topic content: {topic}")

            output_rows.append(
                {
                    "research_field": research_field,
                    "topic_words": ", ".join(topic.get("topic_words", [])),
                    "topic_label": topic.get("topic_label", ""),
                }
            )

# Write the normalized CSV
with open("taxonomy_normalized.csv", "w", encoding="utf-8", newline="") as outfile:
    fieldnames = ["research_field", "topic_words", "topic_label"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(output_rows)

print(
    f"Successfully normalized CSV. Created {len(output_rows)} rows in taxonomy_normalized.csv"
)
