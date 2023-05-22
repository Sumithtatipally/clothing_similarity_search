import os
import csv

# Directory path where the CSV files are located
csv_directory = "//Users/sumith/Documents/Projects/Mercor/GitHub/clothing_similarity_search/scrape/shoppers_stop_data/"

# Output CSV file path
output_csv_file = "shoppers_stop.csv"

# List all CSV files in the directory
csv_files = [file for file in os.listdir(csv_directory) if file.endswith(".csv")]

# Initialize the output CSV file
with open(output_csv_file, "w", newline="") as output_file:
    writer = csv.writer(output_file)

    # Process each CSV file
    for csv_file in csv_files:
        with open(os.path.join(csv_directory, csv_file), "r") as input_file:
            reader = csv.reader(input_file)

            # Skip the header if it's not the first file
            if csv_files.index(csv_file) > 0:
                next(reader)

            # Append the data to the output CSV file
            for row in reader:
                writer.writerow(row)
