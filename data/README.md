# Data Setup Instructions

This directory is used to store data files for the Spiegel RAG system. It is not tracked by git
to avoid storing large data files in the repository.

## Expected Data

The system expects the following data structure:

```
data/
├── csv_daten/           # CSV files from Der Spiegel archive
│   ├── 1948.csv
│   ├── 1949.csv
│   ├── ...
│   └── 1979.csv
└── processed/           # Processed data (optional)
```

## Obtaining Der Spiegel Data

1. Download the Der Spiegel archive CSV files
2. Place them in the `data/csv_daten/` directory
3. Ensure each file has a year-based filename (e.g., `1949.csv`)

## CSV Structure

Each CSV file should have the following columns:

- `Jahrgang`: Publication year
- `Ausgabe`: Issue number
- `nr_in_issue`: Article number in the issue
- `Datum`: Publication date
- `Artikeltitel`: Article title
- `Untertitel`: Subtitle (optional)
- `Schlagworte`: Keywords (optional)
- `Autoren`: Authors (optional)
- `URL`: URL to the article (optional)
- `Text`: The full article text

## Example Code for Downloading Data

If you have access to the HU-Box link, you can use the following code to download and extract the data:

```python
import requests
import zipfile
import os

# URL der Zip-Datei
url = 'https://box.hu-berlin.de/f/1664f2bc4fa9439a8590/?dl=1'

# Zip-Datei herunterladen
response = requests.get(url)
zip_filename = 'downloaded_file.zip'

# Zip-Datei speichern
with open(zip_filename, 'wb') as f:
    f.write(response.content)

# Verzeichnis für extrahierte Dateien erstellen
extract_dir = 'data'
os.makedirs(extract_dir, exist_ok=True)

# Dateien entpacken
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f'Files extracted to {extract_dir}')
```