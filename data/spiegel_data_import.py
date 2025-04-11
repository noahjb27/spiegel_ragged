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
