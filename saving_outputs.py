import shutil
import os

backup_dir = 'outputs_2025_08_02'
os.makedirs(backup_dir, exist_ok=True)

folders_to_backup = ['data_output', 'enriched_data', 'processed_data', 'results', 'triplets_data']
for folder in folders_to_backup:
    shutil.copytree(folder, os.path.join(backup_dir, folder), dirs_exist_ok=True)

print("Backup complete.")
