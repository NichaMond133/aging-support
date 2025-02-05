import os
import shutil

def move_files(source_folder, destination_folder, num_files=None):
    # Check if destination folder exists, if not, create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created destination folder: {destination_folder}")
    
    # Get a list of files in the source folder
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    if not files:
        print("No files to move in the source folder.")
        return

    # Determine how many files to move (all files if num_files is None or larger than available files)
    if num_files is None or num_files > len(files):
        num_files = len(files)
    
    # Move the specified number of files to the destination folder
    for file in files[:num_files]:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(destination_folder, file)
        
        # Move the file
        shutil.move(source_path, destination_path)
        
        print(f"Moved file: {file} to {destination_folder}")


move_files("data/cervical-extension/end position", "data/cervical-extension/end position/train", num_files=35)
move_files("data/cervical-extension/starting position", "data/cervical-extension/starting position/train", num_files=35)

move_files("data/cervical-extension/end position", "data/cervical-extension/end position/valid", num_files=10)
move_files("data/cervical-extension/starting position", "data/cervical-extension/starting position/valid", num_files=10)

move_files("data/cervical-extension/end position", "data/cervical-extension/end position/test", num_files=5)
move_files("data/cervical-extension/starting position", "data/cervical-extension/starting position/test", num_files=5)

