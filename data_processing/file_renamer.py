import os
import uuid

def rename_files(folder_path):
    files = os.listdir(folder_path)

    for i, filename in enumerate(files, start=0):
        file_extension = os.path.splitext(filename)[1]
        # new_name = f"{str(uuid.uuid4())}{file_extension}"
        new_name = f"hand-to-head-{i}{file_extension}"

        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)

        os.rename(old_file, new_file)
        print(f"Renamed: {filename} -> {new_name}")

rename_files("D:/seminar/aging-support/data/hand-to-head/end position")
