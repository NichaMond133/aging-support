from wand.image import Image
import os
import uuid

def convert_heic_to_jpg(folder_path):
    files = os.listdir(folder_path)

    for i, filename in enumerate(files):
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension == ".heic":
            old_file = os.path.join(folder_path, filename)
            new_name = f"cervical-{i}.jpg"
            new_file = os.path.join(folder_path, new_name)

            with Image(filename=old_file) as img:
                img.format = 'jpeg'
                img.save(filename=new_file)

            print(f"Converted: {filename} -> {new_name}")
        else:
            print(f"Skipped: {filename} (not a HEIC file)")

# Example usage
convert_heic_to_jpg("data/cervical-extension/end position")
