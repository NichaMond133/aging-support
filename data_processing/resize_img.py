from wand.image import Image
import os

# Path to the input image
input_folder = "data/cervical-extension/starting position"
output_folder = "data/cervical-extension/starting position"

for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            # Open the image
            with Image(filename=input_path) as img:
                print(f"Original size: {img.size}")  # Print original size

                # Crop the image (left, top, width, height)
                img.crop(top=600, width=3032, left=300)

                # Save the cropped image
                img.save(filename=output_path)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

print("Cropped images saved!")