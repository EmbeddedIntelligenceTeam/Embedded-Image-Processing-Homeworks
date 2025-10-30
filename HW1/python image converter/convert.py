from PIL import Image
import os

IMAGE_FILE = "lena_gray.png"  # Name of the image you will convert
OUTPUT_FILE = "image_data.h" # Name of the output .h file
ARRAY_NAME = "my_image_data" # Name of the C array
WIDTH = 128
HEIGHT = 128
# -----------------

# 1. Open the image, convert to grayscale ('L' mode), and resize
img = Image.open(IMAGE_FILE).convert('L').resize((WIDTH, HEIGHT))

# 2. Get the pixels as a list
pixels = list(img.getdata())

# 3. Create the .h file and write into it
with open(OUTPUT_FILE, 'w') as f:
    f.write(f"// Image: {IMAGE_FILE}, Size: {WIDTH}x{HEIGHT}\n")
    f.write(f"unsigned char {ARRAY_NAME}[{len(pixels)}] = {{\n  ")

    # Write the pixels line by line (to make it more readable)
    for i, pixel in enumerate(pixels):
        f.write(f"{pixel}, ")
        if (i + 1) % 16 == 0: # Let there be 16 pixels per line
            f.write("\n  ")

    f.write("\n};\n")

print(f"'{OUTPUT_FILE}' file was created successfully!")
