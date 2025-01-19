import os
from PIL import Image

# Define dataset path
data_dir = "mushrooms/dataset"

def clean_and_rename_class(class_dir):
    files = sorted(os.listdir(class_dir))
    valid_files = []
    deleted_count = 0

    for file in files:
        file_path = os.path.join(class_dir, file)
        try:
            img = Image.open(file_path)
            width, height = img.size
            img.close()  # Ensure the file handle is closed
            if width >= 400 and height >= 400:
                valid_files.append(file)
            else:
                os.remove(file_path)
                deleted_count += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Rename the remaining valid files
    for idx, file in enumerate(valid_files):
        old_path = os.path.join(class_dir, file)
        new_name = f"{idx+1:04d}.jpg"
        new_path = os.path.join(class_dir, new_name)
        os.rename(old_path, new_path)

    return len(valid_files), deleted_count

total_images = 0
total_deleted = 0

# Iterate through each class in the dataset
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        valid_count, deleted_count = clean_and_rename_class(class_dir)
        total_images += valid_count
        total_deleted += deleted_count
        print(f"Class: {class_name}, Remaining: {valid_count}, Deleted: {deleted_count}")

print("-------------------------------------------------------------------\n\n")
print(f"Total images remaining: {total_images}")
print(f"Total images deleted: {total_deleted}")
