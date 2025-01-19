import os
import shutil
import matplotlib.pyplot as plt

# The paths
base_dir = "mushrooms"
dataset_dir = os.path.join(base_dir, "dataset")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")


def create_histogram(directory, subset_name):
    class_counts = {
        class_name: len(os.listdir(os.path.join(directory, class_name)))
        for class_name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, class_name))
    }

    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color="skyblue")
    plt.title(f"Image Distribution in {subset_name} Subset")
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{subset_name}_histogram.png")
    plt.show()

# Check if train, validation, and test directories exist and contain files
if all(os.path.exists(dir) and os.listdir(dir) for dir in [train_dir, val_dir, test_dir]):
    user_input = input(
        "Train, validation, and test directories already exist and contain files. Do you want to redo the distribution? (y/n): "
    ).lower()

    if user_input == "y":
        # Remove existing directories and recreate them
        for dir in [train_dir, val_dir, test_dir]:
            shutil.rmtree(dir)
            os.makedirs(dir, exist_ok=True)
    else:
        print("Skipping distribution process.")
        # Generate histograms for existing subsets
        for subset_dir, subset_name in zip([train_dir, val_dir, test_dir], ["train", "validation", "test"]):
            create_histogram(subset_dir, subset_name)
        exit()

# Function to split data into train, validation, and test subsets
def split_data(class_dir, class_name):
    files = sorted(os.listdir(class_dir))  # Sort files for consistent order
    total_files = len(files)
    train_count = int(0.8 * total_files)
    val_count = int(0.1 * total_files)

    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]

    return train_files, val_files, test_files

# Create subsets and distribute images
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_dir):
        continue

    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    train_files, val_files, test_files = split_data(class_dir, class_name)

    for idx, file in enumerate(train_files):
        new_name = f"{idx + 1:04d}.jpg"
        shutil.copy(os.path.join(class_dir, file), os.path.join(train_dir, class_name, new_name))

    for idx, file in enumerate(val_files):
        new_name = f"{idx + 1:04d}.jpg"
        shutil.copy(os.path.join(class_dir, file), os.path.join(val_dir, class_name, new_name))

    for idx, file in enumerate(test_files):
        new_name = f"{idx + 1:04d}.jpg"
        shutil.copy(os.path.join(class_dir, file), os.path.join(test_dir, class_name, new_name))


for subset_dir, subset_name in zip([train_dir, val_dir, test_dir], ["train", "validation", "test"]):
    create_histogram(subset_dir, subset_name)
