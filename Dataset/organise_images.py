import os
import shutil

# Define the source directory and the destination directories
source_directory = '/home/cheng/Downloads/Frames_Extracted'
fall_directory = '/home/cheng/Repos/Fall_Detection_NN/Dataset/Fall'
unfall_directory = '/home/cheng/Repos/Fall_Detection_NN/Dataset/unFall'

# Ensure the destination directories exist
os.makedirs(fall_directory, exist_ok=True)
os.makedirs(unfall_directory, exist_ok=True)

# Iterate through the files in the source directory
for filename in os.listdir(source_directory):
    source_path = os.path.join(source_directory, filename)

    if filename.lower().endswith('unfall.jpg') or filename.lower().endswith('unfall'):
        destination_path = os.path.join(unfall_directory, filename)
    elif filename.lower().endswith('fall.jpg') or filename.lower().endswith('fall'):
        destination_path = os.path.join(fall_directory, filename)
    else:
        # Skip files that don't match the criteria
        continue

    try:
        # Copy the file to the appropriate destination directory
        shutil.copy(source_path, destination_path)
    except Exception as e:
        print(f"Error copying '{filename}': {str(e)}")

print("Files have been copied into 'fall' and 'unfall' folders.")

