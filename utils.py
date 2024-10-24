import os
import ants

def delete_file_in_subdirs(root_dir, filename):
    """Deletes a file with the specified filename from a directory and its subdirectories.

    Args:
        root_dir (str): The starting directory to search within.
        filename (str): The name of the file to delete.
    """

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if filename in filenames:
            full_path = os.path.join(dirpath, filename)
            try:
                os.remove(full_path)
                print(f"File deleted: {full_path}")
            except OSError as e:
                print(f"Error deleting file: {full_path} - {e}")

def rename_file_in_subdirs(root_dir, filename, new_filename):
    """Renames a file with the specified filename to a new name in a directory and its subdirectories.

    Args:
        root_dir (str): The starting directory to search within.
        filename (str): The name of the file to rename.
        new_filename (str): The new name for the file.
    """

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if filename in filenames:
            old_path = os.path.join(dirpath, filename)
            new_path = os.path.join(dirpath, new_filename)
            try:
                os.rename(old_path, new_path)
                print(f"File renamed from: {old_path} to {new_path}")
            except OSError as e:
                print(f"Error renaming file: {old_path} - {e}")
                
def shift_intensity(image, offset):
    img_array = image.numpy()
    img_array += offset
    ants_img = ants.from_numpy(img_array)
    ants.set_spacing(ants_img, image.spacing)
    ants.set_origin(ants_img, image.origin)
    ants.set_direction(ants_img, image.direction)
    return ants_img

def create_dir(dir_path):
  """
  Creates a directory if it does not exist.

  Args:
    dir_path: The path of the directory to create.

  Returns:
    A string indicating whether the directory was created or already exists.
  """
  try:
    os.makedirs(dir_path)
    return f"Directory '{dir_path}' created successfully."
  except FileExistsError:
    return f"Directory '{dir_path}' already exists."