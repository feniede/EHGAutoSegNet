import os

def list_files_with_extension(directory, extension):
    """
    Returns a list of file paths with a specific extension in the specified directory
    and its subdirectories.

    Args:
        directory (str): Directory to start the search from.
        extension (str): File extension to filter the files.

    Returns:
        List[str]: List of file paths with the specified extension.
    """
    file_paths = []
    file_names = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_paths.append(root.split(directory)[-1])
                file_names.append(file)
    return file_paths, file_names