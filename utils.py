import os

def get_unique_output_dir(base_dir):
    """
    Returns a unique directory name by appending -v2, -v3, etc., if the base directory already exists.
    
    Args:
        base_dir (str): The base directory name.
    
    Returns:
        str: A unique directory name.
    """
    if os.path.exists(base_dir):
        version = 2
        new_output_dir = f"{base_dir}-v{version}"
        while os.path.exists(new_output_dir):
            version += 1
            new_output_dir = f"{base_dir}-v{version}"
        return new_output_dir
    return base_dir