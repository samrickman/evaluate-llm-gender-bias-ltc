# If re-running the analysis you need to delete everything
# by default it will not overwrite files that take days to generate
import pathlib
from constants import dirs_to_delete


def delete_files_with_extension(directories: list[str], extension: str):
    """
    Deletes all files with the specified extension in the given directories and their subdirectories.

    We need this if we run the analysis again from scratch as it mostly will not over-write.
    """

    for directory in directories:
        directory_path = pathlib.Path(directory)

        if directory_path.exists() and directory_path.is_dir():

            files_to_delete = directory_path.glob(f"*.{extension}")

            for file in files_to_delete:
                if file.is_file():
                    print(f"Deleting file: {file}")
                    file.unlink()
        else:
            print(
                f"Cannot delete files from: {directory_path}. It does not exist or is not a directory."
            )


def delete_all_files():
    for ext, dir in dirs_to_delete.items():
        delete_files_with_extension(dir, ext)


if __name__ == "__main__":
    delete_all_files()
