import os
from utils.file_handler import FileHandler


class LocalFileSaver:
    def __init__(self):
        self.file_handler = FileHandler()

    def download_and_save_file(self, training_id, cloud_file_path, local_file_path):
        """
        Downloads a file from Firebase Cloud Storage and saves it locally.

        :param training_id: The training ID used to locate the file in the cloud
        :param cloud_file_path: The path of the file in the cloud storage
        :param local_file_path: The local path where the file should be saved
        """
        file_content = self.file_handler.retrieve_file(training_id, cloud_file_path)
        if file_content is not None:
            with open(local_file_path, 'wb') as file:
                file.write(file_content)
            print(f"File downloaded and saved locally at {local_file_path}")
        else:
            print(f"File not found in cloud storage: {cloud_file_path}")


# Usage Example
file_handler = FileHandler
local_file_saver = LocalFileSaver()

# Download and save a file
training_id = 'your_training_id'
cloud_file_path = 'path_of_file_in_cloud_storage'
local_file_path = 'path_where_file_should_be_saved_locally'
local_file_saver.download_and_save_file(training_id, cloud_file_path, local_file_path)
