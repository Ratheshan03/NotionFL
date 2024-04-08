import firebase_admin
from firebase_admin import credentials, storage

class FileHandler:
    def __init__(self, service_account_path='service.json'):
        # Initialize Firebase Admin SDK with service account credentials
        if not firebase_admin._apps:
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'notionfl-6a713.appspot.com'
            })

        # Set the root folder name
        self.root_folder = 'data_collector'

    def store_file(self, data, cloud_file_path):
        """
        Stores data directly to Firebase Cloud Storage.

        :param data: Data to be stored (string or bytes)
        :param cloud_file_path: Cloud Storage file path where data will be stored
        :return: URL of the uploaded data
        """
        # Prepend the root folder to the cloud file path
        full_cloud_file_path = f"{self.root_folder}/{cloud_file_path}"
    
        # Create a reference to the Cloud Storage bucket
        bucket = storage.bucket()

        # Create a new blob (file) in Cloud Storage
        blob = bucket.blob(full_cloud_file_path)

        # Upload the data
        blob.upload_from_string(data)

        # Make the file publicly accessible
        blob.make_public()

        # Return the public URL
        return blob.public_url
    

    def retrieve_file(self, training_id, file_path):
        """
            Retrieves a file from Firebase Cloud Storage based on training_id and file_path.

            :param training_id: The training ID used to locate the file
            :param file_path: The specific file path within the training ID directory
            :return: The content of the file or None if the file does not exist
        """
        cloud_file_path = f'{self.root_folder}/{training_id}/{file_path}'

        bucket = storage.bucket()
        blob = bucket.blob(cloud_file_path)

        if blob.exists():
            if file_path.endswith('.png') or file_path.endswith('.jpg'):
                return blob.download_as_bytes()
            else:
                return blob.download_as_string().decode('utf-8')
        else:
            return None


    def list_files(self, prefix):
        """
        List all files in Firebase Cloud Storage under a specific prefix.

        :param prefix: The prefix to list files under
        :return: A list of file paths
        """
        # Ensure the prefix is under the root folder
        full_prefix = f"{self.root_folder}/{prefix}"

        # Create a reference to the Cloud Storage bucket
        bucket = storage.bucket()
        
        # List all blobs (files) under the specified prefix
        blobs = bucket.list_blobs(prefix=full_prefix)
        
        # Extract the file paths from the blobs
        file_paths = [blob.name for blob in blobs]
        
        return file_paths