from pathlib import Path
from .client import fireworks_client

def upload_dataset(
    dataset_path: Path, 
    dataset_id: str, 
    client: fireworks_client
):
    """
    Uploads a dataset to Fireworks.ai.
    """
    print(f"Creating dataset '{dataset_id}' from '{dataset_path}'...")
    client.create_dataset(dataset_path, dataset_id)
    print("Dataset created successfully.")