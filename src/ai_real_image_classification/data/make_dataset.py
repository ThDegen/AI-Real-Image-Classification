import kagglehub
import shutil

def prepare_dataset(dataset_name, destination="data"):
    path = kagglehub.dataset_download(dataset_name)
    shutil.copytree(path, destination, dirs_exist_ok=True)
    return path, destination

if __name__ == "__main__":
    dataset_name = "alessandrasala79/ai-vs-human-generated-dataset"
    print("Downloading dataset...")
    path, destination = prepare_dataset(dataset_name)
    print(f"Dataset downloaded to {destination}")