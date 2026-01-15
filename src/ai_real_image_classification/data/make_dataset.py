import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "data"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "alessandrasala79/ai-vs-human-generated-dataset",
  file_path,
)

print("First 5 records:", df.head())