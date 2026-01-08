import os
from pathlib import Path
from huggingface_hub import hf_hub_download

current_script_path = Path(__file__).resolve()

project_root = current_script_path.parent.parent

data_dir = project_root / "data"

data_dir.mkdir(exist_ok=True)

repo_id = "MoKhaa/FER2013"
filename = "fer2013v2_clean.csv"

# Download the file
print(f"Downloading {filename} from {repo_id}...")

file_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset",     
    local_dir=data_dir,      # saves to 'data/'
    local_dir_use_symlinks=False
)

print(f"Saved at: {file_path}")