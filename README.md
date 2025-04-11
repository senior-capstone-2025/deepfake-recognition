# Group 6 - Deepfake Detection Capstone

## Getting Started

#### Clone our repository
```Bash
git clone https://github.com/senior-capstone-2025/deepfake-recognition
```
Install submodules (StyleFlow/StyleGRU)
```Bash
git submodule update --init
```

#### Create Virtual Python Environment
(makes package mgmt. much easier)
```Bash
python3 -m venv myenv
source myenv/bin/activate
```
#### Install packages from requirements.txt
```Bash
pip install -r requirements.txt
```

#### Install Jupyter
```
pip install jupyterlab
pip install notebook
jupyter notebook
```

#### Get [DeepSpeak](https://huggingface.co/datasets/faridlab/deepspeak_v1#getting-started) dataset:
```Bash
pip install datasets==3.0.1
huggingface-cli login
```
*Add in token and select 'Y'*
```Python
from datasets import load_dataset
dataset = load_dataset("faridlab/deepspeak_v1", trust_remote_code=True)
```
