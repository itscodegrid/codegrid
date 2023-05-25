mkdir -p raw/code_classification
mkdir -p raw/code_clone_detection
mkdir -p raw/code_completion
mkdir -p raw/vulnerability_detection

python join.py raw_splitted/code_classification raw/code_classification/fragments.pkl
python join.py raw_splitted/clone_detection raw/code_clone_detection/fragments.pkl
python join.py raw_splitted/code_completion raw/code_completion/fragments.pkl
python join.py raw_splitted/vulnerability_detection raw/vulnerability_detection/fragments.pkl