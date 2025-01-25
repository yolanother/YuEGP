conda create -n infer_test python=3.10
conda activate infer_test
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# run lm infer
python infer.py

# run reconstruction
bash reconstruct.sh

