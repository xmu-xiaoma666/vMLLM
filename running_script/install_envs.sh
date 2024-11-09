# install python envs
conda create -n vMLLM python=3.10 -y
conda activate vMLLM

# install vMLLM
pip install -e .


# install flash-attn
pip install flash-attn --no-build-isolation --no-cache-dir

# install package
pip install -r requirements.txt
