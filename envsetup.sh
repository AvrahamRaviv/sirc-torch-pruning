#export PATH=$PATH:/miniconda/envs/devel/bin/
#conda init bash
source /miniconda/etc/profile.d/conda.sh && conda activate devel
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

# export REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/ca-r2-public64.crt