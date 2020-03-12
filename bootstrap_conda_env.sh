conda create -y -n eyeglasses python==3.6.10
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate eyeglasses

# CUDA 10.0 tools to compile dlib
conda install -yc anaconda cudatoolkit=10.0.130=0 cudnn=7.6.5=cuda10.0_0
conda install -yc conda-forge/label/cf201901 cudatoolkit-dev=10.0=1

# Essentials
which pip | cat
conda install -yc pytorch pytorch=1.4.0=py3.6_cuda10.0.130_cudnn7.6.3_0 torchvision=0.5.0=py36_cu100
pip install tqdm scikit-learn==0.22.1

# Pillow-SIMD, ligjpeg-turbo
conda uninstall -y --force pillow pil jpeg libtiff libjpeg-turbo
pip uninstall -y pillow pil jpeg libtiff libjpeg-turbo
conda install -yc conda-forge libjpeg-turbo
CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
conda install -y jpeg libtiff

# Dlib with CUDA support
conda install -y cmake
cd "$HOME" || return
git clone https://github.com/davisking/dlib
cd dlib || return
python setup.py install

pip install imutils==0.5.3 opencv-python==4.2.0.32
