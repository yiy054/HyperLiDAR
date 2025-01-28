apt-get -y install libmlpack-dev
apt -y install python3-pybind11
apt-get -y install liblapack-dev
apt-get -y install libblas-dev
apt-get -y install libarmadillo-dev
apt-get -y install qt5-default
apt-get -y install cmake libopenblas-dev liblapack-dev libsuperlu-dev libensmallen-dev #libarpack2-de
#apt-get -y install gcc-10.2 g++-10.2
apt-get -y install libomp-dev
apt-get -y install libcereal-dev libstb-dev
apt-get -y install curl

curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz

tar -xf vscode_cli.tar.gz
# sudo apt-get install libmlpack-dev # add again if crashes with fatal error: mlpack/methods/kmeans/kmeans.hpp
#conda create --name 3DLabelProp python=3.7
#conda activate 3DLabelProp
#cd /home
#git clone https://github.com/DarthIV02/3DLabelProp.git
#cd 3DLabelProp/
echo Y | conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio cudatoolkit=11.0 -c pytorch
echo Y | pip install -r requirements.txt
cd cpp_wrappers
bash compile_wrappers.sh
echo Y | pip install --force-reinstall torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
echo Y | conda install -c torchhd torchhd
#export LD_LIBRARY_PATH=/root/anaconda3/envs/3DLabelProp/lib:$LD_LIBRARY_PATH
# OR export LD_LIBRARY_PATH=/home/ubuntu/anaconda3/envs/3DLabelProp/lib:$LD_LIBRARY_PATH


# >kubectl cp --retries 10 semantickitti.zip label-prop-5c586cbd99-9snt6:/home/
# For GUI 

# For VS tunneling
# ./code tunnel
