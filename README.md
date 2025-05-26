# Quantization_toy
onnx-> tidl build toy project



# install TI QAT module
```
mkdir ~/utils && cd ~/utils
git clone https://github.com/TexasInstruments/edgeai-tensorlab.git
cd edgeai-tensorlab
git checkout -b r9.2 -t origin/r9.2
cd edgeai-modeloptimization/torchmodelopt
./setup.sh
```

# export qdq onnxfile
```
python toy_convert.py
```

# build tidl import env
```
cd ~/utils
git clone https://github.com/TexasInstruments/edgeai-tidl-tools.git
cd edgeai-tidl-tols
git checkout -b 10_00_07_00 10_00_07_00
source ./scripts/docker/build_docker.sh
source ./scripts/docker/run_docker.sh
cd /home/root/
export SOC=am67a
```
