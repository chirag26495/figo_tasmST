# FiGO and TASM\* integration in VDBMS

```
git clone https://github.com/chirag26495/figo_tasmST.git
cd figo_tasmST/
git clone https://github.com/uwdb/TASM.git
cd TASM
git submodule init
git submodule update
cp ../tasm_patch_withFixforRTX30series.patch .
git apply tasm_patch_withFixforRTX30series.patch
cd python/Examples/
git clone https://github.com/jiashenC/FiGO.git
cd FiGO/
### to git home dir
cd ../../../../
cp allObjQueries_FiGO_TASMst.py  figo_newds.patch  figo-tasm_video-object_query.py  TASM/python/Examples/FiGO/
cd TASM/python/Examples/FiGO/
git apply figo_newds.patch
cd weights/
python3 get_weights.py

### to TASM dir
cd ../../../
docker build -t tasm/environment -f docker/Dockerfile.environment  .
docker build -t tasm/tasm -f docker/Dockerfile .
docker run -it --runtime=nvidia -p 8890:8890 --name tasm tasm/tasm:latest /bin/bash

#### Figo dependencies
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install webcolors==1.11.1
sudo apt-get install python3-tk

cd FiGO/
python3 figo-tasm_video-object_query.py --video ../data/birds.mp4 --label bird --labelcount 1 --out query_out/
python3 allObjQueries_FiGO_TASMst.py
```

