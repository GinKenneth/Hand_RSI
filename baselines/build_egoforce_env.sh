#!/bin/bash
# EgoForce needs mmdet 3.x (DetInferencer) -> mmcv 2.x with CUDA ops; ViTPose (front end) needs mmcv 1.x, and there is no
# wheel exists for torch 2.7.1+cu128, so mmcv 2.1.0 is built here and everything lands in a separate --target dir,
# put first on sys.path by run_egoforce.py only.
set -e
E=${BASELINES_OUT:-baselines_out}/envs; mkdir -p $E/wheels $E/egoforce_site
PY=${PYTHON:-python}
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0 MMCV_WITH_OPS=1 FORCE_CUDA=1 MAX_JOBS=64
export PIP_CACHE_DIR=$E/pip_cache
cd $E
# the environment's setuptools may lack pkg_resources (mmcv's setup.py needs it): an older one in a side dir, build only
$PY -m pip install --target $E/buildtools "setuptools<70" wheel
export PYTHONPATH=$E/buildtools
echo "$(date +%H:%M) build mmcv 2.1.0"
$PY -m pip wheel --no-deps --no-build-isolation mmcv==2.1.0 -w $E/wheels
unset PYTHONPATH
echo "$(date +%H:%M) install"
$PY -m pip install --no-deps --target $E/egoforce_site $E/wheels/mmcv-2.1.0*.whl mmengine==0.10.5 mmdet==3.3.0
echo "$(date +%H:%M) BUILD_DONE"
