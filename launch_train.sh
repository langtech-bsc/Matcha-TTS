#!/bin/bash

# module load python/3.9.16  # already loaded in node
PYTHON_ENV=<path/to/pythonenv>
ESPEAK_FOLDER=<path/to/espeak>
# activate env
source $PYTHON_ENV/bin/activate
export PYTHON=$PYTHON_ENV/bin/python
# set espeak vars
export ESPEAK_DATA_PATH=$ESPEAK_FOLDER/espeak-ng-data
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ESPEAK_FOLDER/lib
export PATH="${ESPEAK_FOLDER}/bin:$PATH"
export HYDRA_FULL_ERROR=1

# training variables
# export NUM_NODES = $COUNT_NODE

# train script
CUDA_VISIBLE_DEVICES=0
CUDA_LAUNCH_BLOCKING=1

python matcha/train.py experiment=multispeaker_cv_filt_100h \
                 trainer=default \
                 ++num_workers=10 \
                 ++data.num_workers=4 \
                 ++data.batch_size=32 \
                 ++trainer.devices=[0] \
                 ++callbacks.model_checkpoint.every_n_train_steps=10000 \
                 ++callbacks.model_checkpoint.every_n_epochs=null \
                 ++model.optimizer.lr=1e-4 \
                 --config-name train_cv_filt_100h.yaml
