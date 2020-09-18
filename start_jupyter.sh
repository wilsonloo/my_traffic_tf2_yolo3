#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/usr/local/lib64" 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"../lib" 
export PYTHONPATH=$PYTHONPATH:/home/wilson/workstation/tensorflow/models/research
export PYTHONPATH=$PYTHONPATH:/home/wilson/workstation/tensorflow/models/research/slim

jupyter notebook --allow-root --no-browser --ip="0.0.0.0"
