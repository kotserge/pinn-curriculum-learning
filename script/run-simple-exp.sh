#!/bin/sh

# Run n experiments for each given configuration file
n=$1
config_dir=$2

for config_file in $config_dir/*.yml
do
    for i in $(seq 1 $n)
    do
        python3 src/main.py $config_file
    done
done