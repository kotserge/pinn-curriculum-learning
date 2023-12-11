#!/bin/sh

# Run n experiments for each given configuration file
n=$1
CONFIG=$2

# Check if config file exists and is a file or directory
if [ -d "${CONFIG}" ]; then
    echo "Running ${n} experiments for each configuration file in ${CONFIG}";
    for file in ${CONFIG}/*.yml; do
        for i in $(seq 1 ${n}); do
            echo "Running experiment ${i}";
            python src/main.py ${file};
        done
    done
elif [ -f "${CONFIG}" ]; then
    echo "Running ${n} experiments for configuration file ${CONFIG}";
    for i in $(seq 1 ${n}); do
        echo "Running experiment ${i}";
        python src/main.py ${CONFIG};
    done
    exit 1;
else
    echo "Error: Directory ${CONFIG} does not exists.";
    exit 1;
fi
