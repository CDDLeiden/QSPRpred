#!/bin/bash

export TIMESTAMP=$(date +%Y%m%d%H%M%S)

docker-compose -p "test-runner-${TIMESTAMP}" rm -f
echo "Spawning Runner... TIMESTAMP=${TIMESTAMP}"
mkdir -p logs
docker-compose -p "test-runner-${TIMESTAMP}" config >> "logs/${TIMESTAMP}.log"
docker-compose -p "test-runner-${TIMESTAMP}" up --force-recreate
docker-compose -p "test-runner-${TIMESTAMP}" logs >> "logs/${TIMESTAMP}.log"
echo "Logs saved in logs/${TIMESTAMP}.log"
echo "Cleaning up..."
docker-compose -p "test-runner-${TIMESTAMP}" rm -f
docker image rm "qsprpred-test-runner-${TIMESTAMP}"
