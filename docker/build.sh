#!/bin/bash
set -e

# Build the Docker image for baseline GNN experiments
docker build -t gnn-adversarial-nids:latest -f Dockerfile ..
