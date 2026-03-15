#!/bin/bash
# Create databases for NeuralMemory (app + tests).
# Runs only on first container init (empty data volume).
set -e
createdb -U "$POSTGRES_USER" neuralmemory 2>/dev/null || true
createdb -U "$POSTGRES_USER" neuralmemory_test 2>/dev/null || true
