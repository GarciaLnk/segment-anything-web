#!/bin/bash

mamba run --no-capture-output -n sam-demo python scripts/server.py &
mamba run --no-capture-output -n sam-demo python -m http.server -d dist/ &

wait -n

exit $?
