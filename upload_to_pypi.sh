#!/bin/bash
set -e

CONDA_ENV_NAME="bbmuse"
RUN="conda run -n $CONDA_ENV_NAME --live-stream"

echo "Is the LICENSE year correct?"
grep "20" LICENSE

read -p "(y/n) "
if [ "$REPLY" != "y" ]; then
  echo Abort.
  exit
fi

echo "Have you set the current version number in 'pyproject.toml'?"
echo -n "current "; grep version pyproject.toml

read -p "(y/n) "
if [ "$REPLY" != "y" ]; then
  echo Abort.
  exit
fi

# Remove previous builds (cleaner)
rm -rf dist build src/*.egg-info

# Build the distribution (needs 'pip install build')
$RUN python -m build

# Following needs 'pip install twine'
$RUN twine check dist/*
$RUN twine upload dist/*
