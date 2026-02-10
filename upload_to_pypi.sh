#!/bin/bash

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
echo rm -rf dist build *.egg-info

# Following needs 'pip install twine'
twine check dist/* && twine upload dist/*
