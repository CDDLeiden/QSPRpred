#!/bin/bash

# define function to format qspr code
export QSPR=${QSPR:-$(pwd)}
echo $QSPR
function format_qspr {
    black --line-length 88 "$1"
    isort --profile black "$1"
    yapf -r -i -p --style "${QSPR}/pyproject.toml" "$1"
    ruff --config "${QSPR}/pyproject.toml" "$1"
}

# check if argument is given
if [ $# -eq 0 ]; then
    echo "No arguments supplied"
    exit 1
fi

# iterate through arguments and apply style
for arg in "$@"
do
    # check file exists
    if [ ! -f $arg ]; then
        echo "File $arg does not exist"
        # continue iteration
        continue
    fi
    echo "Applying style to $arg"
    format_qspr $arg
done
