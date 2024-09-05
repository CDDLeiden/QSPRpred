#!/bin/bash

# define function to format qspr code
export QSPR=${QSPR:-$(pwd)}
echo $QSPR
function format_qspr {
    black --line-length 88 "$1"
    isort --profile black "$1"
    yapf -r -i -p --style "${QSPR}/pyproject.toml" "$1"
    ruff check --config "${QSPR}/pyproject.toml" "$1"
}
export -f format_qspr

# check if argument is given
if [ $# -eq 0 ]; then
    echo "No arguments supplied"
    exit 1
fi

# iterate through provided directories and recursively apply style
for arg in "$@"
do
    if [ -d "$arg" ]; then
        find "$arg" -type f -name "*.py" -exec bash -c 'format_qspr "$0"' {} \;
    else
        format_qspr "$arg"
    fi
done
