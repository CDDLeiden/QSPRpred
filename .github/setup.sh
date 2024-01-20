#!/bin/bash

RUNNER_OS=$1
env=$2

# Set up Clustal Omega and MAFFT
export HOME_DIR=$PWD
if [ "$RUNNER_OS" = "macOS" ]; then
wget http://www.clustal.org/omega/clustal-omega-1.2.3-macosx -O clustalo && chmod +x clustalo
brew install mafft && unset MAFFT_BINARIES
echo "$HOME_DIR" >> $GITHUB_PATH # make clustalo available in the next steps
elif [ "$RUNNER_OS" = "Linux" ]; then
wget http://www.clustal.org/omega/clustalo-1.2.4-Ubuntu-x86_64 -O clustalo && chmod +x clustalo
wget https://mafft.cbrc.jp/alignment/software/mafft-7.520-linux.tgz -O mafft.tgz && tar -xzvf mafft.tgz && chmod +x mafft-linux64/mafftdir/bin/mafft
echo "MAFFT_BINARIES=$PWD/mafft-linux64/mafftdir/libexec/" >> $GITHUB_ENV
echo "$HOME_DIR/mafft-linux64/mafftdir/bin/" >> $GITHUB_PATH
echo "$HOME_DIR" >> $GITHUB_PATH # make clustalo available in the next steps
elif [ "$RUNNER_OS" = "Windows" ]; then
choco install clustal-omega mafft
echo "::add-path::$env:ProgramFiles\Clustal Omega"
echo "::add-path::$env:ProgramFiles\MAFFT"
fi
clustalo --version  # For debugging clustalo version
mafft --version  # For debugging mafft version

# Set up Python environment
python -c "print('Python version: ' + '$(python --version)')"
python -c "import platform; print('System info: ', platform.system(), platform.release())" # For debugging OS version
python -m pip install ".[full]"  --no-cache-dir
python -c "import qsprpred; print(qsprpred.__version__)" # For debugging package version
python -m pip install pytest
python -m pip install jupyterlab
python -m pip freeze # For debugging environment