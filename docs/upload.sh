#!/usr/bin/env bash

set -e

# Get the commit hash of current HEAD
COMMIT_ID=`git log -1 --pretty=short --abbrev-commit`
MSG="Adding docs to gh_pages for $COMMIT_ID"

# Clone a temporary copy of the repo with just the gh_pages branch
BASE_DIR="`pwd`"
HTML_DIR="$BASE_DIR/_build/html/"
TMPREPO=/tmp/docs/$USER/QSPRpred/
rm -rf $TMPREPO
mkdir -p -m 0755 $TMPREPO
git clone --single-branch --branch gh_pages `git config --get remote.origin.url` $TMPREPO

# Copy the built html docs into the temporary repo and commit & push changes
cd $TMPREPO
QSPPRED_VERSION=$(python -c "import qsprpred; print(qsprpred.__version__)")
rm -rf "versions/$QSPPRED_VERSION/"
mkdir -p "versions/$QSPPRED_VERSION/"
cp -r $HTML_DIR/* "versions/$QSPPRED_VERSION/"
# link to latest if this is not a dev version or an alpha, beta or rc release
if [[ $QSPPRED_VERSION != *"alpha"* ]] && [[ $QSPPRED_VERSION != *"beta"* ]] && [[ $QSPPRED_VERSION != *"rc"* ]] && [[ $QSPPRED_VERSION != *"dev"* ]]; then
    rm -rf "docs/"
    ln -sf "versions/$QSPPRED_VERSION/" "docs/"
fi
touch .nojekyll
git add -A
git commit -m "$MSG"
git push origin gh_pages
