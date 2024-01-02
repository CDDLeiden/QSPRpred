#!/bin/bash

set -e

exec runuser -u  test-runner -- "$@"


