# Testing  QSPRpred

This is the directory to run various tests on QSPRpred. You can run all tests with
the [`./run.sh`](./run.sh) script. There are a couple environment variables that you can
set to control which tests are run and how:

- `QSPPRED_TEST_TUTORIAL`: Whether to run all tutorial code. Default: `true`.
- `QSPPRED_TEST_SPLIT_UNITS`: Whether to run all unit tests at once or split them to
  smaller chunks. Default: `false`. This is mainly to prevent memory hogging on GitLab
  runners.
- `QSPPRED_TEST_EXTRAS`: Whether to run unit tests for the `qsprpred.extra` module as
  well. Default: `true`.

All tests should be located in a folder with the `test_` prefix and have their
own `run.sh` script that can be used to run them individually or modify behavior.

A Docker-based test runner is also provided in [`./runner`](./runner). This can be used
to run the whole test pipeline locally for given git tags or commits. It is a
replacement for when CI runners are not available. See the
[README](./runner/README.md) for more information.