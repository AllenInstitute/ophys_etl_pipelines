set -e
export COVERAGE_FILE=/tmp/.coverage
cd /repos/ophys_etl/
/envs/ophys_etl/bin/python -m \
    pytest -m "not event_detect_only" \
    --cov-report xml:/coverage_outputs/general_cov.xml \
    --cov ophys_etl \
    tests
