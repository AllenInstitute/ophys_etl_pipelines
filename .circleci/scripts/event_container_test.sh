set -e
export COVERAGE_FILE=/tmp/.coverage
cd /repos/ophys_etl/
/envs/event_detection/bin/python -m \
    pytest -m "event_detect_only" \
    --cov-report xml:/coverage_outputs/event_cov.xml \
    --cov ophys_etl \
    tests
