set -e
export COVERAGE_FILE=/tmp/.coverage
cd /repos/ophys_etl/
/envs/suite2p/bin/python -m \
    pytest -m "suite2p_only or suite2p_also" \
    --ignore=/repos/ophys_etl/tests/modules/event_detection \
    --cov-report xml:/coverage_outputs/suite2p_cov.xml \
    --cov ophys_etl \
    tests
