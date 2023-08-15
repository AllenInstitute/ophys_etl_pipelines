PYTEST_MARK=${1}
PYTHON_VERSION=${2}
ENV_SPECIFICATION=${3}  # a name to use for coverage output file
OUTPUT_XML=${4}    # the name of the coverage xml file (not its full path)
TEST_DIRECTORY=${5}
COVERAGE_EXE_PATH=${6:-/envs/${ENV_SPECIFICATION}/bin/coverage}  # path to coverage executable

echo "mark: "${PYTEST_MARK}
echo "version: "${PYTHON_VERSION}
echo "env: "${ENV_SPECIFICATION}
echo "xml: "${OUTPUT_XML}
echo "test directory": ${TEST_DIRECTORY}

set -e
export COVERAGE_FILE=/tmp/.coverage_${ENV_SPECIFICATION}_${PYTHON_VERSION}
echo "COVERAGE_FILE set to "${COVERAGE_FILE}
cd /repos/ophys_etl/
${COVERAGE_EXE_PATH} run --rcfile .circleci/coveragerc_file --concurrency=multiprocessing -m \
    pytest --verbose \
    -s -m "${PYTEST_MARK}" \
    ${TEST_DIRECTORY}

${COVERAGE_EXE_PATH} combine --data-file=${COVERAGE_FILE} --rcfile .circleci/coveragerc_file

# coverage automatically adds random salt to the end of the
# coverage file name; need to detect the name of the file
# that was actually produced
coverage_file_list=($(ls /tmp/.coverage_${ENV_SPECIFICATION}_${PYTHON_VERSION}*))
echo "after combining "
echo ${coverage_file_list[@]}