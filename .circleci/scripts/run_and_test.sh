PYTEST_MARK=${1}
PYTHON_VERSION=${2}
ENV_SPECIFICATION=${3}  # which conda environment to run in
OUTPUT_XML=${4}    # the name of the coverage xml file (not its full path)

echo "mark: "${PYTEST_MARK}
echo "version: "${PYTHON_VERSION}
echo "env: "${ENV_SPECIFICATION}
echo "xml: "${OUTPUT_XML}

set -e
export COVERAGE_FILE=/tmp/.coverage_${ENV_SPECIFICATION}_${PYTHON_VERSION}
echo "COVERAGE_FILE set to "${COVERAGE_FILE}
cd /repos/ophys_etl/
/envs/${ENV_SPECIFICATION}/bin/coverage run --rcfile .circleci/coveragerc_file --concurrency=multiprocessing -m \
    pytest --verbose -s -m "${PYTEST_MARK}" \
    tests

/envs/${ENV_SPECIFICATION}/bin/coverage combine --data-file=${COVERAGE_FILE} --rcfile .circleci/coveragerc_file

# coverage automatically adds random salt to the end of the
# coverage file name; need to detect the name of the file
# that was actually produced
coverage_file_list=($(ls /tmp/.coverage_${ENV_SPECIFICATION}_${PYTHON_VERSION}*))
echo "after combining "
echo ${coverage_file_list[@]}

/envs/${ENV_SPECIFICATION}/bin/coverage xml --data-file=${COVERAGE_FILE} \
-o /coverage_outputs_${PYTHON_VERSION}/${OUTPUT_XML} \
--rcfile .circleci/coveragerc_file
