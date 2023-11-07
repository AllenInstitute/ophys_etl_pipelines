# Need to pass default value due to
# https://github.com/apache/airflow/issues/28940
# Unfortunately it is impossible to remove this bogus default value
# since airflow has a bug
# TODO once fixed, remove default
INT_PARAM_DEFAULT_VALUE = -999
