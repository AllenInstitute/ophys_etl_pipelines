FROM amazonlinux:latest

RUN yum -y install python3 pip3
RUN pip3 install deepinterpolation

ENV PATH="/opt/ml/code:${PATH}"

## /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
#COPY /cifar10 /opt/ml/code
#WORKDIR /opt/ml/code

