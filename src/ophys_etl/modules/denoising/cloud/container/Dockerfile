# using the image from https://github.com/aws/deep-learning-containers/blob/master/available_images.md
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.5.0-gpu-py37-cu112-ubuntu18.04

RUN pip install git+https://github.com/danielsf/deepinterpolation.git@bb061ab7fdc0986a27d8b8469e54661f9fc7a278

ENV PATH="/opt/ml/code:${PATH}"

RUN mkdir -p /opt/ml/code
COPY /run_finetuning.py /opt/ml/code
COPY /input.json /opt/ml/
COPY /training.json /opt/ml
COPY /validation.json /opt/ml
COPY /pretrained_model.h5 /opt/ml/
WORKDIR /opt/ml/code

ENTRYPOINT ["python", "run_finetuning.py"]