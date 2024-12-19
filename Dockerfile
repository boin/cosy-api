FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get -y install git curl ffmpeg wget vim locales apt-utils libaio-dev
RUN locale-gen en_US en_US.UTF-8

#mount requirements.txt for best cache result
RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    pip3 install --no-cache-dir -r /tmp/requirements.txt
RUN pip3 install -U numpy==1.26.4 pyloudnorm

COPY . /opt/CosyVoice
WORKDIR /opt/CosyVoice
RUN git submodule update --init --recursive
#RUN mkdir -p logs
#RUN echo '#!/bin/bash\npython3 -u webui_auto.py 2>&1 | stdbuf -oL -eL tee -i logs/auto.log &\n  python3 -u webui_train.py 2>&1 | stdbuf -oL -eL tee -i logs/train.log ' >> /opt/CosyVoice/service.sh
#RUN chmod u+x /opt/CosyVoice/service.sh
#CMD ["sleep","infinity"]
CMD [ "python3", "api.py" ]