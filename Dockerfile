FROM nvidia/cuda:11.1-base-ubuntu20.04

# Needed to disable interactive configuration by tzdata.
RUN ln -fs /usr/share/zoneinfo/Canada/Mountain /etc/localtime

WORKDIR /

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.7

RUN apt-get update && apt-get install -y \
  git \
  cmake \
  build-essential \
  libgl1-mesa-dev \
  libsdl2-dev \
  libsdl2-image-dev \
  libsdl2-ttf-dev \
  libsdl2-gfx-dev \
  libboost-all-dev \
  libdirectfb-dev \
  libst-dev \
  mesa-utils \
  xvfb \
  x11vnc \
  libsdl-sge-dev \
  python3-pip \
  libboost-all-dev \
  libboost-python-dev \
  tmux

RUN apt-get install -y \
    python-dev \
    libsdl-image1.2-dev \
    libsdl-mixer1.2-dev \
    libsdl-ttf2.0-dev   \
    libsdl1.2-dev \
    libsmpeg-dev \
    python-numpy \
    subversion \
    libportmidi-dev \
    ffmpeg \
    libswscale-dev \
    libavformat-dev \
    libavcodec-dev


# Install kaggle environments
RUN mkdir /gfootball
WORKDIR /gfootball
COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt -I

# Installs necessary dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
         wget \
         curl \
         python-dev && \
     rm -rf /var/lib/apt/lists/*

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

COPY . /gfootball/
WORKDIR /gfootball/

# Install any new scenarios if required
WORKDIR /gfootball/kaggle-environments/football
RUN pip3 install .

# Install the RL parallel experiment engine
WORKDIR /gfootball/rlpyt
RUN pip3 install .

WORKDIR /gfootball

EXPOSE 6006

ENTRYPOINT ["python3", "run_rl.py", "--cloud", "True", "--cloud_bucket", "gs://kagglefootball-aiplatform"]