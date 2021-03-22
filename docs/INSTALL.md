# Installation

### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 18.04/20.04)
* Python 3.8
* PyTorch 1.4.0
* CUDA 10.2

### Install `pcdet v0.3`
NOTE: Please re-install `pcdet v0.3` by running `python setup.py develop` even if you have already installed previous version. We recommend to install inside a virtual environment to avoid conflicts with other projects.

a. Clone this repository.
```shell
git clone https://github.com/TRAILab/CaDDN.git
```

b. Install the dependent libraries as follows:

* Install the dependent python libraries:
```
pip install -r requirements.txt
```

c. Install this `pcdet` library by running the following command:
```shell
python setup.py develop
```

### Docker
Additionally, we provide a docker image for this project. Please refer to [DOCKER.md](../docker/DOCKER.md) for information on how to use the docker image.