# Docker

## Setup
To use CaDDN in docker please make sure you have `nvidia-docker` installed.
```shell
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Install `nvidia-container-runtime`
```shell
sudo apt install nvidia-container-runtime
```

Edit/create `/etc/docker/daemon.json` with:
```
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```

Restart docker daemon
```shell
sudo systemctl restart docker
```

Navigate to `docker` directory
```shell
cd docker
```

## Get a Docker Image
In order to download the docker image:
```
docker pull codyreading/CaDDN
```

Alternatively, you can build it yourself:
```
./build.sh
```

## Create Docker Container
```
./run.sh
```

If you have symlinks for the KITTI dataset folders, i.e.:
```
CaDDN
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training -> /media/Data/Kitti/object/training
│   │   │── testing -> /media/Data/Kitti/object/testing
├── pcdet
├── tools
```

Run the following:
```
./run.sh --sym
```