1. To compile a Dockerfile into an image:

```
docker build -t [IMAGE] -f [FILENAME] .
```

2. To push the Dockerfile to a hub.

```
sudo docker push [IMAGE]
```

3. To run a Docker image and turn it into a container

```
docker run -it harbor.xaminim.com/minimax-dialogue/verl-tool
```

If you have gpus, use


```
docker run -it --gpus all harbor.xaminim.com/minimax-dialogue/verl-tool
```

If you have want to mount disk


```
docker run -it --gpus all -v [SRC_DISK]:[DOCKER_DISK] harbor.xaminim.com/minimax-dialogue/verl-tool
```