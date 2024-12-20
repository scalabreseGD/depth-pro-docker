# Apple Depth-pro Serving as Docker Rest Endpoint

## Run Local
Follow the instructions in [ML-DEPTH-PRO](https://github.com/apple/ml-depth-pro/tree/main)

## Before starting
<b>Build the model with GPU requires nvidia base image. Follow the instructions before in order to login into nvcr.io</b> 

<ol>
<li>Be sure you have Docker installed and buildkit available</li> 
<li>Connect to [NVIDIA NGC](https://org.ngc.nvidia.com/setup/personal-keys)</li>
<li>Create a personal key</li>
<li>Login into nvcr.io</li>
</ol>

<b>REMEMBER TO DOWNLOAD THE CHECKPOINTS UNDER app/checkpoints</b>
## Build and Run
### Version
```shell
export version=x.y.z
```
### GPU
```shell
bash install.sh depth-pro gpu registry
```

```shell
docker run --gpus all -p 8080:8080 -e PORT=8080 -v ./huggingface:/root/.cache/huggingface/ depth-pro-gpu:${version}
```

### CPU
```shell
bash install.sh  sam-2 cpu registry
```
```shell
docker run -p 8080:8080 -e PORT=8080 -v ./huggingface:/root/.cache/huggingface/ depth-pro-cpu:${version}
```
