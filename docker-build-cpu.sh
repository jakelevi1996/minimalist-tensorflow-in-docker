# Build image

echo Building image...
docker build -f Dockerfile.CPU -t tf-cpu-img .


# Use nvidia-docker to use tensorflow-gpu
# Output saved in subdirectories of a "dout" (docker out) directory:

echo Running image...

docker run \
    -v $(pwd)/dout/models:models \
    -v $(pwd)/dout/results:/results \
    -p 127.0.0.1:6007:6007 \
    --rm \
    -it \
    tf-cpu-img 
