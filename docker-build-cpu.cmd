@echo OFF

echo Building Docker image...
docker build -f Dockerfile.CPU -t tf-cpu-img .

echo Running Docker image...
docker run ^
    -p 127.0.0.1:6007:6007 ^
    --rm ^
    -it ^
    tf-cpu-img 