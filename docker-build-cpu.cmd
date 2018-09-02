@echo OFF

echo Building Docker image...
docker build -f Dockerfile.CPU -t tf-img .

echo Running Docker image...
docker run --rm -it tf-img