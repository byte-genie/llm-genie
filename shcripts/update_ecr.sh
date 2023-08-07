TAG="llm-genie"
REPO="017759265015.dkr.ecr.us-west-2.amazonaws.com"
docker build -t $TAG .
echo "finished container build: pushing to ECR"
#aws ecr get-login-password | docker login --username AWS --password-stdin $REPO
aws ecr get-login-password --profile default --region us-west-2 | docker login --username AWS --password-stdin $REPO
docker tag $TAG:latest "$REPO/$TAG:latest"
docker push "$REPO/$TAG:latest"