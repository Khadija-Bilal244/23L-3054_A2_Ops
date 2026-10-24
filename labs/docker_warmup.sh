#!/usr/bin/env bash
# Task 6.1 - practise with a ready-made image (nginx). Screenshot every step.
docker pull nginx:alpine
docker run -d --name web1 -p 8080:80 nginx:alpine          # port mapping host 8080 -> container 80
curl -s localhost:8080 | head -n 5
mkdir -p site && echo "<h1>hello volume</h1>" > site/index.html
docker run -d --name web2 -p 8081:80 -v "$PWD/site:/usr/share/nginx/html:ro" nginx:alpine   # volume
curl -s localhost:8081
docker logs web2                                           # read logs
docker exec -it web2 sh                                    # shell inside (type 'exit')
docker ps -a                                               # list containers
docker images                                              # list images
docker stop web1 web2 && docker rm web1 web2               # remove containers
docker rmi nginx:alpine                                    # remove image
