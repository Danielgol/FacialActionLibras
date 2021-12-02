# Docker

## Execution

1. Gere uma imagem a partir do arquivo ```FacialActionLibras/FacialActionLibras.Dockerfile```

2. Abra um terminal e deixe o serviço abaixo executando:

**Obs.:** Substitua o nopme da imagem do comando abaixo ```facialactionlibras:0.0.10```, para o nome da imagem que foi criada no passo anterior.

~~~bash

x11docker --hostipc --hostdisplay --webcam --share /home/jms2/Documentos/projetos/libras/FacialActionLibras facialactionlibras:0.0.10 bash

~~~

3. Em outro bash, liste o último o container *x11docker* gerado pelo step 2 e copie o nome do mesmo: 

~~~bash
docker ps -a
~~~

4. Altere o comando abaixo, substituindo o ```x11docker_X0_facialactionlibras-0-0-10-bash_43919746006``` pelo hash da imagem gerada:

~~~bash
docker exec -it x11docker_X0_facialactionlibras-0-0-10-bash_39045809551 bash
~~~

## Problems and Solutions

**Referência:**

https://github.com/mviereck/x11docker#webcam


## Utils

Para listar imagens docker:
```sudo docker images```

Para apagar imagem docker:
```sudo docker rmi 364b083b3311 -f```
ou
```sudo docker rmi pachyderm/opencv:latest```
