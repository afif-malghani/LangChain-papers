FROM python:3.11.4-slim-bullseye

WORKDIR /python-docker

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

RUN python3 downloadModel.py

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]