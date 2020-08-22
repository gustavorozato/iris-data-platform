FROM python:3.7-slim-stretch

COPY requirements.txt train.py tmp/

RUN pip3 install -r tmp/requirements.txt
RUN python3 tmp/train.py

WORKDIR /usr

COPY server.py .

EXPOSE 5000
ENV FLASK_APP=server.py

ENTRYPOINT ["flask", "run", "--host", "0.0.0.0"]