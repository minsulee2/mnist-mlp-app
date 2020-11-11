FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev

RUN pip install keras numpy pillow flask

COPY . .

EXPOSE 80
ENTRYPOINT ["python"]
CMD ["server.py"]
