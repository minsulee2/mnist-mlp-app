FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
RUN pip3 install flask pillow opencv-python keras numpy pillow

COPY . .

EXPOSE 80
ENTRYPOINT ["python"]
CMD ["server.py"]