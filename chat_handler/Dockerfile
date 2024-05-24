FROM amazonlinux:2

RUN yup -y install python38

RUN python3.10 -m ensurepip && \ 
  pip3 install --upgrade pip

COPY lambda_function.py /var/task/

COPY requirements.txt /var/task/
RUN pip3 install -r /var/task/requirements.txt -t /var/task/

CMD ["lambda_function.lambda_handler"]