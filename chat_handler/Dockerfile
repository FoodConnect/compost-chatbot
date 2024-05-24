FROM amazonlinux:2

RUN yup -y install python3.10

RUN python3.10 -m ensurepip && \ 
  pip3 install --upgrade pip

COPY chat_handler/lambda_function.py /var/task/
COPY chat_handler/requirements.txt /var/task/
RUN pip3 install -r /var/task/requirements.txt -t /var/task/

CMD ["lambda_function.lambda_handler"]