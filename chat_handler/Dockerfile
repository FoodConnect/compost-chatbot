FROM amazonlinux:2

# Install Dependencies for Python 3.10
RUN yum -y update && \
  yum -y install gcc openssl-devel bzip2-devel libffi-devel zlib-devel make

# Install Python 3.10
RUN cd /usr/src && \
  curl -O https://www.python.org/ftp/python/3.10.2/Python-3.10.2.tgz && \
  tar xzf Python-3.10.2.tgz && \
  cd Python-3.10.2 && \
  ./configure --enable-optimizations && \
  make altinstall

# Install pip for Python 3.10
RUN /usr/local/bin/python3.10 -m ensurepip && \
  /usr/local/bin/python3.10 -m pip install --upgrade pip

COPY chat_handler/lambda_function.py /var/task/
COPY chat_handler/requirements.txt /var/task/

RUN pip3 install -r /var/task/requirements.txt -t /var/task/
CMD ["lambda_function.lambda_handler"]