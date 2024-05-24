FROM amazonlinux:2

# Install dependencies for building Python 3.10
RUN yum -y update && \
  yum -y groupinstall "Development Tools" && \
  yum -y install openssl-devel bzip2-devel libffi-devel zlib-devel wget make

# Download and install Python 3.10
RUN cd /usr/src && \
  wget https://www.python.org/ftp/python/3.10.2/Python-3.10.2.tgz && \
  tar xzf Python-3.10.2.tgz && \
  cd Python-3.10.2 && \
  ./configure --with-openssl=/usr/include/openssl --enable-optimizations && \
  make altinstall

# Install pip for Python 3.10
RUN /usr/local/bin/python3.10 -m ensurepip && \
  /usr/local/bin/python3.10 -m pip install --upgrade pip

# Print the Python version and pip version for debugging
RUN /usr/local/bin/python3.10 --version && \
  /usr/local/bin/python3.10 -m pip --version

COPY chat_handler/lambda_function.py /var/task/
COPY chat_handler/requirements.txt /var/task/
RUN /usr/local/bin/python3.10 -m pip install -r /var/task/requirements.txt -t /var/task/ --no-cache-dir
CMD ["lambda_function.lambda_handler"]