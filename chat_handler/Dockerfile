FROM amazonlinux:2

# Install development tools and SSL dependencies
RUN yum -y update && \
  yum -y groupinstall "Development Tools" && \
  yum -y install openssl-devel bzip2-devel libffi-devel zlib-devel wget make && \
  yum -y install openssl

# Download and compile Python 3.10
RUN cd /usr/src && \
  wget https://www.python.org/ftp/python/3.10.2/Python-3.10.2.tgz && \
  tar xzf Python-3.10.2.tgz && \
  cd Python-3.10.2 && \
  ./configure --enable-optimizations --with-ensurepip=install --with-openssl=/usr/include/openssl && \
  make altinstall

# Upgrade pip
RUN /usr/local/bin/python3.10 -m ensurepip && \
  /usr/local/bin/python3.10 -m pip install --upgrade pip

WORKDIR /var/task
COPY chat_handler/lambda_function.py .
COPY chat_handler/requirements.txt .
RUN /usr/local/bin/python3.10 -m pip install -r requirements.txt -t .
CMD ["lambda_function.lambda_handler"]