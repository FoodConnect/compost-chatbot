FROM amazonlinux:2

# Install Python and pip
RUN yum -y update && \
  yum -y install python3 python3-pip && \
  yum clean all

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set up the working directory
WORKDIR /var/task

COPY chat_handler/lambda_function.py .
COPY chat_handler/requirements.txt .
RUN python3 -m pip install -r requirements.txt -t .

CMD ["lambda_function.lambda_handler"]
