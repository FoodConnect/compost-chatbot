FROM public.ecr.aws/lambda/python:3.10

WORKDIR /var/task

RUN yum -y update && \
  yum -y install gcc make wget && \
  cd /usr/local/src && \
  wget https://www.sqlite.org/2022/sqlite-autoconf-3390500.tar.gz && \
  tar xzf sqlite-autoconf-3390500.tar.gz && \
  cd sqlite-autoconf-3390500 && \
  ./configure && \
  make && \
  make install && \
  yum -y remove gcc make wget && \
  yum clean all
ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"
ENV PATH="/usr/local/bin:$PATH"

COPY chat_handler/lambda_function.py .
COPY chat_handler/requirements.txt .
COPY illinois-residential-food-scrap-composting.pdf .

RUN /var/lang/bin/python3.10 -m pip install -r requirements.txt

CMD ["lambda_function.lambda_handler"]