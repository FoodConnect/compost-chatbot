FROM public.ecr.aws/lambda/python:3.10

WORKDIR /var/task

COPY data_ingestion_handler/lambda_function.py .
COPY data_ingestion_handler/requirements.txt .

RUN /var/lang/bin/python3.10 -m pip install -r requirements.txt

CMD ["lambda_function.lambda_handler"]