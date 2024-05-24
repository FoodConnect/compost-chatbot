FROM public.ecr.aws/lambda/python:3.10

WORKDIR /var/task

COPY chat_handler/lambda_function.py .
COPY chat_handler/requirements.txt .

RUN python3.10 -m pip install -r requirements.txt

COPY chat_handler/lambda-entrypoint.sh .
RUN chmod +x /lambda-entrypoint.sh

CMD ["/lambda-entrypoint.sh", "lambda_function.lambda_handler"]
