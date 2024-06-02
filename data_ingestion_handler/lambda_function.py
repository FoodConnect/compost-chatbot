import json
import boto3
import fitz


def lambda_handler(event, context):
  try:
    s3_client = boto3.client('s3')
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('DocumentMetadata')

    bucket_name = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    file_content = response['Body'].read()

    document = fitz.open("pdf", file_content)
    text = ""
    for page in document:
          text += page.get_text()

    response = table.put_item(
          Item={
              'documentId': key,
              'text': text,
              'title': 'Extracted from PDF',  
          }
      )
    return {
          'statusCode': 200,
          'body': json.dumps('Text successfully extracted and stored.')
      }
  except Exception as e:
    return {
      'statusCode': 500,
      'body': str(e)
    }