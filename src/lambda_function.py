def lambda_handler(event, context):
  try:
    return {
      'statusCode': 200,
      'body': 'Hello world! -from Compost Chatbot'
    }
  except Exception as e:
    return {
      'statusCode': 500,
      'body': str(e)
    }