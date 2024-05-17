import json
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

def lambda_handler(event, context):
  body = json.loads(event.get("body", "{}"))

  question = body.get("question")

  if question is None:
    return {
      'statusCode': 400,
      'body': json.dups({'error': 'No question was provided'})
    }
  
  chatbot = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
      openai_api_key=os.getenv("OPENAI_API_KEY"),
      temperature=0, model_name="gpt-3.5-turbo", max_tokens=100
    ),
    chain_type="stuff",
    )
  template ="""
  respond as succinctly as possible. {query}?
  """

  prompt = PromptTemplate(
    input_variables=["query"],
    template=template,
  )

  response = chatbot.run(prompt.format(query=question))

  try:
    return {
      'statusCode': 200,
      'body': json.dumps(response)
    }
  except Exception as e:
    return {
      'statusCode': 500,
      'body': str(e)
    }