import boto3
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
import faiss


def lambda_handler(event, context):
  try:
    return {
      'statusCode': 200,
      'body': 'Interaction successful!'
    }
  except Exception as e:
    return {
      'statusCode': 500,
      'body': str(e)
    }