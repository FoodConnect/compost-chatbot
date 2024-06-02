import boto3
from pathlib import Path
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import faiss


def lambda_handler(event, context):
  def query_documents_by_status(status):
    dynamodb = boto3.client('dynamodb')
    response = dynamodb.query(
      TableName = 'DocumentMetadata',
      IndexName = 'status-index',
      KeyConditionExpression = 'status = :status',
      ExpressionAttributeValues={':status': {'S': status}}
    )
  
    return response.get('Items', [])

  documents = query_documents_by_status('pending')

  def split_document(document, documentId, title):
    chunk_size = 512
    chunk_overlap = 32
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    return [Document(page_content=chunk, metadata={"documentId": documentId, "title": title})
            for chunk in text_splitter.split(document)]

  split_documents = [split_document(document["text"]["S"], document["documentId"]["S"], document["title"]["S"]) for document in documents]

  embeddings_model = OpenAIEmbeddings(
    client = None, model = "text-embedding-3-small"
  )
  db = FAISS.from_documents(split_documents, embeddings_model)

  file_path = "/tmp/"
  Path(file_path).mkdir(parents=True, exist_ok=True)

  base_file_name = "faiss_index"

  db.save_local(index_name=f"{base_file_name}.faiss", folder_path=file_path)
  db.save_local(index_name=f"{base_file_name}.pkl", folder_path=file_path)

  s3 = boto3.client("s3")

  faiss_file_path = f"indices/{base_file_name}.faiss"  
  pkl_file_path = f"indices/{base_file_name}.pkl"

  s3.upload_file(
      Filename=f"{file_path}{base_file_name}.faiss",
      Bucket="compost-chatbot-bucket",     
      Key=faiss_file_path                  
  )

  s3.upload_file(
      Filename=f"{file_path}{base_file_name}.pkl",
      Bucket="compost-chatbot-bucket",
      Key=pkl_file_path
  )


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