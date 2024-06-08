from pathlib import Path
import boto3
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def lambda_handler(event, context):
  # Fetch documents by a UserId from DyanmoDB
  dynamodb = boto3.client("dynamodb")
  filter_expression = "userId = :value"
  expression_attribute_values = {":value": {"S": userId}}

  result = dynamodb.query(
      TableName="<DOCUMENT_TABLE_NAME>",
      IndexName="userId-index",
      KeyConditionExpression=filter_expression,
      ExpressionAttributeValues=expression_attribute_values,
  )
  documents = result.get("Items", [])

  # Split fetched documents
  split_documents = []
  for document in documents:
      content = document["content"]["S"]
      documentId = document["id"]["S"]
      title = document["title"]["S"]
      split_documents.extend(split_document(content, documentId, title))

  def split_document(document, documentId, title):
      chunk_size = 512
      chunk_overlap = 32
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=chunk_size, chunk_overlap=chunk_overlap
      )

      documents = []
      for splitted_document in md_header_splits:
          document = Document(
              page_content=splitted_document.page_content,
              metadata={
                  "documentId": documentId,
                  "title": title,
              },
          )
          documents.append(document)

      return documents

  embeddings_model = OpenAIEmbeddings(
      client=None, model="text-embedding-ada-002"
  )
  db = FAISS.from_documents(split_documents, embeddings_model)

  file_path = f"/tmp/"
  Path(file_path).mkdir(parents=True, exist_ok=True)
  file_name = "faiss_index.bin"
  db.save_local(index_name=file_name, folder_path=file_path)
  s3 = boto3.client("s3")
  s3.upload_file(Filename=file_path + "/" + file_name + ".faiss", Bucket="<BUCKET_NAME>", Key="my_faiss.faiss")
  s3.upload_file(Filename=file_path + "/" + file_name + ".pkl", Bucket="<BUCKET_NAME>", Key="my_faiss.pkl")

  document_vectors = get_vectors_from_faiss_index(
      split_documents, db, db.index_to_docstore_id
  )

  def get_vectors_from_faiss_index(documents, db, index_to_docstore_id):
    document_vectors = {}
    for i in range(len(documents)):
        document = db.docstore.search(index_to_docstore_id[i])
        if document:
            if document.metadata["documentId"] not in document_vectors:
                document_vectors[document.metadata["documentId"]] = [
                    index_to_docstore_id[i]
                ]
            else:
                document_vectors[document.metadata["documentId"]].append(
                    index_to_docstore_id[i]
                )
    return document_vectors

  for documentId, vectors in document_vectors.items():
    newItem = {"id": {}, "vectors": {}}
    newItem["id"]["S"] = documentId
    newItem["vectors"]["SS"] = vectors
    dynamodb.put_item(TableName="<VECTOR_TABLE_NAME>", Item=newItem)

  s3.download_file(Bucket="<BUCKET_NAME>", Key="my_faiss.faiss", Filename=file_path)
  s3.download_file(Bucket="<BUCKET_NAME>", Key="my_faiss.pkl", Filename=file_path)

  db = FAISS.load_local(
      index_name="my_faiss",
      folder_path=file_path,
      embeddings=embeddings_model,
  )

  vectors = dynamodb.get_item(
      TableName="<VECTOR_TABLE_NAME>",
      Key={"id": {"S": documentId}},
  )

  if "Item" in vectors:
      vectors = vectors["Item"]["vectors"]["SS"]
      db.delete(vectors)

  content = document["content"]["S"]
  documentId = document["id"]["S"]
  title = document["title"]["S"]
  document_splits = split_document(content, documentId, title)
  document_vectors = get_vectors_from_faiss_index(
      document_splits, db, db.index_to_docstore_id
  )
  db.save_local(index_name=file_name, folder_path=file_path)
  s3.upload_file(Filename=file_path + "/" + file_name + ".faiss", Bucket="<BUCKET_NAME>", Key="my_faiss.faiss")
  s3.upload_file(Filename=file_path + "/" + file_name + ".pkl", Bucket="<BUCKET_NAME>", Key="my_faiss.pkl")
  add_vectors_to_dynamodb(documentId, document_vectors[documentId])