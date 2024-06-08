import boto3
from pathlib import Path
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# This function synchronizes documents stored in DynamoDB with a FAISS vector index. It splits the documents into smaller chunks, generates embeddings, creates a FAISS index, stores the index in S3, and keeps track of vector IDs in DynamoDB. It also includes logic to reload and update the vectors in the FAISS index if the documents are updated.


def lambda_handler(event, context):
  try:

    dynamodb = boto3.client('dynamodb')

    def query_documents_by_status(status):
      response = dynamodb.query(
        TableName = 'DocumentMetadata',
        IndexName = 'status-index',
        KeyConditionExpression = '#s = :status',
        ExpressionAttributeNames={'#s': 'status'},
        ExpressionAttributeValues={':status': {'S': status}}
      )
      return response.get('Items', [])

    def split_document(document, documentId, title):
      chunk_size = 512
      chunk_overlap = 32
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

      return [Document(page_content=chunk, metadata={"documentId": documentId, "title": title})
              for chunk in text_splitter.split(document)]
              
    def store_vector_ids(document_id, vector_ids):
      table_name = "VectorMetadata"
      dynamodb.put_item(
        TableName = table_name,
        Item = {
          'documentId': {'S': document_id},
          'vectorIds': {'SS': vector_ids}
        }
      )

    def get_vectors_from_faiss_index(documents, db, index_to_docstore_id):
      document_vectors = {}
      for i in range(len(documents)):
        document = db.docstore.search(index_to_docstore_id[i])
        if document:
          if document.metadata["documentId"] not in document_vectors:
            document_vectors[document.metadata["documentId"]] = [index_to_docstore_id[i]]
          else:
            document_vectors[document.metadata["documentId"]].append(index_to_docstore_id[i])
      return document_vectors


    # Query documents with 'pending' status
    documents = query_documents_by_status('pending')

    # Split the documents into chunks
    split_documents = [split_document(document["text"]["S"], document["documentId"]["S"], document["title"]["S"]) for document in documents]

    embeddings_model = OpenAIEmbeddings(
      client = None, model = "text-embedding-3-small"
    )

    # Create a FAISS index from the split documents
    db = FAISS.from_documents(split_documents, embeddings_model)

    # Save the FAISS index locally
    file_path = "/tmp/"
    Path(file_path).mkdir(parents=True, exist_ok=True)
    base_file_name = "faiss_index"

    db.save_local(index_name=f"{base_file_name}.faiss", folder_path=file_path)
    db.save_local(index_name=f"{base_file_name}.pkl", folder_path=file_path)


    # Upload the FAISS index files to S3
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

    # Retrieve vectors from the FAISS index and store their IDs in DynamoDB
    document_vectors = get_vectors_from_faiss_index(split_documents, db, db.index_to_docstore_id)
    for documentId, vectors in document_vectors.items():
      store_vector_ids(documentId, vectors)

     # Reload the FAISS index from S3
    file_path = "/tmp/"
    base_file_name = "faiss_index"
    s3.download_file('compost-chatbot-bucket', f"indices/{base_file_name}.faiss", f"{file_path}{base_file_name}.faiss")
    s3.download_file('compost-chatbot-bucket', f"indices/{base_file_name}.pkl", f"{file_path}{base_file_name}.pkl")

    db = FAISS.load_local(
          index_name="my_faiss",
          folder_path=file_path,
          embeddings=embeddings_model,
      )
    
    # Update or delete vectors as needed
    for document in documents:
          documentId = document["documentId"]["S"]
          vectors = dynamodb.get_item(
              TableName="VectorMetadata",
              Key={"documentId": {"S": documentId}},
          )

          if "Item" in vectors:
              vectors = vectors["Item"]["vectorIds"]["SS"]
              db.delete(vectors)

          content = document["text"]["S"]
          title = document["title"]["S"]
          document_splits = split_document(content, documentId, title)
          document_vectors = get_vectors_from_faiss_index(document_splits, db, db.index_to_docstore_id)

          db.save_local(index_name=base_file_name, folder_path=file_path)
          s3.upload_file(Filename=f"{file_path}/{base_file_name}.faiss", Bucket="compost-chatbot-bucket", Key=faiss_file_path)
          s3.upload_file(Filename=f"{file_path}/{base_file_name}.pkl", Bucket="compost-chatbot-bucket", Key=pkl_file_path)

          store_vector_ids(documentId, document_vectors[documentId])
    

    response_message = {
            'status': 'success',
            'message': 'Processed documents and updated FAISS index.',
            'documents_processed': len(documents),
            'vectors_stored': sum(len(vectors) for vectors in document_vectors.values())
        }
    
    return {
        'statusCode': 200,
        'body': response_message
      }
  except Exception as e:
    return {
      'statusCode': 500,
      'body': str(e)
    }
  

