import os
import resource
import boto3
from pathlib import Path
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import logging

# This function synchronizes documents stored in DynamoDB with a FAISS vector index. It splits the documents into smaller chunks, generates embeddings, creates a FAISS index, stores the index in S3, and keeps track of vector IDs in DynamoDB. It also includes logic to reload and update the vectors in the FAISS index if the documents are updated.

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
  try:

    logger.info(f"Lambda function memory size: {os.environ['AWS_LAMBDA_FUNCTION_MEMORY_SIZE']} MB")
    logger.info(f"Lambda log group name: {os.environ['AWS_LAMBDA_LOG_GROUP_NAME']}")

    dynamodb = boto3.client('dynamodb')

    def query_documents_by_status(status):
      logger.info("Querying documents by status")
      response = dynamodb.query(
        TableName = 'DocumentMetadata',
        IndexName = 'status-index',
        KeyConditionExpression = '#s = :status',
        ExpressionAttributeNames={'#s': 'status'},
        ExpressionAttributeValues={':status': {'S': status}}
      )
      logger.info(f"Documents with status '{status}': {response.get('Items', [])}")
      return response.get('Items', [])

    def split_document(document, documentId, title):
      logger.info(f"Splitting document {documentId}")
      chunk_size = 512
      chunk_overlap = 32
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
      documents = [Document(page_content=chunk, metadata={"documentId": documentId, "title": title})
              for chunk in text_splitter.split_text(document)]
      logger.info(f"Split {documentId} into {len(documents)} chunks")
      return documents
              
    def store_vector_ids(document_id, vector_ids):
      logger.info(f"Storing vector IDs for document {document_id}")
      table_name = "VectorMetadata"
      dynamodb.put_item(
        TableName = table_name,
        Item = {
          'documentId': {'S': document_id},
          'vectorIds': {'SS': vector_ids}
        }
      )

    def get_vectors_from_faiss_index(documents, db, index_to_docstore_id):
      logger.info("Getting vectors from FAISS index")
      document_vectors = {}
      for i in range(len(documents)):
        document = db.docstore.search(index_to_docstore_id[i])
        if document:
          if document.metadata["documentId"] not in document_vectors:
            document_vectors[document.metadata["documentId"]] = [index_to_docstore_id[i]]
          else:
            document_vectors[document.metadata["documentId"]].append(index_to_docstore_id[i])
      logger.info(f"Retrieved vector IDs from FAISS index")
      return document_vectors


    # Query documents with 'pending' status
    documents = query_documents_by_status('pending')

    if not documents:
      logger.info("No pending documents found")
      return {
        'statusCode': 200,
        'body': 'No pending documents found.'
      }

    # Split the documents into chunks
    all_split_documents = []
    for document in documents:
      all_split_documents.extend(split_document(document["text"]["S"], document["documentId"]["S"], document["title"]["S"]))
    logger.info("Documents split into chunks")

    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
      raise ValueError("OPENAI_API_KEY environment variable is not set")
    embeddings_model = OpenAIEmbeddings(
      client = None, model = "text-embedding-3-small"
    )

    # Create a FAISS index from the split documents
    db = FAISS.from_documents(all_split_documents, embeddings_model)
    logger.info("FAISS index created")

    # Save the FAISS index locally
    file_path = "/tmp/"
    Path(file_path).mkdir(parents=True, exist_ok=True)
    base_file_name = "faiss_index"

    faiss_local_path = f"{file_path}{base_file_name}.faiss"
    pkl_local_path = f"{file_path}{base_file_name}.pkl"

    logger.info(f"Saving FAISS index to {faiss_local_path} and {pkl_local_path}")

    db.save_local(index_name=base_file_name, folder_path=file_path)
    logger.info("FAISS index saved locally")

    if not os.path.exists(faiss_local_path):
            raise FileNotFoundError(f"FAISS file {faiss_local_path} not found")
    if not os.path.exists(pkl_local_path):
        raise FileNotFoundError(f"PKL file {pkl_local_path} not found")


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
    logger.info("FAISS index uploaded to S3")

    # Retrieve vectors from the FAISS index and store their IDs in DynamoDB
    document_vectors = get_vectors_from_faiss_index(all_split_documents, db, db.index_to_docstore_id)
    for documentId, vectors in document_vectors.items():
      store_vector_ids(documentId, vectors)

     # Reload the FAISS index from S3
    file_path = "/tmp/"
    base_file_name = "faiss_index"
    s3.download_file('compost-chatbot-bucket', f"indices/{base_file_name}.faiss", f"{file_path}{base_file_name}.faiss")
    s3.download_file('compost-chatbot-bucket', f"indices/{base_file_name}.pkl", f"{file_path}{base_file_name}.pkl")
    logger.info("FAISS index downloaded from S3")

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
    
    memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logger.info(f"Memory usage: {memory_usage} KB")
    

    response_message = {
            'status': 'success',
            'message': 'Processed documents and updated FAISS index.',
            'documents_processed': len(documents),
            'vectors_stored': sum(len(vectors) for vectors in document_vectors.values()),
            'memory_usage': memory_usage
        }
    
    logger.info(f"Function completed successfully: {response_message}")
    
    return {
        'statusCode': 200,
        'body': response_message
      }
  except Exception as e:
    logger.error(f"Error: {str(e)}")
    return {
      'statusCode': 500,
      'body': str(e)
    }
  

