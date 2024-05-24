import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma


def lambda_handler(event, context):
  body = json.loads(event.get("body", "{}"))

  pdf_file_name = "illinois-residential-food-scrap-composting.pdf"
  pdf_path = os.path.join("/var/task", pdf_file_name)
  documents = []
  loader = PyPDFLoader(pdf_path)
  documents.extend(loader.load())

  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
  chunked_documents = text_splitter.split_documents(documents)
  vectordb = Chroma.from_documents(chunked_documents, OpenAIEmbeddings())
  
  question = body.get("question")
  if question is None:
    return {
      'statusCode': 400,
      'body': json.dups({'error': 'No question was provided'})
    }
  
  docs = vectordb.similarity_search(documents[0].page_content)
  vectordb.persist()
  
  chatbot = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
      openai_api_key=os.getenv("OPENAI_API_KEY"),
      temperature=0, model_name="gpt-3.5-turbo", max_tokens=100
    ),
    chain_type="stuff",
    retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k":1}),
    )
  
  template ="""
  respond as succinctly as possible. {query}?
  """

  prompt = PromptTemplate(
    input_variables=["query"],
    template=template,
  )

  response = chatbot.invoke(prompt.format(query=question))

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