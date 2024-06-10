import os
import json
import boto3
from pathlib import Path
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS


def lambda_handler(event, context):
  
  def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
    
  store = {}

  try:
    file_path = f"/tmp/"
    base_file_name = "faiss_index"
    Path(file_path).mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")
    s3.download_file('compost-chatbot-bucket', f"indices/{base_file_name}.faiss", f"{file_path}{base_file_name}.faiss")
    s3.download_file('compost-chatbot-bucket', f"indices/{base_file_name}.pkl", f"{file_path}{base_file_name}.pkl")

    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
      raise ValueError("OPENAI_API_KEY environment variable is not set")
    embeddings_model = OpenAIEmbeddings(
      client = None, model = "text-embedding-3-small"
    )

    db = FAISS.load_local(
            index_name=base_file_name,
            folder_path=file_path,
            embeddings=embeddings_model,
            allow_dangerous_deserialization=True,
        )

    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 8})

    contextualize_q_system_prompt = (
              "Given a chat history and the latest user question "
              "which might reference context in the chat history, "
              "formulate a standalone question which can be understood "
              "without the chat history. Do NOT answer the question, "
              "just reformulate it if needed and otherwise return it as is."
          )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
              [
                  ("system", contextualize_q_system_prompt),
                  MessagesPlaceholder("chat_history"),
                  ("human", "{input}"),
              ]
          )

    history_aware_retriever = create_history_aware_retriever(ChatOpenAI(model="gpt-3.5-turbo"), retriever, contextualize_q_prompt)

    system_prompt = (
              "You are an assistant for question-answering tasks. "
              "Use the following pieces of retrieved context to answer "
              "the question. If you don't know the answer, say that you "
              "don't know. Use three sentences maximum and keep the "
              "answer concise."
              "\n\n"
              "{context}" 
          )
    
    qa_prompt = ChatPromptTemplate.from_messages(
              [
                  ("system", system_prompt),
                  MessagesPlaceholder("chat_history"),
                  ("human", "{input}"),
              ]
          )

    question_answer_chain = create_stuff_documents_chain(ChatOpenAI(model="gpt-3.5-turbo"), qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    conversational_rag_chain = RunnableWithMessageHistory(
              rag_chain,
              get_session_history,
              input_messages_key="input",
              history_messages_key="chat_history",
              output_messages_key="answer",
          )
    
    body = json.loads(event['body'])
    query = body.get('query')
    session_id = body.get('session_id', 'default_session')

    if not query:
      return {
        'statusCode': 400,
        'body': json.dumps({'error': 'No query was provided'})
      }
    
    result = conversational_rag_chain.invoke(
      {"input": query},
      config={"configurable": {"session_id": session_id}}
    )

  
    return {
        'statusCode': 200,
        'body': json.dumps({'response': result["answer"]})
    }
  except Exception as e:
    return {
      'statusCode': 500,
      'body': str(e)
    }