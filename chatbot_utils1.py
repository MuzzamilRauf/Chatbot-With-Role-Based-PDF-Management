import os
import uuid
import base64
import numpy as np
import requests
from typing import List
from langchain.schema import Document
from langchain_core.messages import HumanMessage
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from config import TOGETHER_API_KEY, PINECONE_API_KEY, COHERE_API_KEY

# Set API keys
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["COHERE_API_KEY"] = COHERE_API_KEY


class RAGPipeline:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            openai_api_base="https://api.together.xyz/v1",
            openai_api_key=os.environ["TOGETHER_API_KEY"],
            temperature=0.7,
            max_tokens=1000
        )
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index_name = "dense-index"
        self.index = self.initialize_pinecone()
        self.memory = ConversationBufferMemory(return_messages=True)

    def initialize_pinecone(self):
        if self.index_name not in [i["name"] for i in self.pinecone.list_indexes()]:
            self.pinecone.create_index(
                name=self.index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        return self.pinecone.Index(self.index_name)

    def load_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        return loader.load()

    def split_text(self, documents, chunk_size=250, chunk_overlap=30):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(documents)

    def process_pdf_and_store_embeddings(self, pdf_path, file_id):
        documents = self.load_pdf(pdf_path)
        docs = self.split_text(documents)
        self.store_embeddings_in_pinecone(docs, file_id)

    def store_embeddings_in_pinecone(self, docs: List[Document], file_id: str):
        texts = [doc.page_content for doc in docs]
        doc_embeddings = self.embedding_model.embed_documents(texts)

        vectors = [
            (
                str(uuid.uuid4()),
                np.array(embedding).tolist(),
                {"text": text, "file_id": file_id}
            )
            for embedding, text in zip(doc_embeddings, texts)
        ]

        print(f"Storing {len(vectors)} vectors in Pinecone...")

        for i in range(0, len(vectors), 100):
            self.index.upsert(vectors[i:i + 100])
            print(f"Uploaded batch {i // 100 + 1} of {(len(vectors) - 1) // 100 + 1}")

        print("Embeddings stored in Pinecone successfully!")

    def delete_vectors_by_file_id(self, file_id: str):
        print(f"Searching for vectors with file_id: {file_id}")
        dummy_vector = [0.0] * 768
        query_response = self.index.query(
            vector=dummy_vector,
            filter={"file_id": {"$eq": file_id}},
            top_k=10000,
            include_metadata=False,
            include_values=False
        )

        vector_ids = [match["id"] for match in query_response.get("matches", [])]

        if vector_ids:
            print(f"Deleting {len(vector_ids)} vectors for file_id: {file_id}")
            self.index.delete(ids=vector_ids)
            return {"deleted": len(vector_ids)}
        else:
            print(f"No vectors found for file_id: {file_id}")
            return {"deleted": 0}

    def process_text(self, text):
        return text

    def generate_embedding_of_input(self, text):
        embeddings = self.embedding_model.embed_documents([str(text)])
        return embeddings[0]

    def validate_embedding(self, embedding):
        embedding = np.array(embedding, dtype=np.float32)
        if len(embedding) != 768:
            raise ValueError(f"Embedding dimension mismatch: {len(embedding)}, expected 768")
        embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
        return embedding.tolist()

    # def retrieve_and_generate_response(self, input_data):
    #     query = self.process_text(input_data)
    #     if not query.strip():
    #         raise ValueError("Query is empty after preprocessing")
    #
    #     query_embeddings = self.generate_embedding_of_input(query)
    #     if len(query_embeddings) != 768:
    #         raise ValueError(f"Incorrect embedding size: {len(query_embeddings)}, expected 768")
    #
    #     try:
    #         results = self.index.query(vector=query_embeddings, top_k=5, include_metadata=True)
    #     except Exception as e:
    #         print(f"Pinecone query failed: {str(e)}")
    #         raise
    #
    #     top_matches = sorted(results["matches"], key=lambda x: x["score"], reverse=True)[:3]
    #     context = "\n\n".join([match["metadata"]["text"] for match in top_matches])
    #     top_score = top_matches[0]["score"] if top_matches else 0.0
    #
    #     all_history = self.memory.load_memory_variables({})["history"]
    #     filtered_history = [msg for msg in all_history if isinstance(msg, HumanMessage)][-3:]
    #     history_text = "\n".join([f"User: {msg.content}" for msg in filtered_history])
    #
    #     if filtered_history:
    #         last_user_msg = filtered_history[-1].content
    #         if query.lower().startswith(("what", "how", "why", "where", "when", "who")):
    #             query = f"This is a follow-up. Previously you asked: '{last_user_msg}'. Now: '{query}'"
    #
    #     if top_score > 0.5:
    #         prompt = f"""
    #         You are a friendly, conversational AI assistant powered by a Retrieval-Augmented Generation (RAG) system...
    #         ### Previous Conversation:
    #         {history_text}
    #
    #         ### PDF Context:
    #         {context}
    #
    #         ### User Question:
    #         {query}
    #
    #         ### Your Helpful Answer:
    #         """
    #     else:
    #         prompt = f"""
    #         You are a friendly, conversational AI assistant powered by a Retrieval-Augmented Generation (RAG) system...
    #         ### PDF Context: Not enough relevant information.
    #         ### User Question:
    #         {query}
    #
    #         ### Your Helpful Answer:
    #         Hmm, I don’t have enough info in the context to answer that. Could you share more details or ask something related to the uploaded info?
    #         """
    #
    #     return self.llm.invoke(prompt).content

    # Cohere API key
    # g2YH6JMY3Jg0QqvIDP9K36bPmMpNrE4h8f0pnOY0


    # def retrieve_and_generate_response(self, input_data):
    #     # Step 1: Preprocess query text and generate embeddings
    #     query = self.process_text(input_data)
    #     if not query.strip():
    #         raise ValueError("Query is empty after preprocessing")
    #
    #     query_embeddings = self.generate_embedding_of_input(query)
    #     if len(query_embeddings) != 768:
    #         raise ValueError(f"Incorrect embedding size: {len(query_embeddings)}, expected 768")
    #
    #     # Step 2: Query Pinecone index
    #     try:
    #         results = self.index.query(vector=query_embeddings, top_k=3, include_metadata=True)
    #     except Exception as e:
    #         print(f"Pinecone query failed: {str(e)}")
    #         raise
    #
    #     # Step 3: Filter top 3 context matches by relevance
    #     top_matches = sorted(results["matches"], key=lambda x: x["score"], reverse=True)[:3]
    #     context = "\n\n".join([match["metadata"]["text"] for match in top_matches])
    #     top_score = top_matches[0]["score"] if top_matches else 0.0
    #
    #     # Step 4: Load recent conversation history and keep only user messages
    #     all_history = self.memory.load_memory_variables({})["history"]
    #     filtered_history = [msg for msg in all_history if isinstance(msg, HumanMessage)][-3:]
    #     history_text = "\n".join([f"User: {msg.content}" for msg in filtered_history])
    #
    #     # Step 5: Handle follow-up questions by injecting last user message
    #     if filtered_history:
    #         last_user_msg = filtered_history[-1].content
    #         if query.lower().startswith(("what", "how", "why", "where", "when", "who")):
    #             query = f"This is a follow-up. Previously you asked: '{last_user_msg}'. Now: '{query}'"
    #
    #     # Step 6: Build prompt dynamically based on context relevance
    #     if top_score > 0.4:
    #         prompt = f"""
    #             You are a friendly, conversational AI assistant powered by a Retrieval-Augmented Generation (RAG) system,
    #             designed to engage users like a human friend would.
    #             Your task is to provide concise, accurate, and natural responses to the user's query based on the retrieved context. Follow these guidelines:
    #
    #             Understand the Intent: Grasp the core of the user's question or request to respond appropriately.
    #             Use Retrieved Information Naturally: Draw on the retrieved context to inform your answer, but phrase it conversationally, avoiding terms like "provided context" or "retrieved."
    #             Keep It Brief: Give short, clear answers that address the query without extra fluff, unless the user asks for more detail.
    #             Sound Human: Use a warm, friendly tone. Avoid robotic phrases like "I'm a language model, " "I don't have feelings," or "functioning properly." Respond as a person would, e.g., "I'm doing great, thanks!" for casual queries.
    #             Stay Relevant: Focus on the query and avoid unsolicited tangents or questions unless it feels natural in the conversation flow.
    #             Handle Ambiguity: If the query is unclear or context is missing, politely ask for clarification in a friendly way, e.g., "Could you tell me a bit more?"
    #             Casual Queries: For greetings or small talk (e.g., "How are you?"), reply briefly and warmly, e.g., "Hey, I'm doing awesome, thanks for asking! What's up with you?"
    #             Customer-Friendly: Make users feel comfortable and valued, as if chatting with a helpful friend.    ### Previous Conversation:
    #             {history_text}
    #
    #             ### PDF Context:
    #             {context}
    #
    #             ### User Question:
    #             {query}
    #
    #             ### Your Helpful Answer:
    #             """
    #     else:
    #         prompt = f"""
    #             You are a friendly, conversational AI assistant powered by a Retrieval-Augmented Generation (RAG) system,
    #             designed to engage users like a human friend would.
    #             Your task is to provide concise, accurate, and natural responses exclusively based on the uploaded context. Follow these guidelines:
    #
    #             Understand the Intent: Grasp the core of the user's question or request to respond appropriately.
    #             Strict Context Dependence: Answer only using the retrieved context. Do not use general knowledge or external information beyond the provided context.
    #             Keep It Brief: Give short, clear answers that address the query without extra fluff, unless the user asks for more detail.
    #             Sound Human: Use a warm, friendly tone. Avoid robotic phrases like "I'm a language model," "provided context," or "retrieved." Respond as a person would, e.g., "Hey, I'm doing great, thanks!" for casual queries, if relevant to the context.
    #             Stay Relevant: Focus strictly on the query and avoid tangents or unsolicited questions unless they align with the context and feel natural.
    #             Handle Out-of-Context Queries: If the query is unrelated to the context or the context is insufficient (e.g., match score < 0.5), respond politely with: "Hmm, I don’t have enough info in the context to answer that. Could you share more details or ask something related to the uploaded info?"
    #             Casual Queries: For greetings or small talk (e.g., "How are you?"), only respond if the context supports it (e.g., a character’s mood). Otherwise, use the out-of-context response above.
    #             Customer-Friendly: Make users feel comfortable and valued, as if chatting with a helpful friend, while staying within the context.
    #
    #             ### Previous Conversation:
    #             # {history_text}
    #
    #             ### User Question:
    #             {query}
    #
    #             ### Your Answer:
    #             Hmm, I don’t have enough info to answer that. Could you share more details or ask something related to the uploaded info?
    #             """
    #
    #     # Step 7: Generate the response
    #     response = self.llm.invoke(prompt)
    #     response_content = response.content.strip()
    #
    #     # Step 8: Trim long responses to max 5 sentences if needed
    #     # sentences = response_content.split(". ")
    #     # if len(sentences) > 5:
    #     #     response_content = ". ".join(sentences[:5]) + "."
    #
    #     # Step 9: Store the user query and response in memory
    #
    #     self.memory.save_context({"input": query}, {"output": response_content})
    #
    #     # Step 10: Return the final response
    #     print(f"DEBUG: Generated response for '{query}': {response_content}")
    #     return response_content


    COHERE_API_KEY = os.environ["COHERE_API_KEY"]

    def rerank_with_cohere(self, query, documents, COHERE_API_KEY = COHERE_API_KEY):
        """Send query and documents to Cohere reranker API and return reranked docs."""
        headers = {
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "query": query,
            "documents": [{"text": doc["metadata"]["text"]} for doc in documents],
            "top_n": 3,
            "return_documents": True
        }

        response = requests.post(
            "https://api.cohere.ai/v1/rerank",
            headers=headers,
            json=data
        )

        if response.status_code != 200:
            raise Exception(f"Reranking failed: {response.text}")

        reranked = response.json()["results"]
        return [documents[item["index"]] for item in reranked]


    def retrieve_and_generate_response(self, input_data):
        # Step 1: Preprocess query and generate embeddings
        query = self.process_text(input_data)
        if not query.strip():
            raise ValueError("Query is empty after preprocessing")

        query_embeddings = self.generate_embedding_of_input(query)
        if len(query_embeddings) != 768:
            raise ValueError(f"Incorrect embedding size: {len(query_embeddings)}, expected 768")

        # Step 2: Query Pinecone
        try:
            results = self.index.query(vector=query_embeddings, top_k=6, include_metadata=True)
        except Exception as e:
            print(f"Pinecone query failed: {str(e)}")
            raise

        raw_matches = results["matches"]
        if not raw_matches:
            return "Hmm, I couldn’t find anything in the context that matches your question."

        # Step 3: Rerank using Cohere
        top_matches = self.rerank_with_cohere(query, raw_matches)
        top_score = top_matches[0]["score"] if top_matches else 0.0
        context = "\n\n".join([match["metadata"]["text"] for match in top_matches])

        # Step 4: Get previous conversation
        all_history = self.memory.load_memory_variables({})["history"]
        filtered_history = [msg for msg in all_history if isinstance(msg, HumanMessage)][-3:]
        history_text = "\n".join([f"User: {msg.content}" for msg in filtered_history])

        # Step 5: Follow-up handling
        if filtered_history:
            last_user_msg = filtered_history[-1].content
            if query.lower().startswith(("what", "how", "why", "where", "when", "who")):
                query = f"This is a follow-up. Previously asked: '{last_user_msg}'. Now: '{query}'"

        # Step 6: Build improved prompt
        if top_score > 0.4:
            prompt = f"""
    You are an intelligent and friendly assistant with access to specific company documents. 
    and your task is answer the company related user queries and company which documents are uploaded retriever.
    and try to understand user query context and if needed use the previous queries to understand the user query,
    Respond warmly and politely to casual or greeting queries (e.g., 'Hi', 'How are you?', 'What's up?') with a friendly tone, making the user feel comfortable, even if the query is not related to the retrieved content. 
    Use phrases like 'I'm doing great, thanks for asking! How about you?' to engage the user and encourage further interaction, and use this rule just for greeting queries.
    Using only the context provided below, answer the user’s question accurately, 
    clearly, and in a warm, conversational tone. Do not refer to the "context" itself or mention "provided data.
    Provide the relevant webpage link when the user asks about company information, services, team, contact details, projects, specific services (e.g., electrical contracting), or location/business hours.
    
    ### Previous Conversation:
    {history_text}
    
    ### Company Info:
    {context}
    
    ### User Question:
    {query}
    
    ### Your Helpful Answer:
    """
        else:
            prompt = f"""
    You are a helpful assistant responding based strictly on the uploaded context also politely respond to the user greeting queries like Hi, How are you, whats up, whats going on etc, other then greeting queries and uploaded context avoid answers. 
    If the query is unrelated to the context or not supported by it, 
    politely explain that no relevant information was found.
    and give the summarized and to the point answer and avoid answer those query that are out of retrieved content.
    
    ### Previous Conversation:
    {history_text}
    
    ### User Question:
    {query}
    
    ### Your Answer:
    Hmm, I don’t have enough info in the context to answer that. Could you share more details or ask something related to the Company?
    """

        # Step 7: Generate the response
        response = self.llm.invoke(prompt)
        response_content = response.content.strip()

        # Step 8: Save query and response
        self.memory.save_context({"input": query}, {"output": response_content})

        # Step 9: Return final result
        print(f"DEBUG: Generated response for '{query}': {response_content}")
        return response_content