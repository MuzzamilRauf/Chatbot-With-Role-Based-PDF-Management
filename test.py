# import torch
# from sentence_transformers import SentenceTransformer
#
# print(torch.__version__)
# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# from dotenv import load_dotenv
# load_dotenv()
#
# llm = HuggingFaceEndpoint(
#     repo_id="microsoft/bitnet-b1.58-2B-4T",
#     task="text-generation"
#     )
#
# model = ChatHuggingFace(llm = llm)
#
# result = model.invoke("what is the capital of Pakistan")
#
# print(result.content)

from pinecone import Pinecone, Index

# pc = Pinecone(api_key="pcsk_4PGG2i_4NSfCD5Q6PsCs8bsjcRpjgsUUSVYsa4m2AGbTuvnPba8g182Fm9jGVHXgAyHtKn")
# index = pc.Index("dense-index")
#
# # Delete all vectors in the entire index
# index.delete(delete_all=True)
#
# print("All vectors deleted from the index.")