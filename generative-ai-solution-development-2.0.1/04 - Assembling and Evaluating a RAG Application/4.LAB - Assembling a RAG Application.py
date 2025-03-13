# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # LAB - Assembling a RAG Application
# MAGIC
# MAGIC In this lab, we will assemble a Retrieval-augmented Generation (RAG) application using the components we previously created. The primary goal is to create a seamless pipeline where users can ask questions, and our system retrieves relevant documents from a Vector Search index to generate informative responses.
# MAGIC
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC In this lab, you will need to complete the following tasks;
# MAGIC
# MAGIC * **Task 1 :** Setup the Retriever Component
# MAGIC
# MAGIC * **Task 2 :** Setup the Foundation Model
# MAGIC
# MAGIC * **Task 3 :** Assemble the Complete RAG Solution
# MAGIC
# MAGIC * **Task 4 :** Save the Model to Model Registry in Unity Catalog
# MAGIC
# MAGIC **📝 Your task:** Complete the **`<FILL_IN>`** sections in the code blocks and follow the other steps as instructed.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **15.4.x-cpu-ml-scala2.12**
# MAGIC
# MAGIC **🚨 Important:** This lab relies on the resources established in the previous one. Please ensure you have completed the prior lab before starting this one.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %pip install -U -qq databricks-vectorsearch langchain==0.3.7 flashrank langchain-databricks PyPDF2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-04

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this demo, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Setup the Retriever Component
# MAGIC **Steps:**
# MAGIC 1. Define the embedding model.
# MAGIC 1. Get the vector search index that was created in the previous lab.
# MAGIC 1. Generate a **retriever** from the vector store. The retriever should return **three results.**
# MAGIC 1. Write a test prompt and show the returned search results.
# MAGIC

# COMMAND ----------

# Components we created before
vs_endpoint_prefix = "vs_endpoint_"
vs_endpoint_name = vs_endpoint_prefix+str(get_fixed_integer(DA.unique_name("_")))
print(f"Assigned Vector Search endpoint name: {vs_endpoint_name}.")

vs_index_fullname = f"{DA.catalog_name}.{DA.schema_name}.lab_pdf_text_managed_vs_index"

# COMMAND ----------


from databricks.vector_search.client import VectorSearchClient
from langchain_databricks import DatabricksEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain.docstore.document import Document
from flashrank import Ranker, RerankRequest


def get_retriever(cache_dir=f"{DA.paths.working_dir}/opt"):

    def retrieve(query, k: int=10):
        if isinstance(query, dict):
            query = next(iter(query.values()))

        # get the vector search index
        vsc = VectorSearchClient(disable_notice=True)
        vs_index = <FILL_IN>
        
        # get similar k documents
        return <FILL_IN>


    def rerank(query, retrieved, cache_dir, k: int=2):
        # format result to align with reranker lib format 
        passages = []
        for doc in retrieved.get("result", {}).get("data_array", []):
            new_doc = {"file": doc[0], "text": doc[1]}
            passages.append(new_doc)       
        #Load the flashrank ranker
        ranker = <FILL_IN>

        # rerank the retrieved documents
        rerankrequest = RerankRequest(query=query, passages=passages)
        results = ranker.<FILL_IN>

        # format the results of rerank to be ready for prompt
        return [Document(page_content=r.get("text"), metadata={"source": r.get("file")}) for r in results]

    # the retriever is a runnable sequence of retrieving and reranking.
    return <FILL_IN>


# test your retriever
question = <FILL_IN>
vectorstore = get_retriever()
similar_documents = vectorstore.<FILL_IN>
print(f"Relevant documents: {similar_documents}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Setup the Foundation Model
# MAGIC **Steps:**
# MAGIC 1. Define the foundation model for generating responses. Use `llama-3.1` as foundation model. 
# MAGIC 2. Test the foundation model to ensure it provides accurate responses.

# COMMAND ----------

# import necessary libraries
from langchain_databricks import ChatDatabricks

# define foundation model for generating responses
chat_model = <FILL_IN>

# test foundation model
print(f"Test chat model: {<FILL_IN>('What is Generative AI?')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 3: Assemble the Complete RAG Solution
# MAGIC **Steps:**
# MAGIC 1. Merge the retriever and foundation model into a single Langchain chain.
# MAGIC 2. Configure the Langchain chain with proper templates and context for generating responses.
# MAGIC 3. Test the complete RAG solution with sample queries.

# COMMAND ----------

# import necessary libraries
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate

# define template for prompt
TEMPLATE = """You are an assistant for GENAI teaching class. You are answering questions related to Generative AI and how it impacts humans life. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
Use the following pieces of context to answer the question at the end:
{context}
Question: {input}
Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "input"])

# unwrap the longchain document from the context to be a dict so we can register the signature in mlflow
def unwrap_document(answer):
  return answer | {"context": [{"metadata": r.metadata, "page_content": r.page_content} for r in answer['context']]}

# merge retriever and foundation model into Langchain chain
question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
chain = <FILL_IN>


# test the complete RAG solution with sample query
question = {"input": "How Generative AI impacts humans?"}
answer = <FILL_IN>
print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 4: Save the Model to Model Registry in Unity Catalog
# MAGIC **Steps:**
# MAGIC 1. Register the assembled RAG model in the Model Registry with Unity Catalog.
# MAGIC 2. Ensure that all necessary dependencies and requirements are included.
# MAGIC 3. Provide an input example and infer the signature for the model.

# COMMAND ----------

# import necessary libraries
from mlflow.models import infer_signature
import mlflow
import langchain

# set Model Registry URI to Unity Catalog
mlflow.<FILL_IN>
model_name = f"{DA.catalog_name}.{DA.schema_name}.rag_app_demo4"

# register the assembled RAG model in Model Registry with Unity Catalog
with mlflow.start_run(run_name="rag_app_demo4") as run:
    signature = <FILL_IN>
    model_info = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Clean up Resources
# MAGIC
# MAGIC This was the final lab. You can delete all resources created in this course.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you learned how to assemble a Retrieval-augmented Generation (RAG) application using Databricks components. By integrating Vector Search for document retrieval and a foundational model for response generation, you created a powerful tool for answering user queries. This lab provided hands-on experience in building end-to-end AI applications and demonstrated the capabilities of Databricks for natural language processing tasks.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
