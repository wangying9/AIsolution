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
# MAGIC # Preparing Data for RAG
# MAGIC
# MAGIC
# MAGIC **In this demo, we will focus on ingesting PDF documents as a source for our retrieval process.** First, we will ingest the PDF files and process them using self-managed vector search embeddings. 
# MAGIC
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to:*
# MAGIC
# MAGIC * Split the data into chunks that are at least as small as the maximum context window of the LLM to be used later.
# MAGIC
# MAGIC * Describe how to appropriately choose an embedding model.
# MAGIC
# MAGIC * Compute embeddings for each of the chunks using a Databricks-managed embedding model.
# MAGIC
# MAGIC * Use the chunking strategy to divide up the context information to be provided to a model.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **15.4.x-cpu-ml-scala2.12**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Install required libraries.

# COMMAND ----------

# MAGIC %pip install -q -U llama-index PyPDF2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define the configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-02

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
# MAGIC ## Demo Overview
# MAGIC
# MAGIC For this example, we will add research articles on GenerativeAI as PDFs from [Arxiv resources page](https://arxiv.org/abs/2309.07930) to our knowledge database.
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/genai/genai-as-01-rag-pdf-self-managed-0.png" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC
# MAGIC Here are all the detailed steps:
# MAGIC
# MAGIC - Use Spark to load the binary PDFs into our first table. 
# MAGIC - Use the `unstructured` library  to parse the text content of the PDFs.
# MAGIC - Use `llama_index` or `langchain` to split the texts into chunks.
# MAGIC - Compute embeddings for the chunks.
# MAGIC - Save our text chunks + embeddings in a Delta Lake table, ready for Vector Search indexing.
# MAGIC
# MAGIC
# MAGIC Databricks Lakehouse AI not only provides state of the art solutions to accelerate your AI and LLM projects but also to accelerate data ingestion and preparation at scale, including unstructured data like PDFs.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract PDF Content as Text Chunks
# MAGIC
# MAGIC As the first step, we need to ingest PDF files and divide the content into chunks. PDF files are already downloaded during the course step and stored in **datasets path**.

# COMMAND ----------

# Reduce the arrow batch size as our PDF can be big in memory
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)

# COMMAND ----------

DA.paths.datasets.arxiv

# COMMAND ----------

articles_path = f"{DA.paths.datasets.arxiv}/arxiv-articles/"
table_name = f"{DA.catalog_name}.{DA.schema_name}.pdf_raw_text"

# read pdf files
df = (
        spark.read.format("binaryfile")
        .option("recursiveFileLookup", "true")
        .load(articles_path)
        )

# save list of the files to table
df.write.mode("overwrite").saveAsTable(table_name)

display(df)

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's view the content of one of the articles.

# COMMAND ----------

articles_path

# COMMAND ----------

# The parse_bytes_pypdf function was created during our classroom setup using the PyPDF2 library. This function extracts text from a PDF file given its binary content.

with open(f"{articles_path}2302.06476.pdf", mode="rb") as pdf:
  doc = parse_bytes_pypdf(pdf.read()) 
  print(doc)

# COMMAND ----------

# MAGIC %md
# MAGIC This looks great. We'll now wrap it with a `text_splitter` to avoid having too big pages and create a **Pandas UDF function** to easily scale that across multiple nodes.
# MAGIC
# MAGIC **📌Note:** The pdf text isn't clean. To make it nicer, we could use a few extra LLM-based pre-processing steps, asking to remove irrelevant content like the list of chapters and to only keep the core text.

# COMMAND ----------

import io
import os
import pandas as pd 

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.core.utils import set_global_tokenizer
from transformers import AutoTokenizer
from typing import Iterator
from pyspark.sql.functions import col, udf, length, pandas_udf, explode
from PyPDF2 import PdfReader  # Updated import

# COMMAND ----------

# Define a function to split the text content into chunks
@pandas_udf("array<string>")
def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    # Set llama2 as tokenizer
    set_global_tokenizer(
      AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer") #loads the tokenizer for the Llama2 model
    )
    # Sentence splitter from llama_index to split on sentences: 
    # split the input text into chunks of 500 characters, with an overlap of 50 characters between consecutive chunks.
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    
    def extract_and_split(b):
        txt = parse_bytes_pypdf(b)  # Use custom function
        if txt is None:
            return []
        nodes = splitter.get_nodes_from_documents([Document(text=txt)]) #the sentence splitter to break the text into smaller chunks
        return [n.text for n in nodes]

    for x in batch_iter:
        yield x.apply(extract_and_split)

# COMMAND ----------

df_chunks = (df
                .withColumn("content", explode(read_as_chunk("content")))
                .selectExpr('path as pdf_name', 'content')
                )
display(df_chunks)

# COMMAND ----------

# MAGIC %md
# MAGIC **💡 Chunking Overlap**: Review the content chunks and pay attention to the sentence overlap between chunks. This was defined with `SentenceSplitter` above.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Embeddings with Foundation Model Endpoints
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/genai/genai-as-01-rag-pdf-self-managed-4.png" width="100%">
# MAGIC
# MAGIC Foundation Models are provided by Databricks, and can be used out-of-the-box.
# MAGIC
# MAGIC Databricks supports several endpoint types to compute embeddings or evaluate a model:
# MAGIC - A **foundation model endpoint**, provided by databricks (ex: llama3.1-70B, Mixtral-8-7B...)
# MAGIC - An **external endpoint**, acting as a gateway to an external model (ex: Azure OpenAI)
# MAGIC - A **custom**, fine-tuned model hosted on Databricks model service
# MAGIC
# MAGIC Open the [Model Serving Endpoint page](/ml/endpoints) to explore and try the foundation models.
# MAGIC
# MAGIC For this demo, we will use the foundation model `GTE` (embeddings) and `llama3.1-70B` (chat). <br/><br/>
# MAGIC
# MAGIC ![serving](files/images/generative-ai-solution-development-2.0.1/serving.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### How to Use Foundation Model API
# MAGIC
# MAGIC Before the compute the embeddings for the chunked text we created before, let's quickly go over how to use Foundation Model API.
# MAGIC
# MAGIC **🚨 Important:** You will need Foundation Model API access for this section and the rest of the demo.

# COMMAND ----------

from mlflow.deployments import get_deploy_client


# gte-large-en Foundation models are available using the /serving-endpoints/databricks-gte-large-en/invocations api. 
deploy_client = get_deploy_client("databricks")

# NOTE: if you change your embedding model here, make sure you change it in the query step too
embeddings = deploy_client.predict(endpoint="databricks-gte-large-en", inputs={"input": ["What is Apache Spark?"]})
pprint(embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compute Chunking Embeddings
# MAGIC
# MAGIC The last step is to now compute an embedding for all our documentation chunks. Let's create an udf to compute the embeddings using the foundation model endpoint.
# MAGIC
# MAGIC **📌 Note:** This part would typically be setup as a production-grade job, running as soon as a new documentation page is updated.
# MAGIC This could be setup as a Delta Live Table pipeline to incrementally consume updates.
# MAGIC
# MAGIC

# COMMAND ----------

@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    import mlflow.deployments
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    def get_embeddings(batch):
        # NOTE: this will fail if an exception is thrown during embedding creation (add try/except if needed) 
        response = deploy_client.predict(endpoint="databricks-gte-large-en", inputs={"input": batch})
        return [e["embedding"] for e in response.data]

    # splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

    # process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.tolist())

    return pd.Series(all_embeddings)

# COMMAND ----------

import pyspark.sql.functions as F

df_chunk_emd = (df_chunks
                .withColumn("embedding", get_embedding("content"))
                .selectExpr("pdf_name", "content", "embedding")
                )
display(df_chunk_emd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Embeddings to a Delta Table
# MAGIC
# MAGIC Now that the embeddings are ready, let's create a Delta table and store the embeddings in this table.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS pdf_text_embeddings (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   pdf_name STRING,
# MAGIC   content STRING,
# MAGIC   embedding ARRAY <FLOAT>
# MAGIC   -- NOTE: the table has to be CDC because VectorSearch is using DLT that is requiring CDC state
# MAGIC   ) TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

embedding_table_name = f"{DA.catalog_name}.{DA.schema_name}.pdf_text_embeddings"
df_chunk_emd.write.mode("append").saveAsTable(embedding_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Clean up Classroom
# MAGIC
# MAGIC **🚨 Warning:** Please refrain from deleting the tables created in this demo, as they are required for upcoming demos.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, we demonstrated how to ingest and process documents for a RAG application. The first was to extract text chunks from PDF documents. Then, we created embeddings using a foundation model. This process includes setting up an endpoint and computing embeddings for the chunks. In the final step, we stored computed embeddings in a Delta table.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
