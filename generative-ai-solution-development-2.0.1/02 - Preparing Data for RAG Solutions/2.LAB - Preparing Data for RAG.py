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
# MAGIC # Lab: Preparing Data for RAG
# MAGIC
# MAGIC The objective of this lab is to demonstrate the process of ingesting and processing documents for a Retrieval-Augmented Generation (RAG) application. This involves extracting text from PDF documents, computing embeddings using a foundation model, and storing the embeddings in a Delta table.
# MAGIC
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC In this lab, you will need to complete the following tasks:
# MAGIC
# MAGIC * **Task 1 :** Read the PDF files and load them into a DataFrame.
# MAGIC
# MAGIC * **Task 2 :** Extract the text content from the PDFs and split it into manageable chunks.
# MAGIC
# MAGIC * **Task 3 :** Compute embeddings for each text chunk using a foundation model endpoint.
# MAGIC
# MAGIC * **Task 4 :** Create a Delta table to store the computed embeddings.
# MAGIC
# MAGIC **📝 Your task:** Complete the **`<FILL_IN>`** sections in the code blocks and follow the other steps as instructed.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **15.4.x-cpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the lab, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %pip install -qq -U llama-index pydantic PyPDF2
# MAGIC dbutils.library.restartPython()

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
# MAGIC ## Task 1: Read the PDF files and load them into a DataFrame.
# MAGIC
# MAGIC To start, you need to load the PDF files into a DataFrame.
# MAGIC
# MAGIC **Steps:**
# MAGIC
# MAGIC 1. Use Spark to load the binary PDFs into a DataFrame.
# MAGIC
# MAGIC 2. Ensure that each PDF file is represented as a separate record in the DataFrame.

# COMMAND ----------

# run this cell to import the required libraries
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.core.utils import set_global_tokenizer
from transformers import AutoTokenizer
from typing import Iterator
from pyspark.sql.functions import col, udf, length, pandas_udf, explode
import os
import pandas as pd 
import io
from PyPDF2 import PdfReader

# COMMAND ----------

# use Spark to load the PDF files into a DataFrame
# reduce the arrow batch size as our PDF can be big in memory
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)
articles_path = f"{DA.paths.datasets.arxiv}/arxiv-articles/"
table_name = f"{DA.catalog_name}.{DA.schema_name}.lab_pdf_raw_text"

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

# MAGIC %md
# MAGIC ## Task 2: Extract the text content from the PDFs and split it into manageable chunks
# MAGIC
# MAGIC Next, extract the text content from the PDFs and split it into manageable chunks.
# MAGIC
# MAGIC **Steps:**
# MAGIC
# MAGIC 1. Define a function to split the text content into chunks.
# MAGIC
# MAGIC     * Split the text content into manageable chunks.
# MAGIC
# MAGIC     * Ensure each chunk contains a reasonable amount of text for processing.
# MAGIC
# MAGIC 2. Apply the function to the DataFrame to create a new DataFrame with the text chunks.
# MAGIC

# COMMAND ----------

# define a function to split the text content into chunks
@pandas_udf("array<string>")
def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    # set llama2 as tokenizer
    <FILL_IN>
    # sentence splitter from llama_index to split on sentences
    <FILL_IN>
      return [n.text for n in nodes]

    for x in batch_iter:
        <FILL_IN>

df_chunks = (df
                .withColumn("content", explode(read_as_chunk("content")))
                .selectExpr('path as pdf_name', 'content')
                )
display(df_chunks)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Compute embeddings for each text chunk using a foundation model endpoint
# MAGIC Now, compute embeddings for each text chunk using a foundation model endpoint.
# MAGIC
# MAGIC **Steps:**
# MAGIC
# MAGIC 1. Define a function to compute embeddings for text chunks.
# MAGIC     + Use a foundation model endpoint to compute embeddings for each text chunk.
# MAGIC     + Ensure that the embeddings are computed efficiently, considering the limitations of the model.  
# MAGIC
# MAGIC 2. Apply the function to the DataFrame containing the text chunks to compute embeddings for each chunk.
# MAGIC

# COMMAND ----------

# define a function to compute embeddings for text chunks
@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    # define deployment client
    <FILL_IN>
 
    def get_embeddings(batch):
        # calculate embeddings using the deployment client's predict function 
        response = <FILL_IN>
        return [e['embedding'] for e in response.data]

    # splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

    # process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        <FILL_IN>

    return <FILL_IN>
    
df_chunk_emd = (df_chunks
                .withColumn("embedding", get_embedding("content"))
                .selectExpr("pdf_name", "content", "embedding")
               )
display(df_chunk_emd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Create a Delta table to store the computed embeddings
# MAGIC
# MAGIC Finally, create a Delta table to store the computed embeddings.
# MAGIC
# MAGIC Steps:
# MAGIC
# MAGIC   1. Define the schema for the Delta table.
# MAGIC
# MAGIC   2. Save the DataFrame containing the computed embeddings as a Delta table.
# MAGIC
# MAGIC
# MAGIC **Note:** Ensure that the Delta table is properly structured to facilitate efficient querying and retrieval of the embeddings.
# MAGIC
# MAGIC **📌 Instructions:** 
# MAGIC
# MAGIC - Please execute the following SQL code block to create the Delta table. This table will store the computed embeddings along with other relevant information. 
# MAGIC
# MAGIC **Important:** Storing the computed embeddings in a structured format like a Delta table ensures efficient querying and retrieval of the embeddings when needed for various downstream tasks such as retrieval-augmented generation. Additionally, setting the `delta.enableChangeDataFeed` property to true enables Change Data Feed (CDC), which is required for VectorSearch to efficiently process changes in the Delta table.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS lab_pdf_text_embeddings (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   pdf_name STRING,
# MAGIC   content STRING,
# MAGIC   embedding ARRAY <FLOAT>
# MAGIC   -- NOTE: the table has to be CDC because VectorSearch is using DLT that is requiring CDC state
# MAGIC   ) TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

# define the schema for the Delta table
embedding_table_name = f"{DA.catalog_name}.{DA.schema_name}.lab_pdf_text_embeddings"
# save the DataFrame as a Delta table
df_chunk_emd.<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Clean up Classroom
# MAGIC
# MAGIC **🚨 Warning:** Please refrain from deleting tables created in this lab, as they are required for upcoming labs. To clean up the classroom assets, execute the classroom clean-up script provided in the final lab.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you learned how to prepare data for Retrieval-Augmented Generation (RAG) applications. By extracting text from PDF documents, computing embeddings, and storing them in a Delta table, you can enhance the capabilities of language models to generate more accurate and relevant responses.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
