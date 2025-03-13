# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generative AI Solution Development
# MAGIC
# MAGIC This course is designed to introduce participants to contextual generative AI solutions using the retrieval-augmented generation (RAG) method. Firstly, participants will be introduced to the RAG architecture and the significance of contextual information using Mosaic AI Playground. Next, the course will demonstrate how to prepare data for generative AI solutions and connect this process with building an RAG architecture. Finally, participants will explore concepts related to context embedding, vectors, vector databases, and the utilization of the Mosaic AI Vector Search product.
# MAGIC
# MAGIC
# MAGIC ## Course Agenda
# MAGIC | Module &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Lessons &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
# MAGIC |:----:|-------|
# MAGIC | **[From Prompt Engineering to RAG]($./01 - From Prompt Engineering to RAG)**    | **Lecture -** Prompt Engineering Primer <br> **Lecture -** Introduction to RAG <br> [Demo: In Context Learning with AI Playground]($./01 - From Prompt Engineering to RAG/1.1 - In Context Learning with AI Playground) <br>[Lab: In Context Learning with AI Playground]($./01 - From Prompt Engineering to RAG/1.LAB - In Context Learning with AI Playground)|
# MAGIC | **[Preparing Data for RAG Solutions]($./02 - Preparing Data for RAG Solutions)** | **Lecture -** Preparing Data for RAG Solutions </br> [Demo: Preparing Data for RAG]($./02 - Preparing Data for RAG Solutions/2.1 - Preparing Data for RAG) </br> [Lab: Preparing Data for RAG]($./02 - Preparing Data for RAG Solutions/2.LAB - Preparing Data for RAG) | 
# MAGIC | **[Mosaic AI Vector Search]($./03 - Mosaic AI Vector Search)** | **Lecture -** Introduction to Vector Stores </br> **Lecture -** Introduction to Mosaic AI Vector Search </br> [Demo: Create Self-managed Vector Search Index]($./03 - Mosaic AI Vector Search/3.1 - Create Self-managed Vector Search Index) </br> [Lab: Create Vector Search Index]($./03 - Mosaic AI Vector Search/3.LAB - Create Managed Vector Search Index) | 
# MAGIC | **[Assembling and Evaluating a RAG Application]($./04 - Assembling and Evaluating a RAG Application)** | **Lecture -** Assembling a RAG Application </br> [Demo: Assembling and Evaluating a RAG Application]($./04 - Assembling and Evaluating a RAG Application/4.1 - Assembling and Evaluating a RAG Application) </br> [Lab: Assembling a RAG Application]($./04 - Assembling and Evaluating a RAG Application/4.LAB - Assembling a RAG Application)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run demo and lab notebooks, you need to use one of the following Databricks runtime(s): **15.4.x-cpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
