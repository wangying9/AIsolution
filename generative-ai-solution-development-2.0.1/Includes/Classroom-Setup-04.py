# Databricks notebook source
# MAGIC %run ./Classroom-Setup-Common

# COMMAND ----------

import logging

logging.getLogger("py4j").setLevel(logging.ERROR)

DA = DBAcademyHelper()
DA.init()
                           
print("\nThe examples and models presented in this course are intended solely for demonstration and educational purposes.\n Please note that the models and prompt examples may sometimes contain offensive, inaccurate, biased, or harmful content.")
