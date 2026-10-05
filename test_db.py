import os
from dotenv import load_dotenv
import psycopg2
load_dotenv()

conn = psycopg2.connect(
    host="localhost",
    database="avalon_research",
    user="postgres",
    password=os.environ["DB_PASSWORD"]
)
print("Connected successfully")
conn.close()