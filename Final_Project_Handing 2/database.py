import pandas as pd
import sqlite3

class Database():
    def __init__(self,database):
        self.conn = sqlite3.connect(database)
    
    def drop(self,tableName):
        with self.conn:
            self.conn.execute(f"DROP TABLE IF EXISTS {tableName}")
        
    def create(self,tableName, varNames):
        with self.conn:
            if not isinstance(varNames,tuple):
                raise Exception("Variable names should be a tuple")
            varNames = ",".join(varNames)
            self.conn.execute(f"CREATE TABLE IF NOT EXISTS {tableName} ({varNames})")

    
    
    def insert_Into(self,tablename,data,if_exists = "append"):
        with self.conn:
            if isinstance(data,list):
                valList = ",".join("?"*len(data[0]))
                self.conn.executemany(f"INSERT INTO {tablename} values ({valList})", data)
            elif isinstance(data,pd.DataFrame):
                data.to_sql(tablename,self.conn,if_exists = if_exists,index = False)
            elif isinstance(data,tuple):
                self.conn.execute(f"INSERT INTO {tablename} {data}")
            else: raise Exception("Data needs to be in form of tuple, list of tuples or dataframe")


    def query(self,query):
        return pd.read_sql(query,self.conn)
    
    def get_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        print(cursor.fetchall())
        
    
    def table_info(self,tablename):
        return self.query(f"PRAGMA table_info({tablename})")


# In[ ]:




