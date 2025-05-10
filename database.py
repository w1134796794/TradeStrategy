import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_batch
import logging
from typing import List, Dict


class DatabaseManager:
    def __init__(self, dbname, user, password, host, port):
        self.conn = None
        self.params = {
            'dbname': dbname,
            'user': user,
            'password': password,
            'host': host,
            'port': port
        }
        self.connect()

    def connect(self):
        try:
            self.conn = psycopg2.connect(**self.params)
            logging.info("Successfully connected to database")
        except Exception as e:
            logging.error(f"Connection failed: {str(e)}")
            raise

    def execute_batch(self, query: str, data: List[tuple]):
        try:
            with self.conn.cursor() as cur:
                execute_batch(cur, query, data)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Batch execute failed: {str(e)}")
            raise

    def upsert_data(self, table: str, columns: List[str], data: List[tuple], conflict_keys: List[str]):
        """通用upsert方法"""
        insert_sql = sql.SQL("""
            INSERT INTO {table} ({fields})
            VALUES ({values})
            ON CONFLICT ({conflict}) DO UPDATE SET
            {updates}
        """).format(
            table=sql.Identifier(table),
            fields=sql.SQL(', ').join(map(sql.Identifier, columns)),
            values=sql.SQL(', ').join(sql.Placeholder() * len(columns)),
            conflict=sql.SQL(', ').join(map(sql.Identifier, conflict_keys)),
            updates=sql.SQL(', ').join([
                sql.SQL("{}=EXCLUDED.{}").format(
                    sql.Identifier(col),
                    sql.Identifier(col)
                )
                for col in columns if col not in conflict_keys
            ])
        )
        try:
            with self.conn.cursor() as cur:
                execute_batch(cur, insert_sql, data)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Upsert failed: {str(e)}")
            raise

    def close(self):
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed")