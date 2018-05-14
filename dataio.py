import pandas as pd
import psycopg2 as pg2


class DataUtils(object):
    '''
    Class provides Localwise postgres database services.
    '''
    def __init__(self):
        config_file = '../config/access.config'
        config = {}
        with open(config_file) as f:
            for line in f:
                pair = line.rstrip().split('=')
                if len(pair) != 2:
                    print 'Bad condig formating.  ' \
                    'Should be key=value, one pair per line.'
                    return None
                config[pair[0]] = pair[1]

        try:
            self.conn = pg2.connect(dbname=config['name'],
                                    user=config['user'],
                                    host=config['host'],
                                    password=config['pass'])
            self.cur = self.conn.cursor()
        except Exception, e:
            print 'Could not connect to database.'
            print e

    def get_conn(self):
        return self.conn

    def run_query(self, query):
        df = pd.read_sql_query(query, self.conn)
        return df

    def get_table(self, table_name):
        '''
        Get all rows from a table.

        Input:
            table_name
        Output:
            Dataframe
        '''
        query = "select * from " + table_name
        table = pd.read_sql_query(query, self.conn)

        return table

    def get_core_data(self):
        '''
        Get all rows from the core Localwise table used for machine learning.

        Input:
            table_name
        Output:
            Dataframes
        '''
        # cur.execute("select * from users")
        # u = pd.DataFrame(cur.fetchall())
        u = pd.read_sql_query("select * from users", self.conn)
        b = pd.read_sql_query("select * from businesses", self.conn)
        b['lonlat'] = pd.read_sql_query("select ST_AsText(lonlat) as lonlat from businesses", self.conn)
        jp = pd.read_sql_query("select * from job_postings", self.conn)
        ja = pd.read_sql_query("select * from job_applications", self.conn)

        # businesses_job_applications, businesses_job_postings, businesses_job_tags

        return u, b, jp, ja
