import pandas as pd
import numpy as np
import dataio

dbutils = dataio.DataUtils()
_posting_df = ''


def get_job_posting_data():
    '''
    Retrieve all relevant user data from the database and preprocess
    '''
    global _posting_df
    if type(_posting_df) == pd.DataFrame:
        return _posting_df
    query = """
    WITH biz_tags as
        (select bjt.business_id as id, string_agg(jt.name, ',') as tag_names
        from job_tags jt, businesses_job_tags bjt
        where jt.id=bjt.job_tag_id
        group by bjt.business_id)
    SELECT jp.id, jp.business_id, jp.title, jp.description, jt.tag_names
    FROM job_postings jp
    LEFT JOIN biz_tags jt ON jt.id=jp.business_id;
    """

    print 'Getting job posting data...'
    import time
    t1 = time.time()
    df = dbutils.run_query(query)
    t2 = time.time()
    print "- Time: " + str((t2 - t1)) + "\n"

    return df
