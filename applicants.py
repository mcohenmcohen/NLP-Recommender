import pandas as pd
from lxml import etree, objectify
from StringIO import StringIO
import dataio

dbutils = dataio.DataUtils()
_applicant_df = ''


def get_applicant_data():
    '''
    Retrieve all relevant user data from the database and preprocess
    '''
    global _applicant_df
    if type(_applicant_df) == pd.DataFrame:
        return _applicant_df
    query = """
        SELECT u.id, u.first_name, u.last_name,
        ap.parsed_resume_xml, ap.passions, ap.introduction,
        ujt.job_tag_id, ujt.tag_type
        FROM users u
        LEFT JOIN applicant_profiles ap ON ap.user_id=u.id
        LEFT JOIN user_job_tags ujt ON ujt.user_id=u.id
    """

    print 'Getting user and applicant data...'
    import time
    t1 = time.time()
    df = dbutils.run_query(query)
    t2 = time.time()
    print "- Time: " + str((t2 - t1)) + "\n"

    # For each user, pull relevant data out of their resume,
    # place in a list and add to the DataFrame
    elements = ['ExecutiveSummary', 'Description', 'title',
                'Competency', 'Degree']
    print 'Processing user resume data...'
    t1 = time.time()
    l = []
    es = []
    ds = []
    ti = []
    cm = []
    dg = []
    for i in range(df.shape[0]):
        xml = df.iloc[i]['parsed_resume_xml']
        try:
            p = tokenize_xml(xml, elements)
        except Exception, e:
            print 'Tokenize failed for user id %s, error: %s' % (df.iloc[i]['id'], e)
        es.append(p.get('ExecutiveSummary'))
        ds.append(p.get('Description'))
        ti.append(p.get('title'))
        cm.append(p.get('Competency'))
        dg.append(p.get('Degree'))
        l.append(p)
    t2 = time.time()
    print "- Time: " + str((t2 - t1)) + "\n"

    df['resume_elements'] = l
    df['res_executiveSummary'] = es
    df['res_description'] = ds
    df['res_title'] = ti
    df['res_competency'] = cm
    df['res_degree'] = dg

    print 'Done.'

    _applicant_df = df
    return _applicant_df


def tokenize_xml(xml_string, *elements):
    '''
    Input
        xml_string - XML as a string
        elements - list of elements
    Output
        list of words.  Or dict?
    '''
    # dict to capture resume items to return
    resume_element_dict = {}

    if not xml_string:
        return resume_element_dict
    if elements:
        elements = elements[0]
    else:
        # Default data from the user resume xml
        elements = ['ExecutiveSummary', 'Description', 'title',
                    'Competency', 'Degree']
    tree = etree.parse(StringIO(xml_string))
    root = tree.getroot()

    # Remove namespaces
    for elem in root.getiterator():
        if not hasattr(elem.tag, 'find'): continue  # (1)
        i = elem.tag.find('}')
        if i >= 0:
            elem.tag = elem.tag[i+1:]
    objectify.deannotate(root, cleanup_namespaces=True)

    for name in elements:

        s = "//" + name
        elems = tree.findall(s)

        data = []
        #print 's: %s, elems: %s' % (s, elems)
        for elem in elems:
            #print '- elem: %s, text: %s' % (elem, elem.text)
            #print '- elem.attrib: %s' % elem.attrib
            if elem.text:
                elem_str = elem.text.strip()
            else:
                elem_str = ''
            if len(elem_str) > 0:
                if elem.text:
                    val = elem.text
                    if val not in data:
                        data.append(val)
            else:
                for val in elem.attrib.itervalues():
                    if val not in data:
                        data.append(val)
        if len(data) > 0:
            data_as_str = ', '.join(data)
            resume_element_dict[name] = data_as_str

    return resume_element_dict


class User(object):
    def __init__(self):
        pass
