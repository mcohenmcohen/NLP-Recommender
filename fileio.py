'''
Database and file system i/o
'''
# Author:  Matt Cohen
# Python Version 2.7
import numpy as np
import pandas as pd


pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 100)

data_path = '../data/'

def get_file_data():
    us = pd.read_csv(data_path + 'users.csv')
    bz = pd.read_csv(data_path + 'businesses.csv')
    jp = pd.read_csv(data_path + 'job_postings.csv')
    ja = pd.read_csv(data_path + 'job_applications.csv')

    return us, bz, jp, ja

if __name__ == '__main__':
    users, bizs, postings, apps = get_file_data()
