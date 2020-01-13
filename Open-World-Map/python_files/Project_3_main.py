import codecs
import json
import pprint
import re
import sqlite3
import string
import time
import xml.etree.ElementTree as ET
from collections import Counter
from collections import defaultdict
import Project_3_create_json as cj
import Project_3_create_db as cdb

t0 = time.time()
cj.process_map("moab.osm")
print 'moab.osm.json file created'
t1 = time.time()
total = t1-t0
print 'Time to execute', total, "\n"

t0 = time.time()
cdb.create_tables("moab.db")
conn = cdb.create_connection("moab.db")
with open('moab.osm.json') as f:
    for line in f:
        data=json.loads(line)
        cdb.insert_value(data, conn)
    f.close()

conn.close()
print 'moab.db file created'
t1 = time.time()
total = t1-t0
print 'Time to execute', total, "\n"
