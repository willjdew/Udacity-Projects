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

def create_connection(db_file):
    #create a database connection to the SQLite database specified by db_file
    # retuns connection if db was made or None if db was not.
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except IOError as e:
        print(e)

    return None


def create_table(conn, create_table_sql):
    #create a table from the create_table_sql statement
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except IOError as e:
        print(e)

 def create_query_string(table, sql_list):
    # creates strings of the keys, and a string of ? equal to the number of keys
    # and adds them the the sql query
    sql_string = (", ").join(sql_list)
    q_list = ["?"] * len(sql_list)
    q_string = (",").join(q_list)
    sql = """INSERT INTO %s(%s) VALUES(%s);""" % (table, sql_string, q_string)

    return sql

def make_node_way_tuple(node, type_node):
    node_id = lat = lon = uid = user = version = changeset = timestamp = None
    node_sql_list = node_list = []
    node_id = int(node["id"])
    node_dict = {}
    table = type_node
    if "pos" in node.keys():
        lat = node["pos"][0]
        lon = node["pos"][1]
    if "created" in node.keys():
        created = node["created"]
        if "uid" in created.keys(): uid = int(created["uid"])
        if "user" in created.keys(): user = created["user"]
        if "version" in created.keys(): version = int(created["version"])
        if "changeset" in created.keys(): changeset = int(created["changeset"])
        if "timestamp" in created.keys(): timestamp = created["timestamp"]
    if type_node == "node":
        node_sql_list = ["id", "lat", "lon", "uid", "user", "version", "changeset", "timestamp"]
        node_list = [node_id, lat, lon , uid, user, version, changeset, timestamp]
    elif type_node == "way":
        node_sql_list = ["id", "uid", "user", "version", "changeset", "timestamp"]
        node_list = [node_id, uid, user, version, changeset, timestamp]

    node_query = create_query_string(table, node_sql_list)
    node_t = tuple(node_list)
    node_dict[node_query] = node_t

    return node_dict

def make_tag_address_tuple(node, type_node, address=False, tag=False, n_id=False):
    sql_list = []
    values_list = []
    list_of_list = []
    node_id = way_id = key = value = n_type = None

    if type_node == "node" and address == False and tag == False:
        node_id = int(node["id"])
    if type_node == "way" and address == False and tag == False:
        way_id = int(node["id"])
    if address and type_node == "node": node_id = int(n_id)
    if address and type_node == "way": way_id = int(n_id)
    if tag and type_node == "node": node_id = int(n_id)
    if tag and type_node == "way": way_id = int(n_id)
    if "pos" in node.keys(): del node["pos"]
    if "created" in node.keys(): del node["created"]
    if "address" in node.keys(): del node["address"]


    for k in node.keys():
        a_list = []
        a_sql = []
        key = k
        value = node[k]

        if (value.isdigit()) and (k != 'postcode' or k != 'zipcode'):
            value = float(value)
            n_type = "float"
        else:
            n_type = "string"
        if address:
            a_sql = ["node_id", "way_id", "key", "value", "type"]
            a_list = [node_id, way_id, key, value, n_type]
            table = "address"
        elif type_node == "node":
            a_sql = ["id", "key", "value", "type"]
            a_list = [node_id, key, value, n_type]
            table = "node_tag"
        else:
            a_sql = ["id", "key", "value", "type"]
            a_list = [way_id, key, value, n_type]
            table = "way_tag"

        t = tuple(a_list)
        sql_list.append(create_query_string(table, a_sql))
        values_list.append(t)

    list_of_list = [sql_list, values_list]

    return list_of_list

def insert_value(n_dict, conn):
    node_t = node_tag_sql = address_sql = way_t = way_tag_sql = None
    if "node" in n_dict.keys():
        node_t = make_node_way_tuple(n_dict["node"], "node")
        if "tag" in n_dict["node"]:
            node_tag_sql = make_tag_address_tuple(n_dict["node"]["tag"], "node", False, True, n_dict["node"]["id"])
        if "address" in n_dict["node"].keys():
            address_sql = make_tag_address_tuple(n_dict["node"]["address"], "node", True, False, n_dict["node"]["id"])
    if "way" in n_dict.keys():
        way_t = make_node_way_tuple(n_dict["way"], "way")
        if "address" in n_dict["way"].keys():
            address_sql = make_tag_address_tuple(n_dict["way"]["address"], "way", True, False, n_dict["way"]["id"])
        if "tag" in n_dict["way"]:
            node_tag_sql = make_tag_address_tuple(n_dict["way"]["tag"], "way", False, True, n_dict["way"]["id"])

    top_list = [node_t, way_t]
    bottom_list = [node_tag_sql, address_sql, way_tag_sql]

    for dict_i in top_list:
        if dict_i:
            for key in dict_i:
                sql_query = key
                t = dict_i[key]
                conn.execute(sql_query, t)
                conn.commit()
    for list_d in bottom_list:
        if list_d:
            sql_list_query = list_d[0]
            t_list = list_d[1]
            for i, item in enumerate(sql_list_query):
                sql_q = item
                t_q = t_list[i]
                conn.execute(sql_q, t_q)
                conn.commit()
