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

# returns an element from the xml file and Yield element if it is the right type of tag
def get_element(osm_file, tags=('node', 'way', 'relation')):
    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end':
            yield elem
            root.clear()

def is_tag_viable(element):
    # creates a re.compile that checks if the elemet is viable
    lower = re.compile(r'^([a-z]|_)*$')
    lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
    problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

    k = element.get("k")
    if re.search(lower, k):
        return True
    elif re.search(lower_colon, k):
        return True
    elif re.search(problemchars, k):
        return False
    else:
        return False

# Dictionary of all the different abriviations of street names
mapping = { "Av": "Avenue",
            "Ave": "Avenue",
            "Blvd": "Boulevard",
            "Bl": "Boulevard",
            "Ct": "Court",
            "Ctr": "Center",
            "Dr": "Drive",
            "S": "South",
            "N": "North",
            "E": "East",
            "Ln": "Lane",
            "Ml": "Mill",
            "Pk": "Park",
            "Pl": "Place",
            "Py": "Parkway",
            "Rd": "Road",
            "W": "West",
            "street": "Street",
            "St": "Street",
            "st": "Street",
            "Pkwy": "Parkway",
            "Wy": "Way",
            "Ste": "Suite"
            }

def replace(match):
    return mapping[match.group(0)]

def fix_address(string):
    # splits each word into a list.
    wordList = re.sub("[^\w]", " ",  string).split()
    fixed = []
    # For each word in wordList the matching value in the mapping dictionary is
    # is joined to a string then appended to a list.
    for i in wordList:
        p = re.sub('|'.join(r'\b%s\b' % re.escape(s) for s in mapping),
                 replace, i)
        fixed.append(p)
    # fixed is turned into a string and returned
    fixed =(" ").join(fixed)
    return fixed

def shape_way_or_node(element):
    my_dict = {}
    address = {}
    tag_dict = {}
    CREATED = [ "version", "changeset", "timestamp", "user", "uid"]
    created = {}
    key = "pos"
    for a in element.attrib:
        # attributes for latitude and longitude should be added to a "pos"
        # array, for use in geospacial indexing. Make sure the values inside
        # "pos" array are floats and not strings.
        if a == "lat":
            my_dict.setdefault(key, [])
            my_dict[key].insert(0, float(element.attrib[a]))
        elif a == "lon":
            my_dict.setdefault(key, [])
            my_dict[key].append(float(element.attrib[a]))
        # attributes in the CREATED array should be added under a key "created"
        elif a in CREATED:
            created[a] = element.attrib[a]
            my_dict["created"] = created
        else:
            my_dict[a] = element.attrib[a]
        for tag in element.iter("tag"):
            if "k" in tag.attrib:
                k = tag.get("k")
                v = tag.get("v")
                if "addr:" in k:
                    # if the second level tag "k" value starts with "addr:", it
                    # should be added to a dictionary "address"
                    # if the second level tag "k" value contains problematic characters,
                    # it should be ignored
                    if is_tag_viable(tag):
                        fixed_v = fix_address(v)
                        nk = k.split(":")
                        address[nk[1]] = fixed_v
                        my_dict["address"] = address
                # if the second level tag "k" value does not start with "addr:", but
                # contains ":", you can process it in a way that you feel is best.
                else:
                    tag_dict[k] = v
                    my_dict["tag"] = tag_dict

    return my_dict

def shape_element(element):
    json_dict = {}
    # you should process only 2 types of top level tags: "node" and "way"
    if element.tag == "node" or element.tag == "way" :
        if element.tag== "node":
            node = shape_way_or_node(element)
            json_dict["node"] = node
        else:
            way = shape_way_or_node(element)
            json_dict["way"] = way
        return json_dict
    else:
        return None

def process_map(file_in, pretty = False):
    file_out = "{0}.json".format(file_in)
    with codecs.open(file_out, "w") as fo:
        for elem in get_element(file_in):
            el = shape_element(elem)
            if el:
                if pretty:
                    fo.write(json.dumps(el, indent=2)+"\n")
                else:
                    fo.write(json.dumps(el) + "\n")
                elem.clear() # clears the working element so tree is not built
                             # in memory and bogs down computer
