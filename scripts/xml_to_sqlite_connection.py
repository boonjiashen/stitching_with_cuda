import xml.etree.ElementTree as ET
import sqlite3

def xml_to_sqlite_connection(xml_string_block):
    """Returns the cursor to an in-memory SQLite DB given the root element of
    an XML.
    
    Assumes every child of the root is a bodyless element
    where every child has the same tags in the same order
    """

    root = ET.fromstring(xml_string_block)
    key_tuple = tuple(root[0].keys())

    # Create table
    con = sqlite3.connect(":memory:")
    cur = con.cursor()
    cur.execute("CREATE TABLE tbl %s;" % str(key_tuple))

    # Insert records
    insert_statement = "INSERT INTO tbl VALUES (%s)" %  \
            ','.join(['?'] * len(key_tuple))
    for xml_row in root:
        params = [eval(xml_row.get(key)) for key in xml_row.keys()]
        cur.execute(insert_statement, params)
    con.commit()

    return con
