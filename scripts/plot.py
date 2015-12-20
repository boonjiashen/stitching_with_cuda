import matplotlib.pyplot as plt
import numpy as np
import xml_to_sqlite_connection as x2s

INPUT = None

def main():

    textblock = INPUT.read()
    textblock = "<body>\n" + textblock + "\n</body>"

    # Create SQL table from XML file
    con = x2s.xml_to_sqlite_connection(textblock)
    cur = con.cursor()

    # Enables stdev functionality
    # Need to first compile extension-functions.c with
    # gcc -g -shared -fPIC extension-functions.c -o extension-functions.so
    con.enable_load_extension(True)
    con.execute("select load_extension('./extension-functions.so')")
    con.enable_load_extension(False)

    # Get data amenable to plotting
    select_statement = """
    SELECT numImages*numDescriptorsPerImage as n, inclusiveTimingMS
    FROM tbl
    """
    nn, tt = zip(*cur.execute(select_statement).fetchall())
    print(nn, tt)

    # Plot
    plt.figure()
    ax = plt.gca()
    ax.set_xscale('log', basex=10)
    ax.set_yscale('log', basex=10)
    ax.plot(nn, tt, 'o-')
    ax.plot([1, 2], [1, 2])
    #plt.ylim(0, plt.ylim()[-1])
    #plt.xlim(0, plt.xlim()[-1])
    plt.show()
    
if __name__ == "__main__":
    #input = sys.stdin
    INPUT = open('../timingFiles/88878.stdout')
    main()
    INPUT.close()
