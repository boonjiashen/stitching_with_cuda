import matplotlib.pyplot as plt
import numpy as np
import xml_to_sqlite_connection as x2s

INPUT = None

class DataGetter():

    @staticmethod
    def get(filename, select_statement):
        with open(filename) as fid:
            textblock = fid.read()
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
        result_tuple = zip(*cur.execute(select_statement).fetchall())
        return tuple(np.array(x) for x in result_tuple)

    @staticmethod
    # NOTE: Returns timing in seconds
    def get_cpu_timing(filename):
        select_statement = """
        SELECT numImages*numDescriptorsPerImage as n, timingMS
        FROM tbl
        """
        nn, tt = DataGetter.get(filename, select_statement)
        return nn, tt/1000

    @staticmethod
    # NOTE: Returns timing in seconds
    def get_gpu_timing(filename):
        select_statement = """
        SELECT numImages*numDescriptorsPerImage as n, inclusiveTimingMS, exclusiveTimingMS
        FROM tbl
        """
        nn, inclusive, exclusive = DataGetter.get(filename, select_statement)
        return nn, inclusive/1000, exclusive/1000


def main():

    cpu_n, cpu_timing_in_s = DataGetter.get_cpu_timing('../timingFiles/cpu.stdout')
    space_opt_n, space_opt_inclusive_in_s, space_opt_exclusive_in_s =  \
            DataGetter.get_gpu_timing('../timingFiles/gpuSpaceOptimized.stdout')
    occ_opt_n, occ_opt_inclusive_in_s, occ_opt_exclusive_in_s =  \
            DataGetter.get_gpu_timing('../timingFiles/gpuOccupancy.stdout')
    mem_coa_n, mem_coa_inclusive_in_s, mem_coa_exclusive_in_s =  \
            DataGetter.get_gpu_timing('../timingFiles/gpuCoalesced.stdout')

    # Plot
    plt.figure()
    ax = plt.gca()
    ax.set_xscale('log', basex=10)
    ax.set_yscale('log', basex=10)
    ax.set_aspect('equal')
    ax.plot(cpu_n, cpu_timing_in_s, 'o-', label='CPU')
    ax.plot(space_opt_n, space_opt_inclusive_in_s, 'o-', label="Space-optimized")
    ax.plot(occ_opt_n, occ_opt_inclusive_in_s, 'o-', label="Occupancy-optimized")
    ax.plot(mem_coa_n, mem_coa_inclusive_in_s, 'o-', label="Memory-coalesced")

    plt.ylabel('Time (s)')
    plt.xlabel(r'$n = m \times n_i$')
    plt.legend(loc='best')
    plt.title('Performance timings')
    plt.show()
    
if __name__ == "__main__":
    main()
    #input = sys.stdin
    #INPUT = open('../timingFiles/88878.stdout')
    #INPUT = open('../timingFiles/cpu.stdout')
    #main()
    #INPUT.close()
