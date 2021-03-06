import os
import time

import functools
import contextlib

from prettytable import PrettyTable

code_timer_stats = {}

#Code block timer used as a "with" block (taken from https://stackoverflow.com/questions/30433910/decorator-to-time-specific-lines-of-the-code-instead-of-whole-method)
def function_timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print("Finished {} in {} secs".format(repr(func.__name__), round(run_time, 3)))
        return value

    return wrapper

#Function timer that accumulates execution time statistics. To print these statistics use the print_code_stats()
#function below
@contextlib.contextmanager
def code_timer(ident, debug = True, print_single_block_info=False):
    if debug:
        tstart = time.time()
        yield
        elapsed = time.time() - tstart
        if print_single_block_info: print("{0}: {1} ms".format(ident, elapsed))
        if ident not in code_timer_stats:
            code_timer_stats[ident] = {"avg_time" : 0, "total_time" : 0}
        avg_time = (code_timer_stats[ident]["avg_time"] + elapsed)/2
        total_time = code_timer_stats[ident]["total_time"] + elapsed
        code_timer_stats[ident] = {"avg_time": avg_time, "total_time": total_time}
    else:
        yield

#Prints the statistics accumulated by the context manager function code_timer (see above)
def print_code_stats():
    table = PrettyTable(['Name','Avg. time', 'Total time'])
    sort_by_row_value_index = 3
    for name, entry in code_timer_stats.items():
        row = [name]
        for _, value in entry.items():
            row.append("{:.4f}".format(value))
        table.add_row(row)
    table = table.get_string(sort_key=lambda row: row[sort_by_row_value_index], sortby="Total time", reversesort=True)
    print(table)
