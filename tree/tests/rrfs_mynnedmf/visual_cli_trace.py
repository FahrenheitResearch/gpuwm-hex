"""Run the real CLI and retain the stack for a resident-layout startup error."""
import runpy
import sys
import traceback


def trace_layout_error(frame, event, argument):
    if event == 'exception':
        _, error, _ = argument
        if isinstance(error, ValueError) and 'C-contiguous with entities fastest' in str(error):
            sys.settrace(None)
            traceback.print_stack(frame)
            print('Resident layout error:', error, file=sys.stderr, flush=True)
            return None
    return trace_layout_error


sys.settrace(trace_layout_error)
runpy.run_module('hexcore.cli', run_name='__main__')
