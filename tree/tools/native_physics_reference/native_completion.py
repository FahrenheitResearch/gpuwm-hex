"""Reject native I/O failures even if a forecast prints its final banner."""
import re


def completed(text,core):
    # Witness: N1 MPI_File_write_at ENOSPC produced ERROR SMIOL writes. A final
    # banner alone cannot certify initialized/history field bytes after that.
    return ('Finished running the '+core+' core' in text
            and not re.search(r'(?m)^\s*(?:ERROR|CRITICAL|FATAL)(?::|\b)',text))
