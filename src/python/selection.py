from PQAnalysis.io import read_restart_file
from PQAnalysis.topology import Selection


def selection(selection_string: str, restart_file: str, moldescriptor_file: str | None = None):
    system = read_restart_file(
        restart_file,
        moldescriptor_filename=moldescriptor_file
    )

    sel = Selection(selection_string)

    indices = sel.select(system.topology)

    return indices
