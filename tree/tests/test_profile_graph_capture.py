"""Do not report graph execution as device idle in an incomplete trace."""
from pathlib import Path
import importlib.util
import sqlite3
import pytest

def analyzer():
    path=Path(__file__).parents[1]/"tools/hex_speed_profile/analyze_nsys_profile.py"
    spec=importlib.util.spec_from_file_location("profile_graph_test",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_graph_level_activities_refuse_per_kernel_accounting():
    c=sqlite3.connect(":memory:")
    c.execute("create table CUPTI_ACTIVITY_KIND_GRAPH_TRACE(start integer, end integer)")
    c.execute("insert into CUPTI_ACTIVITY_KIND_GRAPH_TRACE values(100,200)")
    with pytest.raises(ValueError,match="--cuda-graph-trace=node"):
        analyzer().require_graph_node_capture(c)

def test_no_whole_graph_activities_admits_node_accounting():
    c=sqlite3.connect(":memory:")
    analyzer().require_graph_node_capture(c)
    c.execute("create table CUPTI_ACTIVITY_KIND_GRAPH_TRACE(start integer, end integer)")
    analyzer().require_graph_node_capture(c)
