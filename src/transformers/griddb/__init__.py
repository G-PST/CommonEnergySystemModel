import importlib.util
import os


def to_griddb(cesm_version, tool_version, *args, **kwargs):
    """Call transform function for specific tool version.

    Args:
        cesm_version: CESM version string, e.g. 'v0.1.0'.
            'cesm_' prefix is added to form the directory name.
        tool_version: GridDB version string, e.g. 'v0.3.0'.
    """
    dir_name = f"cesm_{cesm_version}/{tool_version}"
    module_path = os.path.join(os.path.dirname(__file__), dir_name, "to_griddb.py")

    spec = importlib.util.spec_from_file_location(f"{cesm_version}.{tool_version}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.to_griddb(*args, **kwargs)


def to_cesm(cesm_version, tool_version, *args, **kwargs):
    """Call reverse transform function (GridDB -> CESM) for specific tool version.

    Args:
        cesm_version: CESM version string, e.g. 'v0.1.0'.
            'cesm_' prefix is added to form the directory name.
        tool_version: GridDB version string, e.g. 'v0.3.0'.
    """
    dir_name = f"cesm_{cesm_version}/{tool_version}"
    module_path = os.path.join(os.path.dirname(__file__), dir_name, "to_cesm.py")

    spec = importlib.util.spec_from_file_location(f"{cesm_version}.{tool_version}.to_cesm", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.to_cesm(*args, **kwargs)


# Usage
# from transformers.griddb import to_griddb, to_cesm
# result = to_griddb("cesm_version", "tool_version", dataframes)
# result = to_cesm("cesm_version", "tool_version", "path/to/griddb.sqlite")
