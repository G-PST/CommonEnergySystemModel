"""
Round-trip test: CESM YAML -> GridDB v0.3.0 -> CESM

Tests the v0.3.0 transformer by:
1. Loading cesm-sample.yaml into CESM DataFrames
2. Forward-transforming to GridDB v0.3.0 format
3. Writing to SQLite
4. Reverse-transforming back to CESM format
5. Comparing original and round-tripped DataFrames
"""

import importlib.util
import os
import sys
import traceback

import numpy as np
import pandas as pd

# Add project paths
sys.path.insert(0, os.path.abspath('src'))
sys.path.insert(0, os.path.abspath('scripts'))


def load_cesm_sample():
    """Load CESM sample YAML into DataFrames."""
    from linkml_runtime.loaders import yaml_loader

    from core.linkml_to_dataframes import yaml_to_df
    from generated.cesm_pydantic import Dataset

    yaml_path = 'data/samples/cesm-sample.yaml'
    schema_path = 'model/cesm_v0.1.0.yaml'

    print(f"  Loading YAML: {yaml_path}")
    dataset = yaml_loader.load(yaml_path, target_class=Dataset)

    print(f"  Converting to DataFrames using schema: {schema_path}")
    cesm_dfs = yaml_to_df(dataset, schema_path=schema_path, strict=True)

    print(f"  Loaded {len(cesm_dfs)} DataFrames:")
    for name in sorted(cesm_dfs.keys()):
        df = cesm_dfs[name]
        if hasattr(df, 'shape'):
            print(f"    {name}: {df.shape}")
        else:
            print(f"    {name}: {type(df).__name__}")

    return cesm_dfs


def forward_transform(cesm_dfs):
    """Transform CESM -> GridDB v0.3.0."""
    from transformers.griddb import to_griddb

    print("  Calling to_griddb('cesm_v0.1.0', 'v0.3.0', ...)")
    griddb_dfs = to_griddb("cesm_v0.1.0", "v0.3.0", cesm_dfs)

    print(f"  Produced {len(griddb_dfs)} GridDB DataFrames:")
    for name in sorted(griddb_dfs.keys()):
        df = griddb_dfs[name]
        if hasattr(df, 'shape'):
            print(f"    {name}: {df.shape}")
        else:
            print(f"    {name}: {type(df).__name__}")

    return griddb_dfs


def write_sqlite(griddb_dfs, output_path):
    """Write GridDB DataFrames to SQLite using v0.3.0 writer."""
    writer_path = os.path.join(
        'src', 'transformers', 'griddb', 'cesm_v0.1.0', 'v0.3.0', 'to_griddb_sqlite.py'
    )
    spec = importlib.util.spec_from_file_location(
        'to_griddb_sqlite', os.path.abspath(writer_path)
    )
    writer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(writer_module)

    schema_path = os.path.join(
        'src', 'transformers', 'griddb', 'cesm_v0.1.0', 'v0.3.0', 'schema.sql'
    )

    print(f"  Schema: {schema_path}")
    print(f"  Output: {output_path}")
    writer_module.write_to_sqlite(schema_path, griddb_dfs, output_path)
    print("  SQLite database written successfully.")


def reverse_transform(db_path):
    """Transform GridDB v0.3.0 -> CESM."""
    from transformers.griddb import to_cesm

    print(f"  Reading from: {db_path}")
    print("  Calling to_cesm('cesm_v0.1.0', 'v0.3.0', ...)")
    cesm_roundtrip = to_cesm("cesm_v0.1.0", "v0.3.0", db_path)

    print(f"  Produced {len(cesm_roundtrip)} CESM DataFrames:")
    for name in sorted(cesm_roundtrip.keys()):
        df = cesm_roundtrip[name]
        if hasattr(df, 'shape'):
            print(f"    {name}: {df.shape}")
        else:
            print(f"    {name}: {type(df).__name__}")

    return cesm_roundtrip


def compare_dataframes(original, roundtrip):
    """Compare original and round-tripped DataFrames and print a detailed report."""
    print("\n" + "=" * 80)
    print("COMPARISON REPORT")
    print("=" * 80)

    orig_keys = set(original.keys())
    rt_keys = set(roundtrip.keys())

    common_keys = sorted(orig_keys & rt_keys)
    only_original = sorted(orig_keys - rt_keys)
    only_roundtrip = sorted(rt_keys - orig_keys)

    # Summary counts
    match_count = 0
    diff_count = 0
    shape_mismatch_count = 0

    # --- Common DataFrames ---
    print(f"\nCommon DataFrames ({len(common_keys)}):")
    print("-" * 80)

    for key in common_keys:
        orig_df = original[key]
        rt_df = roundtrip[key]

        # Handle non-DataFrame objects (e.g., timeline)
        if not isinstance(orig_df, pd.DataFrame) or not isinstance(rt_df, pd.DataFrame):
            print(f"\n  {key}: non-DataFrame type, skipping detailed comparison")
            continue

        print(f"\n  {key}:")
        print(f"    Original shape:    {orig_df.shape}")
        print(f"    Round-trip shape:  {rt_df.shape}")

        if orig_df.shape != rt_df.shape:
            shape_mismatch_count += 1
            print("    -> SHAPE MISMATCH")

            # Show column differences
            orig_cols = set(str(c) for c in orig_df.columns)
            rt_cols = set(str(c) for c in rt_df.columns)
            if orig_cols != rt_cols:
                only_orig_cols = sorted(orig_cols - rt_cols)
                only_rt_cols = sorted(rt_cols - orig_cols)
                if only_orig_cols:
                    print(f"    Columns only in original:   {only_orig_cols}")
                if only_rt_cols:
                    print(f"    Columns only in round-trip: {only_rt_cols}")

            # Show index differences for entity tables
            if not key.endswith(('.ts.', )) and '.ts.' not in key:
                orig_idx = set(str(i) for i in orig_df.index)
                rt_idx = set(str(i) for i in rt_df.index)
                only_orig_idx = sorted(orig_idx - rt_idx)
                only_rt_idx = sorted(rt_idx - orig_idx)
                if only_orig_idx:
                    print(f"    Entities only in original:   {only_orig_idx}")
                if only_rt_idx:
                    print(f"    Entities only in round-trip: {only_rt_idx}")

            diff_count += 1
            continue

        # Same shape -- compare contents
        try:
            # Align columns and index for comparison
            shared_cols = sorted(
                set(str(c) for c in orig_df.columns) & set(str(c) for c in rt_df.columns)
            )

            if not shared_cols:
                print("    -> No shared columns to compare")
                diff_count += 1
                continue

            # Try numeric comparison on shared columns
            orig_subset = orig_df.reindex(columns=shared_cols)
            rt_subset = rt_df.reindex(columns=shared_cols)

            # Check if DataFrames are numeric
            orig_numeric = orig_subset.select_dtypes(include=[np.number])
            rt_numeric = rt_subset.select_dtypes(include=[np.number])

            if not orig_numeric.empty and orig_numeric.shape == rt_numeric.shape:
                # Compute relative difference (handle zeros)
                with np.errstate(divide='ignore', invalid='ignore'):
                    abs_diff = np.abs(orig_numeric.values - rt_numeric.values)
                    max_abs = np.nanmax(np.abs(orig_numeric.values), axis=0)
                    max_abs[max_abs == 0] = 1.0  # avoid divide by zero
                    rel_diff = abs_diff / max_abs

                max_rel = np.nanmax(rel_diff) if rel_diff.size > 0 else 0.0
                mean_rel = np.nanmean(rel_diff) if rel_diff.size > 0 else 0.0
                max_abs_diff = np.nanmax(abs_diff) if abs_diff.size > 0 else 0.0

                if max_abs_diff == 0.0:
                    print("    -> NUMERIC MATCH (exact)")
                    match_count += 1
                elif max_rel < 1e-6:
                    print(f"    -> NUMERIC MATCH (max relative diff: {max_rel:.2e})")
                    match_count += 1
                else:
                    print("    -> NUMERIC DIFFERENCE")
                    print(f"       Max absolute diff: {max_abs_diff:.6g}")
                    print(f"       Max relative diff: {max_rel:.6g}")
                    print(f"       Mean relative diff: {mean_rel:.6g}")
                    diff_count += 1
            else:
                # Non-numeric or mixed -- use generic equality
                try:
                    # Reset index for comparison if indices differ in type
                    o = orig_subset.reset_index(drop=True)
                    r = rt_subset.reset_index(drop=True)
                    if o.equals(r):
                        print("    -> MATCH (exact)")
                        match_count += 1
                    else:
                        # Count differing cells
                        diff_mask = o != r
                        n_diff = diff_mask.sum().sum()
                        n_total = diff_mask.size
                        print(f"    -> DIFFERS ({n_diff}/{n_total} cells differ)")
                        diff_count += 1
                except Exception as cmp_err:
                    print(f"    -> COMPARISON ERROR: {cmp_err}")
                    diff_count += 1

        except Exception as e:
            print(f"    -> COMPARISON ERROR: {e}")
            diff_count += 1

    # --- Only in original ---
    if only_original:
        print(f"\nDataFrames only in original ({len(only_original)}):")
        print("-" * 80)
        for key in only_original:
            df = original[key]
            shape = df.shape if hasattr(df, 'shape') else 'N/A'
            print(f"  {key}: {shape}")

    # --- Only in round-trip ---
    if only_roundtrip:
        print(f"\nDataFrames only in round-trip ({len(only_roundtrip)}):")
        print("-" * 80)
        for key in only_roundtrip:
            df = roundtrip[key]
            shape = df.shape if hasattr(df, 'shape') else 'N/A'
            print(f"  {key}: {shape}")

    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Original DataFrames:     {len(orig_keys)}")
    print(f"  Round-trip DataFrames:   {len(rt_keys)}")
    print(f"  Common DataFrames:       {len(common_keys)}")
    print(f"    Matching:              {match_count}")
    print(f"    Content differences:   {diff_count}")
    print(f"    Shape mismatches:      {shape_mismatch_count}")
    print(f"  Only in original:        {len(only_original)}")
    print(f"  Only in round-trip:      {len(only_roundtrip)}")

    if only_original:
        print("\n  Note: DataFrames only in original may include data that the")
        print("  transformer intentionally filters (e.g., heat network units")
        print("  like bio_chp/heatpump that are not in the power_grid domain,")
        print("  or units without units_existing counts).")

    if match_count == len(common_keys) and not only_original and not only_roundtrip:
        print("\n  RESULT: PERFECT ROUND-TRIP")
    elif match_count == len(common_keys):
        print("\n  RESULT: All common DataFrames match, but coverage differs.")
    else:
        print("\n  RESULT: Round-trip has differences (see details above).")

    print("=" * 80)


def main():
    print("=" * 80)
    print("ROUND-TRIP TEST: CESM -> GridDB v0.3.0 -> CESM")
    print("=" * 80)

    # Create artifacts directory
    os.makedirs('artifacts', exist_ok=True)
    output_db = 'artifacts/test_griddb_v030.sqlite'

    # Track which steps succeeded
    cesm_original = None
    griddb_dfs = None
    cesm_roundtrip = None

    # Step 1: Load CESM sample
    print("\n[Step 1] Loading CESM sample...")
    try:
        cesm_original = load_cesm_sample()
        print("  [OK] CESM sample loaded successfully.")
    except Exception as e:
        print(f"  [FAIL] Error loading CESM sample: {e}")
        traceback.print_exc()

    # Step 2: Forward transform
    if cesm_original is not None:
        print("\n[Step 2] Forward transform (CESM -> GridDB v0.3.0)...")
        try:
            griddb_dfs = forward_transform(cesm_original)
            print("  [OK] Forward transform completed.")
        except Exception as e:
            print(f"  [FAIL] Error in forward transform: {e}")
            traceback.print_exc()
    else:
        print("\n[Step 2] SKIPPED (Step 1 failed)")

    # Step 3: Write to SQLite
    if griddb_dfs is not None:
        print("\n[Step 3] Writing to SQLite...")
        try:
            write_sqlite(griddb_dfs, output_db)
            print("  [OK] SQLite database created.")
        except Exception as e:
            print(f"  [FAIL] Error writing SQLite: {e}")
            traceback.print_exc()
            # If write failed, the DB may not be usable
            output_db = None
    else:
        print("\n[Step 3] SKIPPED (Step 2 failed)")
        output_db = None

    # Step 4: Reverse transform
    if output_db is not None:
        print("\n[Step 4] Reverse transform (GridDB v0.3.0 -> CESM)...")
        try:
            cesm_roundtrip = reverse_transform(output_db)
            print("  [OK] Reverse transform completed.")
        except Exception as e:
            print(f"  [FAIL] Error in reverse transform: {e}")
            traceback.print_exc()
    else:
        print("\n[Step 4] SKIPPED (Step 3 failed)")

    # Step 5: Compare
    if cesm_original is not None and cesm_roundtrip is not None:
        print("\n[Step 5] Comparing DataFrames...")
        try:
            compare_dataframes(cesm_original, cesm_roundtrip)
        except Exception as e:
            print(f"  [FAIL] Error comparing DataFrames: {e}")
            traceback.print_exc()
    else:
        print("\n[Step 5] SKIPPED (missing original or round-trip data)")
        if cesm_original is None:
            print("  Reason: Original CESM data not available")
        if cesm_roundtrip is None:
            print("  Reason: Round-trip CESM data not available")

    print("\n" + "=" * 80)
    print("ROUND-TRIP TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
