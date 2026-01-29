import pandas as pd
import generated.cesm as cesm  # This is the generated class

def to_griddb(cesm):
    """
    Transform data from CESM dataframes to GridDB format dataframes.
    """
    # Create time resolution dataframe
    dt_series = pd.to_datetime(cesm["timeline"])
    time_diffs = -dt_series.diff(-1).total_seconds() / 3600
    time_diffs = pd.Series(time_diffs)
    time_diffs.iloc[-1] = time_diffs.iloc[-2]
    
    timeline = pd.DataFrame({
        'datetime': cesm["timeline"],
        'cesm_timeline': time_diffs
    })

    flextool['timeline'] = pd.DataFrame({
        'name': ['cesm_timeline']
    })
    flextool['timeline.table.timestep_duration'] = timeline

    # Create solve.table.period_timeset dataframe
    periods = set()
    periods.update(cesm['solve_pattern']['periods_realise_investments'].iloc[0])
    periods.update(cesm['solve_pattern']['periods_realise_operations'].iloc[0])
    periods.update(cesm['solve_pattern']['periods_additional_horizon'].iloc[0])

    flextool['timeset'] = pd.DataFrame({
        'name': ['cesm_timeset'],
        'timeline': ['cesm_timeline']
    })
    period_timeset = pd.DataFrame({'period': list(periods)})
    for solve_name in flextool['solve']['name']:
        period_timeset[solve_name] = 'cesm_timeset'
    
    flextool['solve.table.period_timeset'] = period_timeset

    return flextool