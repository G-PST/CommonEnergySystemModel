def print_dataframes_structure(df_dict, max_items=3, output_file='dataframes_structure.txt'):
    import pandas as pd
    import sys
    
    def format_list(items, max_n=max_items):
        items_list = list(items)
        if len(items_list) <= max_n:
            return items_list
        return items_list[:max_n] + [f'... ({len(items_list)} total)']
    
    with open(output_file, 'a') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        
        for key, obj in df_dict.items():
            print(f"\n{key}")
            
            if isinstance(obj, pd.DataFrame):
                print("Type: DataFrame")
                print(f"Shape: {obj.shape}")
                print(f"Index: {obj.index.names if hasattr(obj.index, 'names') else 'default'}")
                print(f"Columns: {format_list(obj.columns)}")
                print(f"Dtypes:\n{obj.dtypes}")
                
            elif isinstance(obj, pd.Series):
                print("Type: Series")
                print(f"Shape: {obj.shape}")
                print(f"Index: {obj.index.name or 'default'}")
                print(f"Dtype: {obj.dtype}")
                
            elif isinstance(obj, pd.Index):
                print("Type: Index")
                print(f"Name: {obj.name}")
                print(f"Values: {format_list(obj)}")
                print(f"Dtype: {obj.dtype}")
        
        sys.stdout = original_stdout