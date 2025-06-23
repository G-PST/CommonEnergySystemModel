import yaml
import os
import sys
from pathlib import Path
import spinedb_api as api
from spinedb_api import DatabaseMapping
from spinedb_api import purge


def read_yaml_file(file_path):
    """Read a single YAML file and return as dictionary"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    try:
        sample = read_yaml_file('ines-sample.yaml')
        print("Sample loaded:")
    except FileNotFoundError:
        print("sample.yaml not found")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
    
    
if len(sys.argv) > 1:
    yaml_data = sys.argv[1]
else:
    exit("Please provide input yaml file and output database url as arguments. The latter should be of the form ""sqlite:///path/db_file.sqlite""")
if len(sys.argv) > 2:
    url_db_out = sys.argv[2]
else:
    exit("Please provide output database url as the 2nd argument. It should be of the form ""sqlite:///path/db_file.sqlite""")


def main():
    purge.purge_url(url=url_db_out, purge_settings={"entity": True, "alternative": True})

    with DatabaseMapping(url_db_out, upgrade=True) as target_db:
        try:
            target_db.add_alternative(name="base")
        except:
            pass
        target_db.commit_session("foo")
        for class_name, entities in sample.items():
            class_name = class_name.replace('balance', 'node')
            class_name = class_name.replace('commodity', 'node')
            class_name = class_name.replace('storage', 'node')
            class_name_tuple = tuple(class_name.replace('__to_', '__').split("__"))

            if len(class_name_tuple) == 1:
                try:
                    target_db.add_entity_class(
                        name=class_name,
                        entity_class_byname=class_name_tuple
                    )
                except:
                    pass
            else:
                for class_dimen in class_name_tuple:
                    try:
                        target_db.add_entity_class(
                            name=class_dimen,
                            entity_class_byname=(class_dimen,)
                        )
                    except:
                        pass
                try:
                    target_db.add_entity_class(
                        name=class_name,
                        dimension_name_list=class_name_tuple,
                        entity_class_byname=class_name_tuple
                    )
                except:
                    pass

            for entity in entities:
                for attribute, value in entity.items():
                    if (attribute=="id"):
                        continue
                    if (attribute=="name"):
                        entity_name_tuple = tuple(value.split("__"))
                        target_db.add_entity(
                            name=value,
                            entity_class_name=class_name,
                            entity_byname=entity_name_tuple,
                        )
                        continue
                    try:
                        target_db.add_parameter_definition(
                            entity_class_name=class_name,
                            name=attribute,
                        )
                    except:
                        pass
                    if type(value) is list:
                        value = str(value)
                    try:
                        target_db.add_parameter_value(
                            entity_class_name=class_name,
                            parameter_definition_name=attribute,
                            entity_byname=entity_name_tuple,
                            alternative_name="base",
                            parsed_value=value,
                            type="str"
                        )
                    except:
                        pass
        target_db.commit_session("foo")

                    



if __name__ == "__main__":
    main()
