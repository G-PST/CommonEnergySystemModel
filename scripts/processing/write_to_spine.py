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


def main():
    if len(sys.argv) < 2:
        print("Please provide input yaml file and output database url as arguments.")
        print("Usage: python write_to_spine.py <yaml-file> <db-url> (i.e. sqlite:///path/db_file.sqlite)")
        sys.exit(1)

    yaml_file = sys.argv[1]
    url_db_out = sys.argv[2]

    try:
        yaml_data = read_yaml_file(yaml_file)
        print("Sample loaded")
    except FileNotFoundError:
        print(f"{yaml_file} not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        sys.exit(1)

    purge.purge_url(url=url_db_out, purge_settings={"entity": True, "alternative": True})

    with DatabaseMapping(url_db_out, upgrade=True) as target_db:
        try:
            target_db.add_alternative(name="base")
        except:
            pass
        target_db.commit_session("foo")
        if len(yaml_data["system"]) > 1:
            print("more than 1 system definition - can't handle")
            sys.exit(1)
        try:
            timeline = yaml_data["system"][0]["timeline"]
            system_name = yaml_data["system"][0]["name"]
        except:
            print("no timeline in dataset")
            sys.exit(1)
        for class_name_orig, entities in yaml_data.items():
            class_name = class_name_orig.replace('balance', 'node')
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

            if class_name == 'node':
                try:
                    target_db.add_parameter_definition(
                        entity_class_name=class_name,
                        name='node_type',
                    )
                except:
                    pass

            for entity in entities:
                for attribute, value in entity.items():
                    if (attribute=="id"):
                        continue
                    if (attribute=="name"):
                        entity_name_tuple = tuple(value.split("__"))
                        try:
                            target_db.add_entity(
                                name=value,
                                entity_class_name=class_name,
                                entity_byname=entity_name_tuple,
                            )
                        except:
                            pass
                        continue
                    try:
                        target_db.add_parameter_definition(
                            entity_class_name=class_name,
                            name=attribute,
                        )
                    except:
                        pass
                    if type(value) is list:
                        # value = str(value)
                        if attribute == 'timeline':
                            value = api.Map(
                                timeline,
                                ['1.0'] * len(timeline)
                            )
                        else: 
                            value = api.Map(
                                timeline,
                                value
                            )
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

                if class_name == 'node':
                    try:
                        target_db.add_parameter_value(
                                entity_class_name=class_name,
                                parameter_definition_name='node_type',
                                entity_byname=entity_name_tuple,
                                alternative_name="base",
                                parsed_value=class_name_orig,
                                type="str"
                        )
                    except:
                        pass
        target_db.commit_session("foo")
        print("Data committed")



if __name__ == "__main__":
    main()
