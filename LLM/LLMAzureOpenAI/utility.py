# Copyright © University of Trento and DLR 2025.
# This software is proprietary to the University of Trento and DLR. Use is permitted solely within
# the Horizon Europe project “INVERSE” (Grant Agreement ID: 101136067).
# This license does not override any rights or obligations established in the Grant Agreement.
# Redistribution or use outside the project is prohibited.

import os
import yaml
import tiktoken

try:
    from utility.logger import logger
except Exception:
    import sys

    _path = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):
        if os.path.isdir(os.path.join(_path, "utility")):
            if _path not in sys.path:
                sys.path.append(_path)
            break
        new_path = os.path.dirname(_path)
        if new_path == _path:
            break
        _path = new_path
    from utility.logger import logger

    
def includeYAML(file_name: str, messages: list, system_msg: str):
    """
    :brief: Add examples from a YAML file to the list of messages
    :details: This function reads a YAML file and adds the messages to the list of messages. It recursively reads other
    YAML files included in the current YAML file.
    :param file_name: Name of the YAML file containing examples
    :param messages: List of dictionaries containing the messages
    :param system_msg: System message. If empty and the YAML file contains a system message, it will be added set, otherwise it will be ignored
    """

    logger.info("Adding examples from file: %s", file_name)
    with open(file_name) as file:
        yaml_file = yaml.load(file, Loader=yaml.FullLoader)["entries"]
        # Set headers for few-shots learning
        # Check if system_msg (a dict) has already been added
        if system_msg == "" and "system_msg" in yaml_file.keys():
            system_msg = yaml_file["system_msg"]
            messages.append(system_msg)

        # Follow the order in which the YAML file is created
        for key in yaml_file.keys():
            if key == "convo":
                for msg_id in yaml_file["convo"]:
                    if ["A", "Q"] == sorted(yaml_file["convo"][msg_id].keys()):
                        messages.append(yaml_file["convo"][msg_id]["Q"])
                        messages.append(yaml_file["convo"][msg_id]["A"])

            if key == "files":
                for file in yaml_file["files"]:
                    # Check if file has absolute path
                    if os.path.isabs(file):
                        if os.path.exists(file):
                            includeYAML(file, messages, system_msg)

                    # Get path of current yaml file and add the file
                    else:
                        abs_file = os.path.join(
                            os.path.dirname(file_name), *file.split("/")
                        )
                        includeYAML(abs_file, messages, system_msg)


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
