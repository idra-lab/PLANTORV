# Copyright © University of Trento and DLR 2025.
# This software is proprietary to the University of Trento and DLR. Use is permitted solely within
# the Horizon Europe project “INVERSE” (Grant Agreement ID: 101136067).
# This license does not override any rights or obligations established in the Grant Agreement.
# Redistribution or use outside the project is prohibited.

# This file is the main file which coordinates the different aspects of the planner. 

import os
import re
import sys
from typing import Tuple

try:
    from .llm_gen_support import (
        LLM_CONF_PATH,
        CC_EXAMPLES_CONFIG_PATH,
        CC_EXAMPLES_HL_PATH,
        OUTPUT_FILE_CC,
        write_to_file,
        read_from_file,
        parse_arguments,
        _read_queries_from_files,
        set_default_args,
        build_llm, 
        extract_cc_verdict,
    )
except Exception:
    try:
        from llm_gen_support import (
            LLM_CONF_PATH,
            CC_EXAMPLES_CONFIG_PATH,
            CC_EXAMPLES_HL_PATH,
            OUTPUT_FILE_CC,
            write_to_file,
            read_from_file,
            parse_arguments,
            _read_queries_from_files,
            set_default_args,
            build_llm, 
            extract_cc_verdict,
        )
    except Exception:
        sys.path.append(os.path.dirname(__file__))
        from llm_gen_support import (
            LLM_CONF_PATH,
            CC_EXAMPLES_CONFIG_PATH,
            CC_EXAMPLES_HL_PATH,
            OUTPUT_FILE_CC,
            write_to_file,
            read_from_file,
            parse_arguments,
            _read_queries_from_files,
            set_default_args,
            build_llm, 
            extract_cc_verdict,
        )

try:
    from .llm_gen_multi import hl_llm_multi_step, ll_llm_multi_step
except Exception:
    try:
        from llm_gen_multi import hl_llm_multi_step, ll_llm_multi_step
    except Exception:
        sys.path.append(os.path.dirname(__file__))
        from llm_gen_multi import hl_llm_multi_step, ll_llm_multi_step

try:
    from .llm_gen_single import hl_llm_single_step, ll_llm_single_step
except Exception:
    try:
        from llm_gen_single import hl_llm_single_step, ll_llm_single_step
    except Exception:
        sys.path.append(os.path.dirname(__file__))
        from llm_gen_single import hl_llm_single_step, ll_llm_single_step

try:
    from utility.logger import logger
except Exception:
    def _find_repo_root(start_path):
        path = os.path.abspath(start_path)
        for _ in range(8):
            candidate = os.path.join(path, "utility", "logger.py")
            if os.path.isfile(candidate):
                return path
            new_path = os.path.dirname(path)
            if new_path == path:
                break
            path = new_path
        return None

    _root = _find_repo_root(os.path.dirname(__file__))
    if _root and _root not in sys.path:
        sys.path.insert(0, _root)

    from utility.logger import logger


########################################################################################################################


def _normalize_cc_response(response: str) -> str:
    text = str(response or "").strip()
    if text == "":
        return text

    # Strip hidden-thought tags if the model emitted them.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    # If model emits a verbose heading, remove it before parsing.
    text = re.sub(r"(?is)^\s*thinking process\s*:\s*", "", text).strip()

    verdict_match = re.search(r"\b(OK|PROBLEM)\b", text, flags=re.IGNORECASE)
    verdict = verdict_match.group(1).upper() if verdict_match else ""

    explanation = ""
    explanation_match = re.search(
        r"^\s*EXPLANATION\s*:\s*(.+)$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if explanation_match is not None:
        explanation = explanation_match.group(1).strip()
    else:
        for raw_line in text.splitlines():
            line = raw_line.strip(" \t-*")
            if line == "":
                continue
            lower = line.lower()
            if lower.startswith(
                (
                    "thinking process",
                    "analyze the",
                    "evaluate feasibility",
                    "verdict:",
                    "explanation:",
                )
            ):
                continue
            if line.upper() in {"OK", "PROBLEM"}:
                continue
            explanation = line
            break

    if verdict != "" and explanation != "":
        return "VERDICT: {}\nEXPLANATION: {}".format(verdict, explanation)
    return text


########################################################################################################################


def llm_scenario_comprehension(
    query_hl,
    query_ll,
    llm_config_file=LLM_CONF_PATH,
    output_file_cc=OUTPUT_FILE_CC,
    cc_examples_hl_path=CC_EXAMPLES_HL_PATH,
    cc_examples_config_path=CC_EXAMPLES_CONFIG_PATH,
) -> Tuple[bool, str]:
    parent_dir = os.path.dirname(output_file_cc)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        
    file = open(output_file_cc, "w+")

    logger.info("[CC] Checking LLM comprehension of scenario for high-level")
    llm_scenario = build_llm(
        llm_config_file=llm_config_file,
        examples_yaml_file=[cc_examples_hl_path],
    )
    # Scenario comprehension should return a verdict plus a short rationale.
    # Keep this bounded to avoid long reasoning traces and latency spikes.
    llm_scenario.max_tokens = min(int(llm_scenario.max_tokens), 128)

    logger.info("[CC] Checking LLM comprehension of scenario for high-level")
    scenario_query_hl = (
        f"Given the following high-level scenario:\n{query_hl}\n"
        "When physical dimensions, geometry, positions, reachability, or "
        "spatial support are mentioned, use them to check feasibility. Do not"
        "treat object roles such as support, bridge, architrave, cover," 
        "container, or carrier as purely symbolic labels if the description" 
        "gives dimensions or positions. Return PROBLEM when the stated" 
        "dimensions or positions make the described physical arrangement impossible."
        "Return ONLY these two lines and nothing else:\n"
        "VERDICT: OK or PROBLEM\n"
        "EXPLANATION: one or two short sentences only.\n"
        "Do not provide chain-of-thought, step-by-step reasoning, internal deliberation, headings, or numbered lists."
    )
    succ, response = llm_scenario.query(scenario_query_hl)
    response = _normalize_cc_response(response)
    verdict = extract_cc_verdict(response)
    if succ and verdict == "OK":
        logger.info(f"\rLLM has correctly understood the scenario\n{response}") 
        file.write(f"HL: {response}\n")
    elif succ and verdict == "PROBLEM":
        logger.error(f"\rLLM has not correctly understood the scenario or there is a problem in the scenario\n{response}")
        file.write(f"LLM has not correctly understood the scenario or there is a problem in the scenario\n{response}")
        return False, response
    else: 
        logger.error(f"Problem with the LLM\n{response}")
        sys.exit(1)

    # Check comprehension of both the high-level and low-level scenarios

    if hasattr(llm_scenario, "close"):
        llm_scenario.close()
        del llm_scenario
    llm_scenario = build_llm(
        llm_config_file=llm_config_file,
        examples_yaml_file=[cc_examples_config_path],
    )
    llm_scenario.max_tokens = min(int(llm_scenario.max_tokens), 128)

    logger.info("[CC] Checking LLM comprehension of scenario for both levels")
    scenario_query = (
        f"Given the following high-level description of a scenario:\n{query_hl}\n"
        f"And the following low-level description of the same scenario\n{query_ll}\n"
        "The low-level description may introduce internal state variables and constraints, "
        # "such as arm poses, gripper states, home positions, or controller conventions, "
        "even if they are absent from the high-level description. This is not a problem "
        "unless they contradict an explicit high-level requirement or make the task infeasible.\n"
        "When checking high-level and low-level consistency, do not mark the scenario as PROBLEM only because high-level and low-level action durations differ. Low-level actions may decompose abstract high-level actions and their durations are not required to match exactly. Only treat durations as a problem if the descriptions omit required duration information."
        "Return ONLY these two lines and nothing else:\n"
        "VERDICT: OK or PROBLEM\n"
        "EXPLANATION: one or two short sentences only.\n"
        "Do not provide chain-of-thought, step-by-step reasoning, internal deliberation, headings, or numbered lists.\n"
    )
    succ, response = llm_scenario.query(scenario_query)
    response = _normalize_cc_response(response)
    if hasattr(llm_scenario, "close"):
        llm_scenario.close()
        del llm_scenario

    verdict = extract_cc_verdict(response)
    if succ and verdict == "OK":
        logger.info(f"\rLLM has correctly understood the scenario\n{response}") 
        file.write(f"LLM has correctly understood the scenario\n{response}\n")
    elif succ and verdict == "PROBLEM":
        logger.error(f"\rLLM has not correctly understood the scenario or there is a problem in the scenario\n{response}")
        file.write(f"LLM has not correctly understood the scenario or there is a problem in the scenario\n{response}")
        return False, response
    else: 
        logger.error(f"Problem with the LLM succ: {succ} verdict: {verdict}\n{response}")
        # sys.exit(1)

    file.close()

    return True, ""
    

########################################################################################################################
 

def example_queries() -> Tuple[str, str]:
    query_hl = """
The scenario involves two distinct locations, referred to as Location1 and Location2. These two locations are directly connected, allowing for movement between them.
Containers:

    There are two containers in the system:
        Container c1 is initially located in Location1, placed on the ground.
        Container c2 is also in Location1, positioned on top of c1.

Robot:

    A robot, designated as Robot r1, is initially situated in Location1.
    The robot is capable of transporting a container from one location to another. However, to do so:
        The container must be placed on top of the robot.
        The robot can only move while carrying one container at a time.
        The robot cannot move if the container it is carrying is obstructed by another container.

Cranes:

    Each location is equipped with a crane:
        The crane in Location1 operates only within that location.
        The crane in Location2 operates exclusively within Location2.
    Cranes are versatile and capable of performing the following operations:
        Moving a container from the ground to the top of another container within the same location.
        Loading a container onto the robot or unloading a container from the robot. Container could be everywhere but has to be clear
        Placing a container on the ground in the same location, so cannot place a container in a different location (e.g crane 1 is located
        in location1 so can only operate in location1 not in other).
    A crane can only manipulate a container if the container is clear, meaning there is nothing on top of it.
    When executing an action the crane is busy so it could not execute any other action till the finish of the action.

Goal:

    By the end of the operation:
        Container c2 must be relocated to Location2.
        Container c1 must remain in its original position in Location1.

This setup requires a sequence of coordinated actions involving the robot and the cranes to achieve the desired arrangement of containers.
    """


    query_ll = """
Let the container, crane and robot and their positions be described in the high-level part.
There is available only one wheeled robot that can:
- move_robot(robot, locationFrom, locationTo), which makes the robot move from position (x1,y1) to position (x2,y2).
There is one crane for location that can:
- go_to_position(crane, position), which makes the crane move its gripper to the specified position.
- go_to_container(crane, container), which makes the crane move on the top of the container.
- close(crane), which closes the gripper.
- open(crane), which opens the gripper.
- lift(crane, container), which lifts the container.
- lower(crane, container), which lowers the container.
Remember to use the appropriate tags for the code you produce and not to use prolog tags.
Moreover, remember that the low-level actions you will generate must not contain high-level predicates. '.
    """

    return query_hl, query_ll


def main():
    logger.info("STARTING")
    args = parse_arguments()
    args = set_default_args(args)

    os.makedirs(args.output_path, exist_ok=True)

    hl_llm_gen = hl_llm_multi_step
    ll_llm_gen = ll_llm_multi_step

    if args.single_query:
        logger.debug("Using single_query=True")
        hl_llm_gen = hl_llm_single_step
        ll_llm_gen = ll_llm_single_step

    if args.use_example_queries:
        query_hl, query_ll = example_queries()
    else:
        query_hl, query_ll = _read_queries_from_files(
            query_hl_file=args.query_hl_file,
            query_ll_file=args.query_ll_file,
        )

    if not args.skip_comprehension:
        compr, resp = llm_scenario_comprehension(
            query_hl=query_hl,
            query_ll=query_ll,
            llm_config_file=args.conf,
            output_file_cc=args.output_cc,
            cc_examples_hl_path=args.cc_examples_hl,
            cc_examples_config_path=args.cc_examples_config,
        )
        if not compr:
            logger.error(f"There was a problem with the comprehension of the scenario {resp}")

    if args.only_comprehension:
        logger.info("ALL DONE!")
        return
        
    
    # if WAIT:
    #     input("Consistency check finished, press enter to continue...")

    # Use HL LLM to extract HL knowledge base
    # hl_kb, response = hl_llm(query_hl)
    hl_kb = hl_llm_gen(
        query=query_hl,
        verify=args.verify_hl,
        llm_config_file=args.conf,
        hl_examples_config_path=args.hl_examples_config,
        output_file_hl=args.output_hl,
        output_hl_kb_file=args.output_hl_kb,
        consistency_checks_path=args.consistency_checks,
    )
    write_to_file(hl_kb, output_file=args.output_hl_kb)

    if args.wait:
        input("HL finished, press enter to continue...")
    hl_kb = read_from_file(args.output_hl_kb)

    # use LL LLM to extract LL knowledge base
    kb = ll_llm_gen(
        query=query_ll,
        kb=hl_kb,
        verify=args.verify_ll,
        llm_config_file=args.conf,
        ll_examples_config_path=args.ll_examples_config,
        output_file_ll=args.output_ll,
        output_kb_file=args.output_kb,
        consistency_checks_path=args.consistency_checks,
    )
    write_to_file(kb, output_file=args.output_kb)

    if args.wait:
        input("LL finished, press enter to continue...")
    kb = read_from_file(args.output_kb)

    logger.info("ALL DONE!")


if __name__ == "__main__":
    main()
