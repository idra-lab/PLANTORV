# Copyright © University of Trento and DLR 2025.
# This software is proprietary to the University of Trento and DLR. Use is permitted solely within
# the Horizon Europe project “INVERSE” (Grant Agreement ID: 101136067).
# This license does not override any rights or obligations established in the Grant Agreement.
# Redistribution or use outside the project is prohibited.

# This file is the main file which coordinates the different aspects of the planner. 

import os
import sys

try:
    from .llm_gen_support import (
        LLM_CONF_PATH,
        LL_EXAMPLES_CONFIG_PATH,
        HL_EXAMPLES_CONFIG_PATH,
        CONSISTENCY_CHECKS_PATH,
        OUTPUT_FILE_HL,
        OUTPUT_FILE_LL,
        OUTPUT_KB_FILE,
        OUTPUT_HL_KB_FILE,
        write_to_file,
        scan_and_extract,
        snapshot_high_level_sections,
        apply_low_level_extraction,
        _normalize_verify_retries,
        _run_hl_consistency_check,
        _run_ll_consistency_check,
        _build_hl_repair_query,
        _build_ll_repair_query,
        _write_hl_retry_snapshot,
        _write_ll_retry_snapshot,
        build_llm,
    )
except Exception:
    try:
        from llm_gen_support import (
            LLM_CONF_PATH,
            LL_EXAMPLES_CONFIG_PATH,
            HL_EXAMPLES_CONFIG_PATH,
            CONSISTENCY_CHECKS_PATH,
            OUTPUT_FILE_HL,
            OUTPUT_FILE_LL,
            OUTPUT_KB_FILE,
            OUTPUT_HL_KB_FILE,
            write_to_file,
            scan_and_extract,
            snapshot_high_level_sections,
            apply_low_level_extraction,
            _normalize_verify_retries,
            _run_hl_consistency_check,
            _run_ll_consistency_check,
            _build_hl_repair_query,
            _build_ll_repair_query,
            _write_hl_retry_snapshot,
            _write_ll_retry_snapshot,
            build_llm,
        )
    except Exception:
        sys.path.append(os.path.dirname(__file__))
        from llm_gen_support import (
            LLM_CONF_PATH,
            LL_EXAMPLES_CONFIG_PATH,
            HL_EXAMPLES_CONFIG_PATH,
            CONSISTENCY_CHECKS_PATH,
            OUTPUT_FILE_HL,
            OUTPUT_FILE_LL,
            OUTPUT_KB_FILE,
            OUTPUT_HL_KB_FILE,
            write_to_file,
            scan_and_extract,
            snapshot_high_level_sections,
            apply_low_level_extraction,
            _normalize_verify_retries,
            _run_hl_consistency_check,
            _run_ll_consistency_check,
            _build_hl_repair_query,
            _build_ll_repair_query,
            _write_hl_retry_snapshot,
            _write_ll_retry_snapshot,
            build_llm,
        )

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


## FUNCTIONS ###########################################################################################################

def hl_llm_multi_step(
    query,
    verify=0,
    llm_config_file=LLM_CONF_PATH,
    hl_examples_config_path=HL_EXAMPLES_CONFIG_PATH,
    output_file_hl=OUTPUT_FILE_HL,
    output_hl_kb_file=OUTPUT_HL_KB_FILE,
    consistency_checks_path=CONSISTENCY_CHECKS_PATH,
    mandatory_verify = False,
) -> dict:
    """
    :brief This function uses the LLM to extract the high-level knowledge base, the initial and final states.
    :param query: The query that will be used to extract the knowledge base, initial and final states. 
    :param verify: Number of repair attempts after a failed high-level consistency check (0 disables verification).
    :return: A tuple containing the knowledge base and the response from the LLM
    """
    parent_dir = os.path.dirname(output_file_hl)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    file = open(output_file_hl, "w+")
    max_retries = _normalize_verify_retries(verify)

    # Extract HL knowledge base
    logger.info("[HL] Extracting HL knowledge base")
    llm = build_llm(
        llm_config_file=llm_config_file,
        examples_yaml_file=[hl_examples_config_path],
    )

    kb = {}

    # Generate knowledge base
    logger.info("[HL] Generating knowledge base")
    kb_query = "\nGiven that the previous messages are examples, you now have to produce code for the task that follows.\n" +\
        query +\
        "\nWrite the static knowledge base. Remember to specify all the correct predicates and identify which are the predicates that are resources and to wrap it into Markdown tags \"```kb```\" and NOT with \"```Prolog```\". Output ONLY the code block between the tags and do not forget the closing backticks. No reasoning, no preamble."
    succ, tmp_response = llm.query(kb_query)
    assert succ == True, "Failed to generate static knowledge base"
    logger.debug(f"KB query result: {succ}, {tmp_response}")
    file.write(f"KB: {tmp_response}\n")
    scan_and_extract(kb, tmp_response)
    logger.debug(f"KB after initial generation: {kb}")

    # Generate initial and final states
    logger.info("[HL] Generating initial and final states")
    states_query = "\nGiven that the previous messages are examples, you now have to produce code for the task that follows.\n" + query + \
        "\nGiven the following static knowledge base\n```kb\n{}\n```".format(kb["kb"]) +\
        "\nWrite the initial and final states, minding to include all the correct predicates. Remember to wrap it into Markdown tags \"```init```\" and \"```goal```\" and NOT with \"```prolog```\". Output ONLY the code block between the tags and do not forget the closing backticks. No reasoning, no preamble."
    succ, tmp_response = llm.query(states_query)
    assert succ == True, "Failed to generate initial and final states"
    # logger.debug(f"States query result: {succ}, {tmp_response}")
    logger.debug(f"States query result: {succ}")
    logger.debug("==========================")
    logger.debug(tmp_response)
    logger.debug("==========================")
    file.write(f"INIT: {tmp_response}\n")
    scan_and_extract(kb, tmp_response)
    logger.debug(f"KB after initial generation: {kb}")

    # Generate action set
    logger.info("[HL] Generating actions set")
    final_query = "\nGiven that the previous messages are examples, you now have to produce code for the actions of the following task.\n" + query + \
        "\nGiven the following static knowledge base\n```kb\n{}\n```".format(kb["kb"]) +\
        "\nGiven the following initial state\n```init\n{}\n```".format(kb["init"]) +\
        "\nGiven the following goal state\n```goal\n{}\n```".format(kb["goal"]) +\
        "\nRemember to wrap it into Markdown tags \"```actions```\" and NOT with \"```prolog```\". Output ONLY the code block between the tags and do not forget the closing backticks. No reasoning, no preamble." 
    succ, response = llm.query(final_query)
    assert succ == True, "Failed to generate final state"
    logger.debug(f"Final query result: {succ}, {response}")
    file.write(f"ACTIONS: {response}\n")
    scan_and_extract(kb, response)
    logger.debug(f"KB after initial generation: {kb}")


    if max_retries > 0:
        logger.info("[HL] Verifying high-level knowledge base consistency")
        write_to_file(kb, output_file=output_hl_kb_file)
        ok, check_feedback = _run_hl_consistency_check(
            output_hl_kb_file,
            consistency_checks_path=consistency_checks_path,
        )

        retries = 0
        while not ok and retries < max_retries:
            retries += 1
            logger.error(
                f"\r[HL] Consistency check failed (attempt {retries}/{max_retries}) for reason: \n{check_feedback}"
            )
            file.write(f"VERIFY_{retries}_ERROR: {check_feedback}\n")

            repair_query = _build_hl_repair_query(query, kb, check_feedback)
            succ, repair_response = llm.query(repair_query)
            assert succ == True, "Failed to repair high-level KB"
            logger.debug(f"Repair query result: {succ}, {repair_response}")

            file.write(f"VERIFY_{retries}_FIX: {repair_response}\n")
            scan_and_extract(kb, repair_response)
            _write_hl_retry_snapshot(
                kb,
                f"verify_attempt_{retries}",
                output_hl_kb_file=output_hl_kb_file,
            )

            write_to_file(kb, output_file=output_hl_kb_file)
            ok, check_feedback = _run_hl_consistency_check(
                output_hl_kb_file,
                consistency_checks_path=consistency_checks_path,
            )

        if not ok:
            if mandatory_verify:
                raise RuntimeError(f"High-level KB consistency check failed after retries.\n{check_feedback}")
            else:
                logger.warning(f"High-level KB consistency check failed after retries.\n{check_feedback}")

    write_to_file(kb, output_file=output_hl_kb_file)

    if hasattr(llm, "close"):
        llm.close()
        del llm

    file.close()

    return kb


########################################################################################################################


def ll_llm_multi_step(
    query,
    kb,
    verify=0,
    llm_config_file=LLM_CONF_PATH,
    ll_examples_config_path=LL_EXAMPLES_CONFIG_PATH,
    output_file_ll=OUTPUT_FILE_LL,
    output_kb_file=OUTPUT_KB_FILE,
    consistency_checks_path=CONSISTENCY_CHECKS_PATH,
    mandatory_verify = False,
) -> dict:
    """
    :brief This function uses the LLM to extract the low-level knowledge base, states, actions, and mappings.
    :param query: The low-level query used to guide extraction.
    :param kb: The current knowledge-base dictionary (including the high-level content to refine).
    :param verify: Number of repair attempts after a failed low-level consistency check (0 disables verification).
    :return: The updated knowledge-base dictionary.
    """
    max_retries = _normalize_verify_retries(verify)
    hl_kb = """
    ```kb
    {}
    ```
    ```init
    {}
    ```
    ```goal
    {}
    ```
    ```actions
    {}
    ```""".format(kb["kb"], kb["init"], kb["goal"], kb["actions"])
    high_level_sections = snapshot_high_level_sections(kb)
    
    parent_dir = os.path.dirname(output_file_ll)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    file = open(output_file_ll, "w+")

    LLM_LL_WARNING = (
        "Remember to prepend the low-level predicates with `ll_` and also to not use the high-level "
        "predicates inside the low-level actions as they may lead to errors. The high-level "
        "`kb`, `init`, `goal`, and `actions` sections are immutable: do not remove, rename, "
        "change the arity of, simplify, or rewrite any non-`ll_` predicate or any `hl_d_action`. "
        "Only add low-level information using `ll_` predicates, and output low-level operators "
        "under `ll_actions`, never under `actions`."
    )

    # Extract LL knowledge base
    logger.info("[LL] Extract LL knowledge base")
    llm = build_llm(
        llm_config_file=llm_config_file,
        examples_yaml_file=[ll_examples_config_path],
    )
    
    # Generate static knowledge-base
    logger.info("[LL] Generating knowledge base")
    kb_query = "\nYou now have to produce code for the task that follows.\n" + query + \
        "Given the following high-level knowledge-base:\n{}\n".format(kb['kb']) + \
        "\nUpdate only the general knowledge base by adding the new low-level predicates and resources. {}".format(LLM_LL_WARNING)
    succ, response = llm.query(kb_query)
    assert succ == True, "Failed to generate LL KB"
    print(succ, response)
    file.write(f"KB: {response}\n")
    apply_low_level_extraction(kb, response, high_level_sections, allowed_tags={"kb"})

    # Generate initial and final states
    logger.info("[LL] Generating initial and final states")
    states_query = "\nYou now have to produce code for the task that follows.\n" + query + \
        "Given the low-level knowledge-base:\n```kb\n{}\n```\n".format(kb["kb"]) + \
        "Given the high-level initial and final states:\n```init\n{}\n```\n```goal\n{}\n```\n".format(kb["init"], kb["goal"]) + \
        "\nUpdate the initial and final states by adding low-level state predicates while preserving all high-level state predicates. Mind to include all the necessary predicates. {}".format(LLM_LL_WARNING)
        # "Given the low-level actions set:\n```actions\n{}\n```\n".format(kb["ll_actions"]) + \
    succ, response = llm.query(states_query)
    assert succ == True, "Failed to generate LL KB"
    print(succ, response)
    file.write(f"INIT: {response}\n")
    apply_low_level_extraction(kb, response, high_level_sections, allowed_tags={"init", "goal"})

    # Generate actions set
    logger.info("[LL] Generating actions set")
    ll_actions_query = "\nGiven that the previous messages are examples, you now have to produce code for the task that follows.\n" + query + \
        "Given the following high-level knowledge-base:\n{}\n".format(hl_kb) + \
        "Given the refactored low-level knowledge-base:\n```kb\n{}\n```\n".format(kb["kb"]) + \
        "\nWrite only the low-level actions set using the `ll_actions` tag. {}".format(LLM_LL_WARNING)
    succ, response = llm.query(ll_actions_query)
    assert succ == True, "Failed to generate LL KB"
    print(succ, response)
    file.write(f"ACTIONS: {response}\n")
    apply_low_level_extraction(kb, response, high_level_sections, allowed_tags={"ll_actions"})
    assert kb.get("ll_actions", "").strip() != "", "LL action generation did not return a non-empty `ll_actions` block"

    # Generate mappings
    logger.info("[LL] Generating mappings")
    mappings_query = "\nGiven that the previous messages are examples, you now have to produce code for the task that follows.\n" + query + \
        "Given the following high-level knowledge-base:\n{}\n".format(hl_kb) + \
        "Given the refactored low-level knowledge-base:\n```kb\n{}\n```\n".format(kb["kb"]) + \
        "Given the initial state:\n```init\n{}\n```\n".format(kb["init"]) + \
        "Given the final state:\n```goal\n{}\n```\n".format(kb["goal"]) + \
        "Given the low-level actions set:\n```ll_actions\n{}\n```\n".format(kb["ll_actions"]) + \
        "\nProvide the mappings from high-level actions to low-level actions. Remember that the mappings are only for the start actions. {}".format(LLM_LL_WARNING)
    succ, response = llm.query(mappings_query)
    assert succ == True, "Failed to generate LL KB"
    print(succ, response)
    file.write(f"MAPPINGS: {response}\n")
    apply_low_level_extraction(kb, response, high_level_sections, allowed_tags={"mappings"})

    if max_retries > 0:
        logger.info("[LL] Verifying low-level knowledge base consistency")
        write_to_file(kb, output_file=output_kb_file)
        ok, check_feedback = _run_ll_consistency_check(
            output_kb_file,
            consistency_checks_path=consistency_checks_path,
        )

        retries = 0
        while not ok and retries < max_retries:
            retries += 1
            logger.error(
                f"\r[LL] Consistency check failed (attempt {retries}/{max_retries}) for reason: \n{check_feedback}"
            )
            file.write(f"VERIFY_{retries}_ERROR: {check_feedback}\n")

            repair_query = _build_ll_repair_query(query, kb, check_feedback)
            succ, repair_response = llm.query(repair_query)
            assert succ == True, "Failed to repair low-level KB"
            print(succ, repair_response)
            print()

            file.write(f"VERIFY_{retries}_FIX: {repair_response}\n")
            apply_low_level_extraction(
                kb,
                repair_response,
                high_level_sections,
                allowed_tags={"kb", "init", "goal", "ll_actions", "mappings"},
            )
            _write_ll_retry_snapshot(
                kb,
                f"verify_attempt_{retries}",
                output_kb_file=output_kb_file,
            )

            write_to_file(kb, output_file=output_kb_file)
            ok, check_feedback = _run_ll_consistency_check(
                output_kb_file,
                consistency_checks_path=consistency_checks_path,
            )

        if not ok:
            if mandatory_verify:
                raise RuntimeError(f"Low-level KB consistency check failed after retries.\n{check_feedback}")
            else:
                logger.warning(f"Low-level KB consistency check failed after retries.\n{check_feedback}")

    file.close()
    write_to_file(kb, output_file=output_kb_file)

    if hasattr(llm, "close"):
        llm.close()
        del llm

    return kb 
    
