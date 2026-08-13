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

def hl_llm_single_step(
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

    # Generate action set
    query = "\nGiven that the previous messages are examples, you now have to produce the code for the knowledge base for this task.\n" + query + \
        "\nRemember to wrap it into Markdown tags \"```KB```\", \"```init```\", \"```goal```\",  \"```actions```\",  and NOT with \"```prolog```\". Output ONLY the code block. No reasoning and no preamble." 
    succ, response = llm.query(query)
    assert succ == True, "Failed to generate final state"
    print(succ, response)
    print()
    file.write(f"ACTIONS: {response}\n")
    kb = {}
    scan_and_extract(kb, response)

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
            print(succ, repair_response)
            print()

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


def ll_llm_single_step(
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
    
    query = "\nYou now have to produce code for the task that follows.\n" + query + \
        "Given the following high-level knowledge-base:\n{}\n".format(kb['kb']) + \
        "\nUpdate the knowledge-base with the new low-level predicates, generate the low-level actions and mappings. {}".format(LLM_LL_WARNING) 
    succ, response = llm.query(query)
    assert succ == True, "Failed to generate LL KB"
    print(succ, response)
    file.write(f"LL: {response}\n")
    apply_low_level_extraction(
        kb,
        response,
        high_level_sections,
        allowed_tags={"kb", "init", "goal", "ll_actions", "mappings"},
    )
    assert kb.get("ll_actions", "").strip() != "", "LL generation did not return a non-empty `ll_actions` block"

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
