# Copyright © University of Trento and DLR 2025.
# This software is proprietary to the University of Trento and DLR. Use is permitted solely within
# the Horizon Europe project “INVERSE” (Grant Agreement ID: 101136067).
# This license does not override any rights or obligations established in the Grant Agreement.
# Redistribution or use outside the project is prohibited.

# This file contains global variables, auxiliary functions and script arguments for the LLM generation process.

import os
import sys
import re
import subprocess
from typing import Dict, Iterable, Optional, Tuple
from pathlib import Path
import argparse

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

try:
    from llm_base import create_llm_from_config
except Exception:
    try:
        from .llm_base import create_llm_from_config
    except Exception:
        sys.path.append(os.path.dirname(__file__))
        from llm_base import create_llm_from_config


## GLOBAL VARIABLES ####################################################################################################

# Anthropic
# LLM_CONF_PATH = os.path.join(os.path.dirname(__file__), 'conf', 'claude-46-opus.yaml')
# # Gemini
# LLM_CONF_PATH = os.path.join(os.path.dirname(__file__), 'conf', 'gemini_2-0_flash.yaml')
# LLM_CONF_PATH = os.path.join(os.path.dirname(__file__), 'conf', 'gemini_3-0_flash.yaml')
# # GLM
# LLM_CONF_PATH = os.path.join(os.path.dirname(__file__), 'conf', 'glm4_7.yaml')
# # OpenAI
# LLM_CONF_PATH = os.path.join(os.path.dirname(__file__), 'conf', 'gpt52.yaml')
# # Hugging Face
# LLM_CONF_PATH = os.path.join(os.path.dirname(__file__), 'conf', 'hf_google_gemma-2-9b-it.yaml')
# LLM_CONF_PATH = os.path.join(os.path.dirname(__file__), 'conf', 'hf_meta-llama_Meta-Llama-3-8B-Instruct.yaml')
# LLM_CONF_PATH = os.path.join(os.path.dirname(__file__), 'conf', 'hf_microsoft_Phi-3-mini-128k-instruct.yaml') 
# LLM_CONF_PATH = os.path.join(os.path.dirname(__file__), 'conf', 'hf_mistralai_Mistral-7B-Instruct-v0.3.yaml')
# LLM_CONF_PATH = os.path.join(os.path.dirname(__file__), 'conf', 'hf_Qwen_Qwen3-Coder-Next.yaml')
# # vLLM (offline, in-process)
# LLM_CONF_PATH = os.path.join(os.path.dirname(__file__), 'conf', 'vllm_Qwen_Qwen3.5-9B.yaml')
# # Azure OpenAI
# LLM_CONF_PATH = os.path.join(os.path.dirname(__file__), 'conf', 'azure_gpt35-turbo.yaml')
# LLM_CONF_PATH = os.path.join(os.path.dirname(__file__), 'conf', 'azure_gpt40-128k.yaml')
# LLM_CONF_PATH = os.path.join(os.path.dirname(__file__), 'conf', 'azure_gpt40-32k.yaml')
# LLM_CONF_PATH = os.path.join(os.path.dirname(__file__), 'conf', 'azure_gpt40-8k.yaml')
# LLM_CONF_PATH = os.path.join(os.path.dirname(__file__), 'conf', 'azure_gpt4o.yaml')
# LLM_CONF_PATH = os.path.join(os.path.dirname(__file__), 'conf', 'azure_gpt50-mini.yaml')
LLM_CONF_PATH = os.path.join(os.path.dirname(__file__), 'conf', 'azure_gpt52.yaml')

EXAMPLES_PATH           = os.path.join(os.path.dirname(__file__), 'examples')
CC_EXAMPLES_CONFIG_PATH = os.path.join(EXAMPLES_PATH, 'cc', 'few-shots-cc.yaml')
CC_EXAMPLES_HL_PATH     = os.path.join(EXAMPLES_PATH, 'cc', 'hl.yaml')
CC_EXAMPLES_LL_PATH     = os.path.join(EXAMPLES_PATH, 'cc', 'll.yaml')
LL_EXAMPLES_CONFIG_PATH = os.path.join(EXAMPLES_PATH, 'multi', 'few-shots-ll.yaml')
HL_EXAMPLES_CONFIG_PATH = os.path.join(EXAMPLES_PATH, 'multi', 'few-shots-hl.yaml')

WAIT = False

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'output')

OUTPUT_KB_FILE    = os.path.join(OUTPUT_PATH, 'kb_ll.pl')
OUTPUT_HL_KB_FILE = os.path.join(OUTPUT_PATH, 'kb_hl.pl')

OUTPUT_FILE_CC = os.path.join(OUTPUT_PATH, "output_cc.txt")
OUTPUT_FILE_HL = os.path.join(OUTPUT_PATH, "output_hl.txt")
OUTPUT_FILE_LL = os.path.join(OUTPUT_PATH, "output_ll.txt")

QUERY_HL_FILE = os.path.join(OUTPUT_PATH, "query_hl.txt")
QUERY_LL_FILE = os.path.join(OUTPUT_PATH, "query_ll.txt")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PLANNER_SRC_PATH = os.path.join(REPO_ROOT, 'TP', 'prolog_planner', 'src')
CONSISTENCY_CHECKS_PATH = os.path.join(PLANNER_SRC_PATH, 'consistency_checks.pl')

MISSING = object() # Sentinel value for missing arguments


## AUXILIARY FUNCTIONS #################################################################################################


def write_to_file(kb, output_file = OUTPUT_KB_FILE):
    logger.info(f"Writing knowledge base to {output_file}")
    os.makedirs(Path(output_file).parent, exist_ok=True)
    with open(output_file, "w") as file:
        file.write("% This file was automatically generated by the LLM system\n")
        first_line = True
        for key, value in kb.items():
            if not first_line:
                file.write("\n")
            file.write(f"%%%%%%%%%%%%%%%%%%%%%%%\n% {key}\n%%%%%%%%%%%%%%%%%%%%%%%\n{value}\n")
            first_line = False

###############################################################################

def read_from_file(kb_file = OUTPUT_KB_FILE):
    """
    :brief This function reads a knowledge base from a file and returns it as a dictionary. The file is expected to be in the format:
    ```
    %%%%%%%%%%%%%%%%
    % key1
    %%%%%%%%%%%%%%%%
    value1
    %%%%%%%%%%%%%%%%
    % key2
    %%%%%%%%%%%%%%%%
    value2
    ...
    ```

    :param kb_file: The path to the knowledge base file.
    :return: A dictionary containing the knowledge base, where the keys are the strings following the %%%%%%%%%%%%%%%% lines and the values are the corresponding values in the file.    
    """
    kb = dict()
    key = None
    with open(kb_file, "r") as file:
        new_key = False
        new_value = False
        value = ""
        for line in file:
            if new_key and line == "%%%%%%%%%%%%%%%%%%%%%%%\n":
                new_key = False

            elif new_key:
                key = line.split('%')[1].strip()
                new_value = True
                kb[key] = ""

            elif not new_key and line == "%%%%%%%%%%%%%%%%%%%%%%%\n":
                new_value = False
                if key is not None and value != "":
                    kb[key] = value
                value = ""
                new_key = True

            elif not new_key and new_value:
                value += line

        if key is not None:
            kb[key] = value

    return kb

###############################################################################

_KB_SECTION_TAGS = {"kb", "init", "goal", "actions", "ll_actions", "mappings"}


def _normalize_section_tag(tag: str) -> str:
    return tag.lower().replace(" ", "")


def extract_tagged_sections(response, allowed_tags: Optional[Iterable[str]] = None) -> Dict[str, str]:
    normalized_allowed_tags = None
    if allowed_tags is not None:
        normalized_allowed_tags = {_normalize_section_tag(tag) for tag in allowed_tags}

    sections = {}
    pattern = re.compile(r'\`\`\`\s*(\w+)\s*([^\`]*?)\`\`\`', re.DOTALL)
    matches = pattern.findall(response)
    for key, value in matches:
        key = _normalize_section_tag(key)
        if key not in _KB_SECTION_TAGS:
            logger.warning(f"Unexpected tag {key} found in LLM response. This tag will be ignored, but it might indicate that the LLM is not following the expected format.")
            continue
        if normalized_allowed_tags is not None and key not in normalized_allowed_tags:
            logger.warning(f"Ignoring tag {key} because this generation step only accepts {sorted(normalized_allowed_tags)}.")
            continue

        value = value.strip()
        if value == "":
            continue

        sections[key] = value

    return sections

###############################################################################

def scan_and_extract(kb, response, allowed_tags: Optional[Iterable[str]] = None):
    """
    :brief: This function scans the code produced by the LLM and extracts the different parts that are in the form of
            ```<tag> 
            <content>
            ```
            If content is not empty, it is added to the knowledge base, even if the key was already present.
    :param kb: Dictionary where the extracted information will be stored
    :param response: The response from the LLM
    :param allowed_tags: Optional set of tags accepted for this extraction step.
    :return: None
    """
    for key, value in extract_tagged_sections(response, allowed_tags=allowed_tags).items():
        kb[key] = value

###############################################################################

def _normalized_prolog_line(line: str) -> str:
    return re.sub(r'\s+', ' ', line.strip()).rstrip(',')

###############################################################################

def merge_low_level_kb_section(high_level_kb: str, generated_kb: str) -> str:
    base = high_level_kb.strip()
    existing_lines = {
        _normalized_prolog_line(line)
        for line in base.splitlines()
        if _normalized_prolog_line(line)
    }
    additions = []
    added_lines = set()

    for line in generated_kb.strip().splitlines():
        stripped = line.strip()
        if stripped == "" or stripped.startswith("%"):
            continue
        if "ll_" not in stripped:
            continue

        normalized = _normalized_prolog_line(line)
        if normalized in existing_lines or normalized in added_lines:
            continue

        additions.append(line.rstrip())
        added_lines.add(normalized)

    if len(additions) == 0:
        return base

    return base + "\n\n" + "\n".join(additions)

###############################################################################

def _extract_state_functor(state_section: str) -> Optional[str]:
    match = re.search(r'\b(init_state|goal_state)\s*\(', state_section)
    if match is None:
        return None
    return match.group(1)

###############################################################################

def _split_top_level_terms(body: str):
    terms = []
    current = []
    depth = 0

    for char in body:
        if char in "([{":
            depth += 1
        elif char in ")]}" and depth > 0:
            depth -= 1

        if char == "," and depth == 0:
            term = "".join(current).strip()
            if term:
                terms.append(term)
            current = []
            continue

        current.append(char)

    term = "".join(current).strip()
    if term:
        terms.append(term)

    return terms

###############################################################################

def _extract_state_terms(state_section: str):
    match = re.search(r'\b(?:init_state|goal_state)\s*\(\s*\[(.*)\]\s*\)\s*\.', state_section, re.DOTALL)
    if match is None:
        return []

    body = re.sub(r'%.*', '', match.group(1))
    return _split_top_level_terms(body)

###############################################################################

def merge_low_level_state_section(high_level_state: str, generated_state: str) -> str:
    functor = _extract_state_functor(high_level_state) or _extract_state_functor(generated_state)
    if functor is None:
        return high_level_state.strip()

    base_terms = _extract_state_terms(high_level_state)
    generated_terms = _extract_state_terms(generated_state)
    if len(base_terms) == 0:
        return high_level_state.strip()

    merged_terms = list(base_terms)
    seen = {_normalized_prolog_line(term) for term in merged_terms}

    for term in generated_terms:
        if "ll_" not in term:
            continue
        normalized = _normalized_prolog_line(term)
        if normalized in seen:
            continue
        merged_terms.append(term)
        seen.add(normalized)

    return "{}([\n{}\n]).".format(
        functor,
        ",\n".join(f"  {term}" for term in merged_terms),
    )

###############################################################################

def snapshot_high_level_sections(kb: dict) -> Dict[str, str]:
    return {
        "kb": kb.get("kb", ""),
        "init": kb.get("init", ""),
        "goal": kb.get("goal", ""),
        "actions": kb.get("actions", ""),
    }

###############################################################################

def apply_low_level_extraction(
    kb: dict,
    response: str,
    high_level_sections: Dict[str, str],
    allowed_tags: Optional[Iterable[str]] = None,
):
    sections = extract_tagged_sections(response, allowed_tags=allowed_tags)

    if "kb" in sections:
        kb["kb"] = merge_low_level_kb_section(high_level_sections["kb"], sections["kb"])
    if "init" in sections:
        kb["init"] = merge_low_level_state_section(high_level_sections["init"], sections["init"])
    if "goal" in sections:
        kb["goal"] = merge_low_level_state_section(high_level_sections["goal"], sections["goal"])
    if "ll_actions" in sections:
        kb["ll_actions"] = sections["ll_actions"]
    if "mappings" in sections:
        kb["mappings"] = sections["mappings"]

    kb["actions"] = high_level_sections["actions"]

## FUNCTIONS ###########################################################################################################

def build_llm(llm_config_file, examples_yaml_file):
    """
    :brief This function builds the LLM using the configuration file and the examples file. 

    :param llm_config_file: The path to the LLM configuration file.
    :param examples_yaml_file: The path to the examples file in YAML format.
    """
    return create_llm_from_config(
        llm_config_file=llm_config_file,
        examples_yaml_file=examples_yaml_file,
    )


def extract_cc_verdict(response: str) -> str:
    """Extract a strict OK/PROBLEM verdict from model output."""
    if not isinstance(response, str):
        return ""

    match = re.search(r"\b(OK|PROBLEM)\b", response, flags=re.IGNORECASE)
    if match is None:
        return ""
    return match.group(1).upper()

###############################################################################

def _prolog_quote(path: str) -> str:
    return path.replace("\\", "/").replace("'", "\\'")

###############################################################################

def _strip_ansi(text: str) -> str:
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

###############################################################################

def _clean_consistency_feedback(raw_output: str) -> str:
    lines = []
    for line in raw_output.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        if stripped.startswith("Warning:"):
            continue
        if "Singleton variables:" in stripped:
            continue
        if "Clauses of " in stripped:
            continue
        if "Use :- discontiguous" in stripped:
            continue
        lines.append(stripped)

    if len(lines) == 0:
        return raw_output.strip()
    return "\n".join(lines)

###############################################################################

def _normalize_verify_retries(verify) -> int:
    try:
        retries = int(verify)
    except (TypeError, ValueError):
        raise ValueError(f"verify must be an integer >= 0, got {verify!r}")

    if retries < 0:
        raise ValueError(f"verify must be an integer >= 0, got {verify!r}")

    return retries

###############################################################################

def _run_hl_consistency_check(
    kb_file: str,
    consistency_checks_path: str = CONSISTENCY_CHECKS_PATH,
) -> Tuple[bool, str]:
    if not os.path.exists(consistency_checks_path):
        return False, f"Consistency checker not found at {consistency_checks_path}"
    if not os.path.exists(kb_file):
        return False, f"Generated knowledge base not found at {kb_file}"

    goal_parts = [
        "style_check(-singleton)",
        "style_check(-discontiguous)",
        "set_prolog_flag(verbose, silent)",
        f"['{_prolog_quote(consistency_checks_path)}']",
        f"check_hl_kb_consistency('{_prolog_quote(kb_file)}')",
    ]

    command = ["swipl", "-q", "-g", ", ".join(goal_parts), "-t", "halt"]
    try:
        logger.debug(f'Running command: {" ".join(command)}')
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return False, "Could not run `swipl`: executable not found in PATH."
    except subprocess.TimeoutExpired:
        return False, "Consistency check timed out after 30 seconds."
    output = _strip_ansi((result.stdout or "") + "\n" + (result.stderr or ""))
    cleaned_output = _clean_consistency_feedback(output)

    return result.returncode == 0, cleaned_output

###############################################################################

def _run_ll_consistency_check(
    kb_file: str,
    consistency_checks_path: str = CONSISTENCY_CHECKS_PATH,
) -> Tuple[bool, str]:
    if not os.path.exists(consistency_checks_path):
        return False, f"Consistency checker not found at {consistency_checks_path}"
    if not os.path.exists(kb_file):
        return False, f"Generated knowledge base not found at {kb_file}"

    goal_parts = [
        "style_check(-singleton)",
        "style_check(-discontiguous)",
        "set_prolog_flag(verbose, silent)",
        f"['{_prolog_quote(consistency_checks_path)}']",
        f"check_ll_kb_consistency('{_prolog_quote(kb_file)}')",
    ]

    command = ["swipl", "-q", "-g", ", ".join(goal_parts), "-t", "halt"]
    try:
        logger.debug(f'Running command: {" ".join(command)}')
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return False, "Could not run `swipl`: executable not found in PATH."
    except subprocess.TimeoutExpired:
        return False, "Consistency check timed out after 30 seconds."
    output = _strip_ansi((result.stdout or "") + "\n" + (result.stderr or ""))
    cleaned_output = _clean_consistency_feedback(output)

    return result.returncode == 0, cleaned_output

###############################################################################

def _build_hl_repair_query(query: str, kb: dict, feedback: str) -> str:
    feedback = feedback.strip()
    if len(feedback) > 4000:
        feedback = feedback[:4000] + "\n...[truncated]..."

    return (
        "\nGiven that the previous messages are examples, you now have to fix the generated code for the following task.\n"
        + query
        + "\nThe generated high-level code failed the consistency checks with this feedback:\n"
        + "```text\n{}\n```".format(feedback)
        + "\nCurrent high-level code is:\n"
        + "```kb\n{}\n```\n".format(kb["kb"])
        + "```init\n{}\n```\n".format(kb["init"])
        + "```goal\n{}\n```\n".format(kb["goal"])
        + "```actions\n{}\n```\n".format(kb["actions"])
        + "\nProvide a corrected version. You must output the whole knowledge base, including `kb`, `init`, `goal`, and `actions`."
        + "\nDo not use `prolog` tags."
    )

###############################################################################

def _build_ll_repair_query(query: str, kb: dict, feedback: str) -> str:
    feedback = feedback.strip()
    # if len(feedback) > 4000:
    #     feedback = feedback[:4000] + "\n...[truncated]..."

    return (
        "\nGiven that the previous messages are examples, you now have to fix the generated code for the following task.\n"
        + query
        + "\nThe generated low-level code failed the consistency checks with this feedback:\n"
        + "```text\n{}\n```".format(feedback)
        + "\nCurrent low-level code is:\n"
        + "```kb\n{}\n```\n".format(kb["kb"])
        + "```init\n{}\n```\n".format(kb["init"])
        + "```goal\n{}\n```\n".format(kb["goal"])
        + "```ll_actions\n{}\n```\n".format(kb.get("ll_actions", ""))
        + "```mappings\n{}\n```\n".format(kb.get("mappings", ""))
        + "\nProvide a corrected version. You must output the whole knowledge base, including `kb`, `init`, `goal`, `ll_actions`, and `mappings`."
        + "\nThe high-level layer is immutable: preserve all non-`ll_` predicates in `kb`, `init`, and `goal`, and do not output or rewrite the high-level `actions` section."
        + "\nDo not use `prolog` tags. Do not output the reasoning process."
    )

###############################################################################

def _build_step_output_file(base_file: str, step: str) -> str:
    base_path = Path(base_file)
    safe_step = re.sub(r'[^a-zA-Z0-9_.-]+', '_', str(step)).strip('_')
    if safe_step == "":
        safe_step = "step"
    return str(base_path.with_name(f"{base_path.stem}_{safe_step}{base_path.suffix}"))

###############################################################################

def _write_hl_retry_snapshot(
    kb: dict,
    step: str,
    output_hl_kb_file: str = OUTPUT_HL_KB_FILE,
) -> str:
    output_file = _build_step_output_file(output_hl_kb_file, step)
    write_to_file(kb, output_file=output_file)
    return output_file

###############################################################################

def _write_ll_retry_snapshot(
    kb: dict,
    step: str,
    output_kb_file: str = OUTPUT_KB_FILE,
) -> str:
    output_file = _build_step_output_file(output_kb_file, step)
    write_to_file(kb, output_file=output_file)
    return output_file

###############################################################################

def _read_queries_from_files(query_hl_file: str, query_ll_file: str) -> Tuple[str, str]:
    assert os.path.exists(query_hl_file), f"High-level query file not found at {query_hl_file}"
    assert os.path.exists(query_ll_file), f"Low-level query file not found at {query_ll_file}"

    with open(query_hl_file, "r") as file_hl:
        query_hl = file_hl.read()

    with open(query_ll_file, "r") as file_ll:
        query_ll = file_ll.read()

    return query_hl, query_ll


## ARGUMENTS ###########################################################################################################

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Generation Support Script")
    parser.add_argument("--conf",               type=str, default=MISSING, help="Path to the LLM configuration file.")
    parser.add_argument("--consistency-checks", type=str, default=MISSING, help="Path to Prolog consistency checks file.")
    parser.add_argument("--cc-examples-config", type=str, default=MISSING, help="Path to scenario-comprehension examples (HL+LL).")
    parser.add_argument("--cc-examples-hl",     type=str, default=MISSING, help="Path to scenario-comprehension high-level examples.")
    parser.add_argument("--cc-examples-ll",     type=str, default=MISSING, help="Path to scenario-comprehension low-level examples.")
    parser.add_argument("--hl-examples-config", type=str, default=MISSING, help="Path to high-level generation examples.")
    parser.add_argument("--ll-examples-config", type=str, default=MISSING, help="Path to low-level generation examples.")

    parser.add_argument("--output-path",        type=str, default=MISSING, help="Directory for all output files. Cannot be used together with individual output file arguments.")
    parser.add_argument("--output-cc",          type=str, default=MISSING, help="Path for scenario-comprehension output log. Cannot be used together with --output-path.")
    parser.add_argument("--output-hl",          type=str, default=MISSING, help="Path for high-level generation output log. Cannot be used together with --output-path.")
    parser.add_argument("--output-ll",          type=str, default=MISSING, help="Path for low-level generation output log. Cannot be used together with --output-path.")
    parser.add_argument("--output-hl-kb",       type=str, default=MISSING, help="Path for generated high-level knowledge base. Cannot be used together with --output-path.")
    parser.add_argument("--output-kb",          type=str, default=MISSING, help="Path for generated low-level knowledge base. Cannot be used together with --output-path.")

    parser.add_argument("--query-hl-file",      type=str, default=MISSING, help="Path to high-level scenario description text file.")
    parser.add_argument("--query-ll-file",      type=str, default=MISSING, help="Path to low-level scenario description text file.")

    parser.add_argument("--verify-hl",          type=int, default=MISSING, help="Repair attempts for high-level consistency verification (0 disables).")
    parser.add_argument("--verify-ll",          type=int, default=MISSING, help="Repair attempts for low-level consistency verification (0 disables).")
    parser.add_argument("--mandatory-verify",   action="store_true", help="If set, the script will exit with an error if consistency checks fail after all repair attempts.")

    parser.add_argument("--log-file",           type=str, default=MISSING, help="Path to log file. If not provided, logs will only be printed to console.")
    parser.add_argument("--log-level",          type=str, default=MISSING, help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")

    comprehension_group = parser.add_mutually_exclusive_group()
    comprehension_group.add_argument("--skip-comprehension",   action="store_true", help="Skip scenario comprehension checks.")
    comprehension_group.add_argument("--only-comprehension",   action="store_true", help="Run scenario comprehension checks and exit before HL/LL generation.")
    parser.add_argument("--use-example-queries",  action="store_true", help="Use the built-in demo scenario instead of reading query files.")
    parser.add_argument("--wait",    dest="wait", action="store_true", default=WAIT, help="Wait for user input between high-level and low-level generation steps.")
    parser.add_argument("--no-wait", dest="wait", action="store_false", help="Disable waiting for user input between phases.")
    parser.add_argument("--single-query",         action="store_true", help="When set to true, the LLM will be instructed to create the whole HL and LL KB in one go instead of the single components." )

    return parser.parse_args()

def set_default_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.conf is MISSING:
        args.conf = LLM_CONF_PATH

    if args.consistency_checks is MISSING:
        args.consistency_checks = CONSISTENCY_CHECKS_PATH

    # Few-shot examples are optional: remember which paths come from the defaults
    # so that a missing 'examples' folder only warns, while an explicitly
    # requested (but absent) examples file still fails loudly.
    defaulted_examples = set()
    for arg_name, default_path in [
        ("cc_examples_config", CC_EXAMPLES_CONFIG_PATH),
        ("cc_examples_hl",     CC_EXAMPLES_HL_PATH),
        ("cc_examples_ll",     CC_EXAMPLES_LL_PATH),
        ("hl_examples_config", HL_EXAMPLES_CONFIG_PATH),
        ("ll_examples_config", LL_EXAMPLES_CONFIG_PATH),
    ]:
        if getattr(args, arg_name) is MISSING:
            setattr(args, arg_name, default_path)
            defaulted_examples.add(arg_name)
    args.defaulted_examples = defaulted_examples


    output_args = [args.output_path, args.output_cc, args.output_hl, args.output_ll, args.output_hl_kb, args.output_kb]
    individual_output_args = [args.output_cc, args.output_hl, args.output_ll, args.output_hl_kb, args.output_kb]

    # If all output file arguments are default, set them to the predefined paths.
    if all(arg is MISSING for arg in output_args):
        args.output_path = OUTPUT_PATH
        args.output_cc = OUTPUT_FILE_CC
        args.output_hl = OUTPUT_FILE_HL
        args.output_ll = OUTPUT_FILE_LL
        args.output_hl_kb = OUTPUT_HL_KB_FILE
        args.output_kb = OUTPUT_KB_FILE
    elif args.output_path is not MISSING:
        print(args.output_path)
        assert os.path.exists(args.output_path), f"Output path not found at {args.output_path}"
        if any(arg is not MISSING for arg in individual_output_args):
            raise ValueError("Cannot use individual output file arguments together with --output-path, See --help.")
        args.output_cc = os.path.join(args.output_path, os.path.basename(OUTPUT_FILE_CC))
        args.output_hl = os.path.join(args.output_path, os.path.basename(OUTPUT_FILE_HL))
        args.output_ll = os.path.join(args.output_path, os.path.basename(OUTPUT_FILE_LL))
        args.output_hl_kb = os.path.join(args.output_path, os.path.basename(OUTPUT_HL_KB_FILE))
        args.output_kb = os.path.join(args.output_path, os.path.basename(OUTPUT_KB_FILE))
    elif args.only_comprehension and args.output_cc is not MISSING:
        args.output_path = os.path.dirname(args.output_cc) or "."
        args.output_hl = OUTPUT_FILE_HL
        args.output_ll = OUTPUT_FILE_LL
        args.output_hl_kb = OUTPUT_HL_KB_FILE
        args.output_kb = OUTPUT_KB_FILE
    elif all(arg is not MISSING for arg in individual_output_args):
        pass
    else:
        logger.error(f"There is something wrong with the output file arguments. {args.output_path}, {args.output_cc}, {args.output_hl}, {args.output_ll}, {args.output_hl_kb}, {args.output_kb}")
        raise ValueError("If --output-path is not used, all individual output file arguments must be provided. See --help.")

    if args.query_hl_file is MISSING:
        args.query_hl_file = QUERY_HL_FILE
    if args.query_ll_file is MISSING:
        args.query_ll_file = QUERY_LL_FILE

    if args.verify_hl is MISSING:
        args.verify_hl = 3
    if args.verify_ll is MISSING:
        args.verify_ll = 3

    if args.log_file is not MISSING:
        log_dir = os.path.dirname(args.log_file)
        assert os.path.exists(log_dir), f"Log file directory not found at {log_dir}"
        logger.change_output_dir_file(log_dir, os.path.basename(args.log_file))

    if args.log_level is not MISSING:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if args.log_level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {args.log_level}. Valid options are: {', '.join(valid_levels)}")
        logger.setLevel(args.log_level.upper())
    
    assert os.path.exists(args.conf), f"LLM configuration file not found at {args.conf}"

    for arg_name, description in [
        ("cc_examples_config", "CC examples"),
        ("cc_examples_hl",     "CC high-level examples"),
        ("cc_examples_ll",     "CC low-level examples"),
        ("ll_examples_config", "Low-level examples"),
        ("hl_examples_config", "High-level examples"),
    ]:
        path = getattr(args, arg_name)
        if os.path.exists(path):
            continue
        # A default path is only a hint: the 'examples' folder is optional, so run
        # without few-shot examples instead of aborting.
        assert arg_name in getattr(args, "defaulted_examples", set()), \
            f"{description} path not found at {path}"
        logger.warning(f"{description} path not found at {path}, continuing without these few-shot examples.")
        setattr(args, arg_name, "")

    assert os.path.exists(args.consistency_checks), f"Consistency checks file not found at {args.consistency_checks}"

    logger.debug(f"Final arguments after setting defaults: {args}")

    return args
