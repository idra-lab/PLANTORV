#!/usr/bin/env python3
"""Download the model checkpoints used by the pipeline into ``models/``.

Covers:

- SAM 1 (Ultralytics, used through SAMModel -- SAM 1 ViT-H is downloaded from Meta because Ultralytics does not publish it);
- MobileSAM (Ultralytics, used through SAMModel);
- SAM 2 (Ultralytics, used through SAMModel);
- SAM 2.1 (Ultralytics, used through SAMModel);
- FastSAM (Ultralytics' YOLOv8-seg, used through FastSAMModel);
- SAM 3 (gated, requires a Hugging Face token);
- DAM-3B (Describe Anything, used through ``scene_understanding/dam_annotator.py``);
- RandLA-Net trained on S3DIS (Open3D-ML, used through ``segment_pcd.py``).

Ultralytics also downloads its own checkpoints on first use, so this script is
mainly useful for four cases:

- ``sam_h``, which Ultralytics does not publish at all. Meta's original weights
  work fine, they just have to be saved under the name Ultralytics expects.
- ``sam3``, whose weights are gated. Meta requires an approved access request
  on Hugging Face, so Ultralytics cannot fetch them on first use. Once access
  is granted, a machine that ran ``hf auth login`` needs nothing further;
  otherwise put a token in ``HF_TOKEN``, in the environment or in ``.env``.
- ``dam_3b``, which is a whole Hugging Face repository rather than one file.
  Nothing else in the pipeline downloads it, so without this it has to be
  fetched by hand.
- Pre-seeding ``models/`` before running somewhere without outbound network
  access, such as the cluster jobs in ``scripts/PBS/``.

Every checkpoint is saved under the name Ultralytics needs, because
``segmentation/sam_model.py`` selects the architecture from the file name; see
``SAM_CHECKPOINTS`` there for the full list. Meta distributes the same weights
as ``.pth``; the extension is only a naming convention, so saving them as
``.pt`` changes nothing about the contents. The Open3D-ML checkpoints are the
exception: ``segment_pcd.py`` is given their path explicitly, so they keep the
upstream file name, and ``dam_3b``, which keeps the layout its own loader
expects and so is saved as a directory rather than as a single file.

Open3D-ML models also need a configuration file, which Open3D ships inside the
installed package rather than publishing as a download. Those are copied out of
the ``open3d`` package of the interpreter running this script, so the version of
the configuration always matches the version of Open3D that will read it. A
missing one is reported as a warning, not an error: the checkpoint is still
usable with a configuration obtained by hand.


Examples
--------
List what can be downloaded::

    python3 scripts/install_models.py --list

Download one or more checkpoints by name::

    python3 scripts/install_models.py sam_h sam_b

Download the gated SAM 3 checkpoint, once the access request was approved. The
first form uses a stored ``hf auth login``, the second an explicit token::

    python3 scripts/install_models.py sam3
    HF_TOKEN=hf_... python3 scripts/install_models.py sam3

Download everything, or pick interactively when no name is given::

    python3 scripts/install_models.py --all
    python3 scripts/install_models.py
"""

import argparse
import importlib.util
import os
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.client import HTTPMessage
from pathlib import Path
from typing import IO, Optional, Sequence
from urllib.parse import urlsplit

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utility.utility import bold, logger  # noqa: E402

# Read/connect timeout for the download requests, in seconds.
TIMEOUT = 30
# Size of a single read from the socket. Large enough to keep the 2.4 GB ViT-H
# download from being dominated by Python-level loop overhead.
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class Model:
    """A downloadable checkpoint.

    Attributes
    ----------
    key : str
        Name used to select the model on the command line. Identical to the
        file name without its extension. For the SAM checkpoints this is also
        what gets passed to ``SAMModel``.
    filename : str
        Name the checkpoint is saved as. This is the name Ultralytics needs to
        recognise the architecture, not necessarily the upstream one. For a
        ``repo_id`` model it names the directory the repository is saved into
        instead, since those are many files rather than one.
    url : str or None
        Direct download URL, for a checkpoint that is a single file. Mutually
        exclusive with ``repo_id``.
    repo_id : str or None
        Hugging Face repository holding the checkpoint, for the models that are
        a tree of files rather than one (``nvidia/DAM-3B``, for instance, is 19
        files across four subdirectories). Every file in the repository is
        downloaded, keeping its path inside ``models/<subdir>/<filename>/``.
        Mutually exclusive with ``url``.
    subdir : str
        Directory under the models root where the file is stored.
    description : str
        Short human-readable summary shown by ``--list`` and the picker.
    requires_hf_token : bool
        True when the URL points at a gated Hugging Face repository, which only
        answers to a request carrying a token of an account whose access request
        was approved. See :func:`hf_token`.
    config : str or None
        File name of the Open3D-ML configuration the checkpoint is used with,
        copied next to it out of the installed ``open3d`` package. None for the
        checkpoints that need no configuration file. See :func:`install_config`.
    """

    key: str
    subdir: str
    description: str
    filename: Optional[str] = None
    url: Optional[str] = None
    repo_id: Optional[str] = None
    requires_hf_token: bool = False
    config: Optional[str] = None

    def __post_init__(self) -> None:
        """Reject a model that is neither a single file nor a repository.

        Raises
        ------
        ValueError
            If neither or both of ``url`` and ``repo_id`` are set, or if
            ``filename`` is missing.
        """
        if (self.url is None) == (self.repo_id is None):
            raise ValueError(
                f"{self.key}: set exactly one of url (single file) or repo_id (repository)"
            )
        if self.filename is None:
            raise ValueError(f"{self.key}: filename is required")

    @property
    def repo_url(self) -> str:
        """Human-facing address of the Hugging Face repository the weights come from."""
        return f"{HF_BASE_URL}/{self.repo_id}" if self.repo_id else HF_SAM3_REPO_URL


# Checkpoints Ultralytics publishes itself, on the release its own downloader
# defaults to. These are the same weights Ultralytics would fetch on first use.
ULTRALYTICS_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
# Meta's original checkpoints, from https://github.com/facebookresearch/segment-anything.
# Only needed for ViT-H. The weights are unchanged, they are just stored under
# the `.pt` name Ultralytics matches on rather than Meta's `.pth` one.
META_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything"
# SAM 3, which Meta distributes only through its gated Hugging Face repository.
# The repository holds `sam3.pt` next to the Transformers weights, and that file
# is the one Ultralytics loads, so it needs no renaming.
HF_BASE_URL = "https://huggingface.co"
HF_SAM3_REPO_URL = f"{HF_BASE_URL}/facebook/sam3"
HF_SAM3_URL = f"{HF_SAM3_REPO_URL}/resolve/main/sam3.pt"
# Environment variables searched for a Hugging Face token, in order. The first is
# what `.env` is expected to carry; the second is the older name their own
# libraries still honour.
HF_TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
# Where `hf auth login` stores its token when no environment variable is set.
# `HF_TOKEN_PATH` names the file outright, otherwise it sits under `HF_HOME`.
# Resolved the same way `huggingface_hub` does, so a machine that is already
# logged in needs no further setup.
HF_HOME_DEFAULT = "~/.cache/huggingface"
# Open3D-ML's model zoo, which hosts the point-cloud semantic segmentation weights.
OPEN3D_ZOO_URL = "https://storage.googleapis.com/open3d-releases/model-zoo"
# Where the matching configurations sit inside an installed `open3d`, relative to the
# package root. Open3D vendors Open3D-ML as `open3d._ml3d`, so they ship with the wheel
# and are never downloaded.
OPEN3D_CONFIG_SUBDIR = ("_ml3d", "configs")
# Upstream copy of the same configurations, for the warning shown when the installed
# Open3D has none.
OPEN3D_ML_CONFIGS_URL = "https://github.com/isl-org/Open3D-ML/tree/main/ml3d/configs"

# SAM 1. Sources are mixed, so these stay spelled out one by one.
MODELS: dict[str, Model] = {
    "sam_h": Model(
        key="sam_h",
        filename="sam_h.pt",
        url=f"{META_BASE_URL}/sam_vit_h_4b8939.pth",
        subdir="sam",
        description="SAM 1 ViT-H (pipeline default, best quality, ~2.4 GB) [from Meta]",
    ),
    "sam_l": Model(
        key="sam_l",
        filename="sam_l.pt",
        url=f"{ULTRALYTICS_BASE_URL}/sam_l.pt",
        subdir="sam",
        description="SAM 1 ViT-L (~1.2 GB) [from Ultralytics]",
    ),
    "sam_b": Model(
        key="sam_b",
        filename="sam_b.pt",
        url=f"{ULTRALYTICS_BASE_URL}/sam_b.pt",
        subdir="sam",
        description="SAM 1 ViT-B (~360 MB) [from Ultralytics]",
    ),
    "mobile_sam": Model(
        key="mobile_sam",
        filename="mobile_sam.pt",
        url=f"{ULTRALYTICS_BASE_URL}/mobile_sam.pt",
        subdir="sam",
        description="MobileSAM (SAM 1, tiny distilled encoder, ~39 MB) [from Ultralytics]",
    ),
}

# SAM 2 and SAM 2.1. Both generations ship the same four sizes from the same
# place, so they are generated rather than repeated eight times. SAM 2.1 is the
# later release of the same architecture and supersedes SAM 2 at equal size;
# SAM 2 is kept so older runs stay reproducible.
_SAM2_VARIANTS = {
    "t": ("tiny", "~75 MB"),
    "s": ("small", "~88 MB"),
    "b": ("base+", "~154 MB"),
    "l": ("large", "~428 MB"),
}

for _generation, _note in (("sam2", ""), ("sam2.1", ", recommended over sam2")):
    for _size, (_label, _weight) in _SAM2_VARIANTS.items():
        _key = f"{_generation}_{_size}"
        MODELS[_key] = Model(
            key=_key,
            filename=f"{_key}.pt",
            url=f"{ULTRALYTICS_BASE_URL}/{_key}.pt",
            subdir="sam",
            description=(
                f"SAM {_generation.removeprefix('sam')} {_label} "
                f"({_weight}{_note}) [from Ultralytics]"
            ),
        )

# SAM 3. A concept segmenter rather than a purely geometric one. Unlike every
# other checkpoint here the weights are gated, hence the token.
MODELS["sam3"] = Model(
    key="sam3",
    filename="sam3.pt",
    url=HF_SAM3_URL,
    subdir="sam",
    description="SAM 3 (concept segmentation, ~3.2 GB) [from Meta, gated: needs HF access]",
    requires_hf_token=True,
)

# FastSAM. Not a SAM architecture at all.
for _size, _weight in (("s", "~23 MB"), ("x", "~138 MB")):
    _key = f"FastSAM-{_size}"
    MODELS[_key] = Model(
        key=_key,
        filename=f"{_key}.pt",
        url=f"{ULTRALYTICS_BASE_URL}/{_key}.pt",
        subdir="fastsam",
        description=f"FastSAM {_size} ({_weight}, used via FastSAMModel) [from Ultralytics]",
    )

# Describe Anything, used through `scene_understanding/dam_annotator.py` as the local
# alternative to the remote LLM in `gpt_annotator.py`. Unlike every other entry here
# this is a whole repository rather than one file: DAM's loader expects `llm/`,
# `vision_tower/`, `context_provider/` and `mm_projector/` beside a top-level
# `config.json`, so the tree is mirrored as published. The weights are open, but the
# `dam` package that reads them is not on PyPI; see requirements.txt.
MODELS["dam_3b"] = Model(
    key="dam_3b",
    filename="DAM-3B",
    repo_id="nvidia/DAM-3B",
    subdir="dam",
    description="DAM-3B (region captioning, ~7.1 GB in 19 files, used via DAMAnnotator) [from NVIDIA]",
)

# Open3D-ML point-cloud semantic segmentation, used through `segment_pcd.py` rather
# than through any of the SAM wrappers. The name encodes the training run, and
# `segment_pcd.py` takes the path explicitly, so it is kept exactly as published.
MODELS["randlanet_s3dis"] = Model(
    key="randlanet_s3dis",
    filename="randlanet_s3dis_202201071330utc.pth",
    url=f"{OPEN3D_ZOO_URL}/randlanet_s3dis_202201071330utc.pth",
    subdir="randla_net",
    description=(
        "RandLA-Net, 13 S3DIS indoor classes (~57 MB, used via segment_pcd.py) [from Open3D-ML]"
    ),
    config="randlanet_s3dis.yml",
)

# Width of the key column in `--list` and in the interactive picker, sized to the
# longest key so a new model cannot silently break the alignment.
KEY_WIDTH = max(len(key) for key in MODELS) + 1


def hf_token() -> Optional[str]:
    """Return the Hugging Face token, from the environment or from a stored login.

    The variables in :data:`HF_TOKEN_VARS` are tried first, in order. ``.env`` has
    already been loaded by :func:`main`, so a token written there counts as being
    in the environment. Failing that, the token ``hf auth login`` writes to disk is
    used, which is what makes an already logged in machine work with no setup.

    Returns
    -------
    str or None
        The token, or None when neither the environment nor a stored login
        provides a non-empty one.
    """
    for variable in HF_TOKEN_VARS:
        token = os.environ.get(variable, "").strip()
        if token:
            return token

    token_path = os.environ.get("HF_TOKEN_PATH", "").strip()
    path = (
        Path(token_path)
        if token_path
        else Path(os.environ.get("HF_HOME", "").strip() or HF_HOME_DEFAULT) / "token"
    )
    try:
        return path.expanduser().read_text(encoding="utf-8").strip() or None
    except OSError:
        # Missing or unreadable is just "not logged in", not an error worth raising.
        return None


class _StripAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Redirect handler that drops ``Authorization`` when the host changes.

    Hugging Face answers a download with a redirect to a CDN that authenticates
    the request through a signature in the URL itself. Forwarding the bearer
    token there is useless and the CDN rejects requests that carry both, but
    ``urllib`` copies every header onto the redirected request by default.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> Optional[urllib.request.Request]:
        """Build the redirected request, without the token if it leaves the host.

        Parameters
        ----------
        req : urllib.request.Request
            The request that was redirected.
        fp : IO[bytes]
            The response body of the redirect.
        code : int
            HTTP status code of the redirect.
        msg : str
            HTTP status message of the redirect.
        headers : http.client.HTTPMessage
            Headers of the redirect response.
        newurl : str
            URL being redirected to.

        Returns
        -------
        urllib.request.Request or None
            The request to send next, or None when the redirect is not followed.
        """
        new_request = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_request is not None and urlsplit(newurl).netloc != urlsplit(req.full_url).netloc:
            new_request.remove_header("Authorization")
        return new_request


def human_size(num_bytes: float) -> str:
    """Format a byte count as a short human-readable string.

    Parameters
    ----------
    num_bytes : float
        Size in bytes.

    Returns
    -------
    str
        Size rounded to one decimal with a binary unit suffix, e.g. ``"2.4 GiB"``.
    """
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def open3d_configs_dir() -> Optional[Path]:
    """Locate the Open3D-ML configuration directory of the interpreter running this script.

    Found through the import system rather than by building a path out of a
    Python version, so it resolves to whichever ``open3d`` the interpreter would
    actually import: the virtualenv's, a user install, or a system one.
    ``importlib.util.find_spec`` only locates the package, it does not execute
    it, which keeps this off Open3D's slow native import.

    Returns
    -------
    pathlib.Path or None
        The directory holding the bundled ``*.yml`` configurations, or None when
        Open3D is not installed or ships no Open3D-ML configurations.
    """
    try:
        spec = importlib.util.find_spec("open3d")
    except (ImportError, ValueError):
        # ImportError: no such package. ValueError: `open3d` is in sys.modules
        # without a spec, which a broken or partial install can produce.
        return None

    if spec is None or spec.submodule_search_locations is None:
        return None

    for location in spec.submodule_search_locations:
        candidate = Path(location).joinpath(*OPEN3D_CONFIG_SUBDIR)
        if candidate.is_dir():
            return candidate
    return None


def install_config(model: Model, models_dir: Path, force: bool = False) -> bool:
    """Copy a model's Open3D-ML configuration next to its checkpoint.

    A missing configuration is a warning rather than a failure: the checkpoint is
    downloaded and usable, and the file can be fetched by hand from Open3D-ML.

    Parameters
    ----------
    model : Model
        Model whose configuration to install. Does nothing when it needs none.
    models_dir : Path
        Root directory the model subdirectories are created in.
    force : bool, optional
        Copy again even if the configuration is already there.

    Returns
    -------
    bool
        True if the configuration is present after the call, False if it could
        not be found or copied.
    """
    if model.config is None:
        return True

    target = models_dir / model.subdir / model.config
    if target.exists() and not force:
        logger.info(f"{model.key}: configuration already present at {target}")
        return True

    configs_dir = open3d_configs_dir()
    if configs_dir is None:
        logger.warning(
            f"{model.key}: no Open3D-ML configurations found for {sys.executable}, so "
            f"{model.config} was not installed."
        )
        logger.warning(
            f"  Install Open3D for this interpreter, or copy {model.config} by hand from "
            f"{OPEN3D_ML_CONFIGS_URL} into {target.parent}."
        )
        return False

    source = configs_dir / model.config
    if not source.is_file():
        logger.warning(
            f"{model.key}: {model.config} is not in {configs_dir}, so it was not installed."
        )
        logger.warning(
            f"  The installed Open3D may be too old or too new for this checkpoint. Copy the file "
            f"by hand from {OPEN3D_ML_CONFIGS_URL} into {target.parent}."
        )
        return False

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    except OSError as error:
        logger.warning(f"{model.key}: could not copy {source} to {target} ({error}).")
        return False

    logger.info(f"{model.key}: configuration copied from {source} to {target}")
    return True


def list_models() -> None:
    """Print the available models, their destination, and their description."""
    logger.info(bold("Available models:"))
    for model in MODELS.values():
        logger.info(f"  {model.key:<{KEY_WIDTH}} {model.description}")
        # A repository lands as a directory, so it is shown with a trailing slash:
        # the file names inside it are only known once the Hub has been asked.
        suffix = "/ (whole repository)" if model.repo_id else ""
        logger.info(f"  {'':<{KEY_WIDTH}} -> models/{model.subdir}/{model.filename}{suffix}")
        if model.config is not None:
            logger.info(
                f"  {'':<{KEY_WIDTH}} -> models/{model.subdir}/{model.config} "
                "(copied from the installed open3d)"
            )
    if any(model.requires_hf_token for model in MODELS.values()):
        logger.info(
            f"Models marked gated need an approved access request at {HF_SAM3_REPO_URL}, plus "
            f"either a stored `hf auth login` token or {' or '.join(HF_TOKEN_VARS)} set in the "
            "environment or in .env."
        )


def prompt_for_models() -> list[Model]:
    """Ask the user which models to download.

    Returns
    -------
    list[Model]
        Selected models, empty if the user cancelled or the selection was invalid.
    """
    models = list(MODELS.values())

    print(bold("Which models do you want to download?"), file=sys.stderr)
    for position, model in enumerate(models, start=1):
        print(f"  {position}) {model.key:<{KEY_WIDTH}} {model.description}", file=sys.stderr)
    print("  a) all of them", file=sys.stderr)

    try:
        answer = input("Selection (e.g. '1', '1,3', 'a'; empty to cancel): ").strip()
    except EOFError:
        # Non-interactive stdin: treat it like an empty answer rather than crashing.
        answer = ""

    if not answer:
        logger.info("Nothing selected, exiting.")
        return []

    if answer.lower() in {"a", "all"}:
        return models

    selected: list[Model] = []
    for token in answer.replace(" ", ",").split(","):
        if not token:
            continue
        if token in MODELS:
            selected.append(MODELS[token])
            continue
        if not token.isdigit() or not 1 <= int(token) <= len(models):
            logger.error(f"Invalid selection: {token}")
            return []
        selected.append(models[int(token) - 1])

    # Deduplicate while keeping the order the user typed.
    unique: list[Model] = []
    for model in selected:
        if model not in unique:
            unique.append(model)
    return unique


def _report_progress(downloaded: int, total: Optional[int], last_reported: int) -> int:
    """Report download progress on stderr.

    On a TTY the progress is redrawn in place; otherwise (log files, PBS jobs) it
    is logged once every 10 percent to keep the output readable.

    Parameters
    ----------
    downloaded : int
        Bytes written so far.
    total : int or None
        Expected total size, or None when the server did not advertise one.
    last_reported : int
        Percentage reported by the previous call, used to throttle non-TTY output.

    Returns
    -------
    int
        Percentage that has now been reported.
    """
    if sys.stderr.isatty():
        if total:
            percent = downloaded * 100 // total
            bar_width = 30
            filled = bar_width * downloaded // total
            bar = "#" * filled + "-" * (bar_width - filled)
            line = f"  [{bar}] {percent:3d}% ({human_size(downloaded)}/{human_size(total)})"
        else:
            percent = last_reported
            line = f"  {human_size(downloaded)} downloaded"
        print(f"\r{line}", end="", file=sys.stderr, flush=True)
        return percent

    if not total:
        return last_reported

    percent = downloaded * 100 // total
    if percent >= last_reported + 10:
        logger.info(f"  {percent}% ({human_size(downloaded)}/{human_size(total)})")
        return percent - percent % 10
    return last_reported


def _resolve_token(model: Model) -> tuple[bool, Optional[str]]:
    """Find the Hugging Face token a model needs, if it needs one.

    Resolved once per model rather than once per file, so a repository download
    fails before its first request instead of part way through the tree.

    Parameters
    ----------
    model : Model
        Model about to be downloaded.

    Returns
    -------
    tuple[bool, str or None]
        ``(True, None)`` when no token is needed, ``(True, token)`` when one was
        found, and ``(False, None)`` when one is needed but could not be found.
    """
    if not model.requires_hf_token:
        return True, None

    token = hf_token()
    if token is None:
        logger.error(
            f"{model.key}: the weights are gated and no token was found in "
            f"{' or '.join(HF_TOKEN_VARS)} or in a stored Hugging Face login."
        )
        logger.error(
            f"  Request access at {model.repo_url}, then either run `hf auth login` or "
            "put the token of the account the access was granted to in .env."
        )
        return False, None

    return True, token


def _download_file(
    url: str,
    target: Path,
    label: str,
    token: Optional[str] = None,
    force: bool = False,
) -> bool:
    """Download one file, resuming a previous partial download.

    Parameters
    ----------
    url : str
        Direct download URL.
    target : Path
        Where the finished file is written. Parent directories are created.
    label : str
        Name used in the log lines, e.g. a model key or ``"dam_3b: llm/config.json"``.
    token : str or None
        Bearer token sent with the request, for gated repositories.
    force : bool, optional
        Discard a previous partial download instead of resuming it.

    Returns
    -------
    bool
        True if the file is present and complete after the call.
    """
    partial = target.with_suffix(target.suffix + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)

    if force and partial.exists():
        partial.unlink()

    request = urllib.request.Request(url)
    if token is not None:
        request.add_header("Authorization", f"Bearer {token}")

    offset = partial.stat().st_size if partial.exists() else 0
    if offset:
        logger.info(f"{label}: resuming from {human_size(offset)}")
        request.add_header("Range", f"bytes={offset}-")

    logger.info(f"{label}: downloading {url}")
    downloaded = offset
    total: Optional[int] = None
    # A plain urlopen would forward the token to whatever host the download is
    # redirected to; see `_StripAuthOnRedirect`.
    opener = urllib.request.build_opener(_StripAuthOnRedirect)
    try:
        with opener.open(request, timeout=TIMEOUT) as response:
            # A 206 means the range was honoured; anything else (typically 200)
            # means the server is sending the whole file, so start over.
            if offset and response.status != 206:
                logger.warning("  Server ignored the resume request, restarting the download.")
                offset = 0

            content_length = response.headers.get("Content-Length")
            remaining = int(content_length) if content_length is not None else None
            total = offset + remaining if remaining is not None else None
            if total:
                logger.info(f"  Total size: {human_size(total)}")

            downloaded = offset
            reported = 0
            # Append when resuming, truncate when starting (or restarting) from zero.
            mode = "ab" if offset else "wb"
            with open(partial, mode) as handle:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    reported = _report_progress(downloaded, total, reported)
    except urllib.error.HTTPError as error:
        logger.error(f"{label}: HTTP error {error.code} ({error.reason}) for {url}")
        if token is not None and error.code in (401, 403):
            # The token was sent, so this is about the account behind it rather
            # than about the token being missing.
            logger.error(
                "  The token was rejected. Check that the access request was approved for "
                "this account and that the token can read gated repositories."
            )
        return False
    except urllib.error.URLError as error:
        logger.error(f"{label}: could not reach {url} ({error.reason})")
        return False
    except OSError as error:
        logger.error(f"{label}: failed to write {partial} ({error})")
        return False
    finally:
        if sys.stderr.isatty():
            print(file=sys.stderr)

    if total is not None and downloaded != total:
        # Keep the .part file so the next run can resume instead of restarting.
        logger.error(
            f"{label}: incomplete download, got {human_size(downloaded)} of {human_size(total)}. "
            "Re-run the script to resume."
        )
        return False

    partial.replace(target)
    logger.info(f"{label}: saved to {target} ({human_size(target.stat().st_size)})")
    return True


def _download_repo(
    model: Model,
    models_dir: Path,
    token: Optional[str] = None,
    force: bool = False,
) -> bool:
    """Download every file of a Hugging Face repository, keeping its layout.

    Used for the checkpoints that are a tree rather than a single file, such as
    DAM-3B, whose loader expects to find ``llm/``, ``vision_tower/``,
    ``context_provider/`` and ``mm_projector/`` next to a top-level
    ``config.json``. The tree is mirrored under
    ``models/<subdir>/<filename>/``.

    The file list is read from the Hub rather than hardcoded, so re-sharded
    weights are picked up without editing this script. Files already on disk are
    skipped, which makes an interrupted run resumable at file granularity on top
    of the byte-level resume in :func:`_download_file`.

    Parameters
    ----------
    model : Model
        Model to download. Its ``repo_id`` must be set.
    models_dir : Path
        Root directory the model subdirectories are created in.
    token : str or None
        Bearer token, for gated repositories.
    force : bool, optional
        Re-download files that already exist.

    Returns
    -------
    bool
        True if every file of the repository is present after the call.
    """
    root = models_dir / model.subdir / str(model.filename)

    try:
        from huggingface_hub import list_repo_files
    except ImportError:
        logger.error(
            f"{model.key}: listing {model.repo_id} needs the huggingface_hub package. "
            "Install it with `pip install huggingface-hub`."
        )
        return False

    try:
        files = sorted(list_repo_files(str(model.repo_id), token=token))
    except Exception as error:
        # The Hub client raises its own exception types, and which ones depend on
        # its version, so this stays broad on purpose: any failure here means the
        # listing is unusable and the download cannot start.
        logger.error(f"{model.key}: could not list {model.repo_url} ({error})")
        return False

    if not files:
        logger.error(f"{model.key}: {model.repo_url} lists no files")
        return False

    missing = [name for name in files if not (root / name).exists()]
    if not missing and not force:
        size = human_size(sum((root / name).stat().st_size for name in files))
        logger.info(f"{model.key}: already present at {root} ({len(files)} files, {size})")
        logger.info("  Use --force to download it again.")
        return True

    wanted = files if force else missing
    logger.info(f"{model.key}: {len(wanted)} of {len(files)} file(s) to download into {root}")

    failed: list[str] = []
    for position, name in enumerate(wanted, start=1):
        url = f"{model.repo_url}/resolve/main/{name}"
        label = f"{model.key} [{position}/{len(wanted)}] {name}"
        if not _download_file(url, root / name, label, token, force):
            failed.append(name)

    if failed:
        logger.error(f"{model.key}: {len(failed)} file(s) failed: {', '.join(failed)}")
        return False

    logger.info(f"{model.key}: saved to {root}")
    return True


def download(model: Model, models_dir: Path, force: bool = False) -> bool:
    """Download a checkpoint, resuming a previous partial download.

    Dispatches on how the checkpoint is published: a single file goes straight to
    :func:`_download_file`, a repository to :func:`_download_repo`.

    Parameters
    ----------
    model : Model
        Model to download.
    models_dir : Path
        Root directory the model subdirectories are created in.
    force : bool, optional
        Re-download even if the target already exists.

    Returns
    -------
    bool
        True if the checkpoint is present and complete after the call.
    """
    available, token = _resolve_token(model)
    if not available:
        return False

    if model.repo_id is not None:
        return _download_repo(model, models_dir, token, force)

    target = models_dir / model.subdir / str(model.filename)
    if target.exists() and not force:
        size = human_size(target.stat().st_size)
        logger.info(f"{model.key}: already present at {target} ({size})")
        logger.info("  Use --force to download it again.")
        return True

    return _download_file(str(model.url), target, model.key, token, force)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line arguments.

    Parameters
    ----------
    argv : Sequence[str], optional
        Arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download model checkpoints into the models/ directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available models: " + ", ".join(MODELS),
    )
    parser.add_argument(
        "models",
        nargs="*",
        default=[],
        metavar="MODEL",
        help="Models to download. If omitted, the script asks interactively.",
    )
    parser.add_argument("--all", action="store_true", help="Download every available model.")
    parser.add_argument("--list", action="store_true", help="List the available models and exit.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Root directory for the checkpoints (default: %(default)s).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download checkpoints that already exist."
    )
    args = parser.parse_args(argv)

    unknown = [key for key in args.models if key not in MODELS]
    if unknown:
        parser.error(f"unknown model(s): {', '.join(unknown)}. Choose from: {', '.join(MODELS)}")

    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the installer.

    Parameters
    ----------
    argv : Sequence[str], optional
        Command line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code: 0 if every requested checkpoint is available, 1 otherwise.
        A configuration file that could not be installed only warns; see
        :func:`install_config`.
    """
    args = parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env")

    if args.list:
        list_models()
        return 0

    if args.all:
        selected = list(MODELS.values())
        if hf_token() is None:
            # Downloading these is guaranteed to fail, and failing them would make
            # `--all` report an error for a checkpoint the user never named.
            gated = [model for model in selected if model.requires_hf_token]
            if gated:
                selected = [model for model in selected if not model.requires_hf_token]
                logger.warning(
                    f"Skipping {', '.join(model.key for model in gated)}: gated weights and no "
                    "Hugging Face token found. See --list."
                )
    elif args.models:
        selected = [MODELS[key] for key in dict.fromkeys(args.models)]
    else:
        selected = prompt_for_models()

    if not selected:
        return 0

    free_space = shutil.disk_usage(PROJECT_ROOT).free
    logger.info(
        f"Downloading {len(selected)} model(s) into {args.models_dir} "
        f"({human_size(free_space)} free on disk)"
    )

    failed: list[str] = []
    for model in selected:
        if not download(model, args.models_dir, args.force):
            failed.append(model.key)
            continue
        # A configuration that could not be installed only warns, so it does not
        # count as a failure: the checkpoint itself is there.
        install_config(model, args.models_dir, args.force)

    if failed:
        logger.error(f"Failed: {', '.join(failed)}")
        return 1

    logger.info(bold("All requested models are installed."))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("Interrupted. Re-run the script to resume the download.")
        sys.exit(130)
