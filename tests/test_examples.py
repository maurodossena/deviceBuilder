"""Every example still exports the deck that is committed with it.

The suite is driven by the examples under `examples/`: each one is a script that
builds a device and exports it, next to the input data it reads and the deck it
produced:

    examples/<name>/
        build_<something>.py   the script that builds the device
        <inputs>/              whatever the script loads, e.g. a CP2K run
        <deck>/config.toml     the deck the script exported

The test runs one example per case: it copies the directory to a temporary one,
runs the script there, and compares the deck it writes to the committed one,
file by file. Nothing here is specific to an example, so dropping a new one in
`examples/` is enough to have it covered.
"""

import contextlib
import os
import runpy
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py", reason="h5py is needed to read the matrix files")
toml = pytest.importorskip("toml", reason="toml is needed to read config.toml")

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"

# The build script of an example is the one whose name starts with `build`.
SCRIPT_GLOB = "build*.py"


@dataclass(frozen=True)
class Example:
    """One example device: a directory and the script that builds it."""

    name: str
    root: Path
    script: Path

    def __str__(self) -> str:
        return self.name


def discover_examples(examples_dir: Path = EXAMPLES_DIR) -> list[Example]:
    """
    Find every example that carries a build script.

    Parameters:
    -----------
    examples_dir : Path
        Directory holding one subdirectory per example.
    Returns:
    --------
    list
        The examples, sorted by name. A directory with several build scripts
        contributes one example per script.
    """

    examples = []
    for directory in sorted(p for p in examples_dir.glob("*") if p.is_dir()):
        scripts = sorted(directory.glob(SCRIPT_GLOB))
        for script in scripts:
            name = directory.name if len(scripts) == 1 else f"{directory.name}:{script.stem}"
            examples.append(Example(name=name, root=directory, script=script))
    return examples


def find_decks(root: Path) -> list[Path]:
    """
    Find the exported decks under a directory, i.e. what holds a `config.toml`.

    Parameters:
    -----------
    root : Path
        Directory to search.
    Returns:
    --------
    list
        The deck directories, sorted by path.
    """

    return sorted(path.parent for path in root.rglob("config.toml"))


@contextlib.contextmanager
def working_directory(path: Path):
    """
    Run a block with the process sitting in `path`.

    Parameters:
    -----------
    path : Path
        Directory to move into.
    """

    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def run_example(example: Example, workdir: Path) -> Path:
    """
    Run the script of an example in a copy of its own directory.

    Parameters:
    -----------
    example : Example
        The example to run.
    workdir : Path
        Empty directory to copy the example into and run it from.
    Returns:
    --------
    Path
        The deck the script wrote, at the same place the committed one sits.
    """

    shutil.copytree(example.root, workdir, dirs_exist_ok=True)

    # Drop the committed deck from the copy, so that only what this run writes
    # can be compared against it.
    for deck in find_decks(workdir):
        shutil.rmtree(deck)

    try:
        with working_directory(workdir):
            runpy.run_path(str(workdir / example.script.name))
    except ModuleNotFoundError as err:
        pytest.skip(f"{example.name} needs a package that is not installed: {err}")

    return workdir


def files_in(deck: Path) -> set:
    """
    List the files of a deck, relative to it.

    Parameters:
    -----------
    deck : Path
        The deck directory.
    Returns:
    --------
    set
        The paths of every file it holds.
    """

    return {path.relative_to(deck) for path in deck.rglob("*") if path.is_file()}


def assert_same_toml(generated, reference, path: str = ""):
    """
    Compare two parsed TOML trees, allowing floats to differ by a rounding error.

    Parameters:
    -----------
    generated, reference : object
        The two trees to compare.
    path : str
        Location within the tree, used in the failure message.
    """

    assert type(generated) is type(reference), f"type differs at {path or 'root'}"

    if isinstance(generated, dict):
        assert generated.keys() == reference.keys(), f"keys differ at {path or 'root'}"
        for key in generated:
            assert_same_toml(generated[key], reference[key], f"{path}.{key}")
    elif isinstance(generated, list):
        assert len(generated) == len(reference), f"length differs at {path}"
        for i, (a, b) in enumerate(zip(generated, reference)):
            assert_same_toml(a, b, f"{path}[{i}]")
    elif isinstance(generated, float):
        assert generated == pytest.approx(reference, abs=1e-9), f"value differs at {path}"
    else:
        assert generated == reference, f"value differs at {path}"


def assert_same_hdf5(generated: Path, reference: Path, name: str):
    """
    Compare two exported matrix files, block by block.

    Parameters:
    -----------
    generated, reference : Path
        The two files to compare.
    name : str
        Name of the file, used in the failure message.
    """

    with h5py.File(generated, "r") as new, h5py.File(reference, "r") as old:
        assert set(new) == set(old), f"{name} holds different blocks"
        for key in old:
            for dataset in old[key]:
                np.testing.assert_allclose(
                    new[key][dataset][:],
                    old[key][dataset][:],
                    atol=1e-12,
                    err_msg=f"{name}[{key}][{dataset}] differs",
                )


def assert_same_file(generated: Path, reference: Path, name: str):
    """
    Compare one file of the deck against the committed one.

    Parameters:
    -----------
    generated, reference : Path
        The two files to compare.
    name : str
        Path of the file within the deck, used in the failure message.
    """

    if reference.suffix == ".toml":
        assert_same_toml(toml.load(generated), toml.load(reference), name)
    elif reference.suffix == ".h5":
        assert_same_hdf5(generated, reference, name)
    elif reference.suffix == ".npy":
        np.testing.assert_allclose(
            np.load(generated), np.load(reference), err_msg=f"{name} differs"
        )
    else:
        assert generated.read_bytes() == reference.read_bytes(), f"{name} differs"


@pytest.mark.parametrize("example", discover_examples(), ids=str)
def test_the_example_reproduces_its_deck(example: Example, tmp_path: Path):
    decks = find_decks(example.root)
    if not decks:
        pytest.skip(f"no deck committed under {example.root}")
    assert len(decks) == 1, f"{example.name} has more than one deck: {decks}"
    reference_deck = decks[0]

    workdir = run_example(example, tmp_path)
    exported_deck = workdir / reference_deck.relative_to(example.root)
    assert (exported_deck / "config.toml").is_file(), (
        f"{example.script.name} did not write a deck to {exported_deck}"
    )

    assert files_in(exported_deck) == files_in(reference_deck)
    for path in sorted(files_in(reference_deck)):
        assert_same_file(exported_deck / path, reference_deck / path, str(path))
