"""Execute walkthrough cells that are not tagged `live`.

Live cells call the model or open widgets. The published notebook keeps their
source and leaves them unexecuted. Snapshot cells load saved judgments and
are what GitHub readers see.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "walkthrough_local_judge.ipynb"
KERNEL = "pc-snapshot"


class SnapshotClient(NotebookClient):
    async def async_execute_cell(  # type: ignore[override]
        self,
        cell,
        cell_index: int,
        execution_count: int | None = None,
        store_history: bool = True,
    ):
        tags = list((cell.get("metadata") or {}).get("tags") or [])
        if "live" in tags:
            return cell
        return await super().async_execute_cell(
            cell,
            cell_index,
            execution_count=execution_count,
            store_history=store_history,
        )


def main() -> None:
    os.environ["MPLBACKEND"] = "Agg"
    mpl_dir = ROOT / ".mplconfig"
    mpl_dir.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--sys-prefix",
            "--name",
            KERNEL,
            "--display-name",
            "preference-consistency snapshot",
        ]
    )
    nb = nbformat.read(NB_PATH, as_version=4)
    client = SnapshotClient(
        nb,
        timeout=600,
        kernel_name=KERNEL,
        resources={"metadata": {"path": str(ROOT / "notebooks")}},
        allow_errors=False,
    )
    client.execute()
    nbformat.write(nb, NB_PATH)
    print("executed snapshot cells in", NB_PATH)


if __name__ == "__main__":
    main()
