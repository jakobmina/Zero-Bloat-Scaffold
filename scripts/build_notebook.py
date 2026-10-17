"""Splice the current `agent/my_agent.py` into `notebooks/submission.ipynb`.

The notebook follows the exact pattern used by Kaggle's official sample
("ARC3 Sample Submission - Stochastic Goose"):

  Cell 1: install the `arc-agi` wheel from the offline competition dataset.
  Cell 2: write `my_agent.py` to /kaggle/working/ — its body is THIS file.
  Cell 3: if running inside the Kaggle competition rerun, wait for the
          gateway sidecar, copy the framework into /kaggle/working/, register
          MyAgent, and run `python main.py --agent myagent`.
  Cell 4: otherwise (during commit / save-and-run-all), write a dummy
          submission.parquet so Kaggle accepts the commit.

You don't normally need to call this directly — `make submit` runs it for you.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE THIS ONE LINE TO PICK YOUR KAGGLE ACCELERATOR
# Options:
#   "cpu"      — no GPU. Good for the random starter or any non-ML agent.
#   "t4"       — Nvidia T4 ×2 (default; matches Kaggle's sample submission).
#   "p100"     — Nvidia P100 (single big-memory GPU).
#   "rtx6000"  — Nvidia RTX 6000 (g4-standard-48). ARC-AGI-3 exclusive,
#                burns GPU quota faster — use only when you're confident.
# ─────────────────────────────────────────────────────────────────────────────
ACCELERATOR = "cpu"  # el explorador H7 no usa GPU: no gastes cuota

# Internal mapping; don't edit unless Kaggle adds new options.
_ACCELERATORS = {
    "cpu":     {"name": "none",            "gpu": False},
    "t4":      {"name": "nvidiaTeslaT4",   "gpu": True},
    "p100":    {"name": "nvidiaTeslaP100", "gpu": True},
    "rtx6000": {"name": "nvidiaRtx6000",   "gpu": True},
}

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "my_agent.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build() -> dict:
    if not AGENT_SRC.exists():
        raise SystemExit(f"Could not find {AGENT_SRC}")
    agent_body = AGENT_SRC.read_text()

    install_cell = code_cell(
        "!pip install --no-index --find-links \\\n"
        "    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n"
        "    arc-agi python-dotenv"
    )

    # We write the agent to /tmp/ (not /kaggle/working/) so it does NOT appear
    # as a notebook output. Otherwise the "Submit to Competition" UI would
    # offer it as a candidate submission file alongside submission.parquet,
    # and an unlucky default selection rejects the submission.
    write_agent_cell = code_cell(
        "%%writefile /tmp/my_agent.py\n" + agent_body
    )

    run_cell_source = dedent(
        """\
        import os

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            # Wait for the gateway sidecar to be ready.
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                  --retry-max-time 600 http://gateway:8001/api/games

            # Copy the framework into a writable location.
            !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
                   /kaggle/working/ARC-AGI-3-Agents

            # Drop our agent in as a framework template.
            !cp /tmp/my_agent.py \\
                /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py

            # Register MyAgent in the framework's agent registry. We rewrite
            # __init__.py because the upstream version eagerly imports
            # templates with deps we don't ship (langgraph, smolagents, etc.).
            with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
                f.write(\"\"\"from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent

        load_dotenv()

        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
            'random': Random,
            'myagent': MyAgent,
        }
        \"\"\")

            # Point the framework at the gateway sidecar.
            with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
                f.write(\"\"\"SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/kaggle/working/server_recording
        \"\"\")

            # Run it. The gateway records every action and emits submission.parquet.
            !cd /kaggle/working/ARC-AGI-3-Agents && \\
                MPLBACKEND=agg \\
                python main.py --agent myagent
        """
    )
    run_cell = code_cell(run_cell_source)

    dummy_submission_cell = code_cell(
        dedent(
            """\
            import os
            if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                # Save-and-run-all (commit) mode: emit a dummy submission so the
                # commit succeeds. The real submission.parquet is produced by the
                # gateway during competition rerun.
                import pandas as pd
                submission = pd.DataFrame(
                    data=[['1_0', '1', True, 1]],
                    columns=['row_id', 'game_id', 'end_of_game', 'score'])
                submission.to_parquet('/kaggle/working/submission.parquet', index=False)
                submission.head()
            """
        )
    )

    # Evaluación local contra los juegos públicos que la competencia monta en
    # environment_files/. Solo corre en Save & Run All (no en el rerun), va
    # DESPUÉS del parquet dummy y está envuelta en try/except: nunca puede
    # romper el commit. Trabaja en /tmp para no agregar outputs al notebook.
    local_eval_cell = code_cell(
        dedent(
            """\
            # ── Evaluación local con los juegos públicos (solo en Save & Run All) ──
            import os
            if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                import importlib.util, logging, shutil, sys, time, traceback
                COMP = '/kaggle/input/competitions/arc-prize-2026-arc-agi-3'
                EVAL_STEPS = 2000          # tope de acciones por juego en esta evaluación
                WORK = '/tmp/arc3_local'   # fuera de /kaggle/working: no ensucia los outputs
                cwd = os.getcwd()
                try:
                    shutil.rmtree(WORK, ignore_errors=True)
                    shutil.copytree(f'{COMP}/ARC-AGI-3-Agents', f'{WORK}/ARC-AGI-3-Agents')
                    # __init__ vacío: el original importa plantillas LLM que no vienen instaladas
                    open(f'{WORK}/ARC-AGI-3-Agents/agents/__init__.py', 'w').close()
                    sys.path.insert(0, f'{WORK}/ARC-AGI-3-Agents')
                    os.chdir(WORK)
                    logging.getLogger().setLevel(logging.WARNING)

                    import arc_agi
                    from arc_agi import OperationMode
                    spec = importlib.util.spec_from_file_location('my_agent_local', '/tmp/my_agent.py')
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    MyAgent = mod.MyAgent
                    MyAgent.MAX_ACTIONS = min(MyAgent.MAX_ACTIONS, EVAL_STEPS)

                    arc = arc_agi.Arcade(operation_mode=OperationMode.OFFLINE,
                                         environments_dir=f'{COMP}/environment_files',
                                         recordings_dir=f'{WORK}/recordings')
                    games = sorted({e.game_id.split('-')[0] for e in arc.get_environments()})
                    print(f'{len(games)} juegos públicos · tope {EVAL_STEPS} acciones por juego\\n')
                    t0 = time.time()
                    for gid in games:
                        env = arc.make(gid)
                        if env is None:
                            print(f'  {gid:6} no se pudo crear')
                            continue
                        ag = MyAgent(card_id='local', game_id=gid, agent_name=f'local.{gid}',
                                     ROOT_URL='http://localhost', record=False, arc_env=env,
                                     tags=['local'])
                        ag.main()
                        f = ag.frames[-1]
                        print(f'  {gid:6} niveles {f.levels_completed:>2}/{f.win_levels:<2} '
                              f'acciones {ag.action_counter:>5}  {f.state.value}')
                    sc = arc.get_scorecard()
                    print(f'\\nPuntaje local (RHAE, juegos públicos): {sc.score:.2f}'
                          f'  ·  {time.time() - t0:.0f} s')
                except Exception:
                    traceback.print_exc()   # la evaluación nunca debe romper el commit
                finally:
                    os.chdir(cwd)
            """
        )
    )

    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(
            f"Unknown ACCELERATOR={ACCELERATOR!r}. Pick one of: "
            f"{sorted(_ACCELERATORS)}"
        )
    accel = _ACCELERATORS[ACCELERATOR]

    notebook = {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": accel["name"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# H7 · ARC-AGI-3 Graph Explorer\n\n"
                "Agente sin ML: cada frame es un nodo, cada acción una arista. "
                "Explora acciones no probadas, viaja por BFS a la frontera más "
                "cercana, enmascara el HUD (barras/contadores), baja de prioridad "
                "acciones sin efecto y hace click solo sobre objetos.\n\n"
                "En *Save & Run All* la última celda juega los juegos públicos de "
                "`environment_files/` y reporta niveles y puntaje local; en el "
                "rerun de la competencia solo corre la celda del gateway.\n\n"
                "Generado desde `agent/my_agent.py` con `scripts/build_notebook.py`. "
                "No edites las celdas a mano: edita el archivo fuente y vuelve a "
                "correr `make submit`."
            ),
            install_cell,
            write_agent_cell,
            run_cell,
            dummy_submission_cell,
            local_eval_cell,
        ],
    }
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1))
    print(f"[build_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)}  "
          f"(accelerator: {ACCELERATOR})")

    # Keep notebooks/kernel-metadata.json in sync so the user never has to
    # edit it just to flip CPU ↔ GPU.
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text())
        wanted = _ACCELERATORS[ACCELERATOR]["gpu"]
        if meta.get("enable_gpu") != wanted:
            meta["enable_gpu"] = wanted
            METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n")
            print(f"[build_notebook] Synced enable_gpu={wanted} in "
                  f"{METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
