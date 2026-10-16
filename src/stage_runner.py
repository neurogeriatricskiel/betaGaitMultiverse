"""Subject loop shared by the per-subject reference stages (prepana02-05)."""

import argparse
import traceback
from collections.abc import Callable

from src.config import SUBJECTS


def subjects_from_command_line(description: str) -> list[str]:
    """Parse an optional single-subject argument; default: every configured subject.

    Parameters
    ----------
    description : str
        Help text of the stage script.

    Returns
    -------
    list of str
        Subjects to process.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("subject", nargs="?", help="process only this subject")
    args = parser.parse_args()
    return [args.subject] if args.subject else SUBJECTS


def run_for_subjects(
    subjects: list[str],
    process: Callable[[str], None],
    reraise: tuple[type[BaseException], ...] = (),
) -> None:
    """Call ``process(subject)`` for each subject; report and skip failures.

    Parameters
    ----------
    subjects : list of str
        Subjects to process.
    process : callable
        Per-subject stage function.
    reraise : tuple of exception types
        Exceptions that stop the run instead of skipping the subject.
    """
    for subject in subjects:
        print(f"\nProcessing sub-{subject}")
        try:
            process(subject)
        except reraise:
            raise
        except FileNotFoundError as e:
            print(f"\n  [SKIP] sub-{subject}: file not found -- {e}")
        except Exception as e:
            print(f"\n  [ERROR] sub-{subject}: unexpected error -- {e}")
            traceback.print_exc()
    print(f"\nDone. Processed {len(subjects)} subject(s): {subjects}")
