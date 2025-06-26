import logging
import sys

from enum import Enum

import argparse
from bbmuse.engine.project import BbMuseProject

logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    args = process_args()

    if args.gui:
        main_gui(args)
    else:
        main_headless(args)    

def main_headless(args):

    logger.info("Init project..")
    try:
        project = BbMuseProject(args.dir)
    except Exception:
        logger.exception("Init project failed.")
        sys.exit(1)

    logger.info("Build project..")
    try:
        project.build_all()
    except Exception:
        logger.exception("Building project failed.")
        sys.exit(1)

    logger.info("Run project..")
    try:
        project.run(quit_after=args.quit_after)
    except Exception:
        logger.exception("Failure while running project.")
        sys.exit(1)
    
    logger.info("bbmuse exits normally.")

def main_gui(args):
    try:
        import bbmuse.gui
    except Exception:
        logger.error("GUI mode is not implemented yet.")

def process_args():
    parser = argparse.ArgumentParser(prog="bbmuse", description="BlackBoard MUSic Engine")
    parser.add_argument("dir", nargs='?', default=None, type=str, help="Path to desired working directory. If none is given, editor opens.")
    parser.add_argument("-e", "--gui", action="store_true", help="Start the editor instead of running the project.")
    #parser.add_argument("-p", "--project-manager", action="store_true", help="Start the Project Manager, even if a project is auto-detected.")

    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug messages.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Show warning and error messages. Overwrites --verbose.")
    parser.add_argument("--silent", action="store_true", help="Show no messages. Overwrites --quiet and --verbose")

    parser.add_argument("--quit-after", type=int, default=0, help="Quit after the given number of iterations. Set to 0 to disable.")
    args = parser.parse_args()

    # if no directory is provided, enable gui mode
    if args.dir is None:
        args.gui = True

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    if args.silent:
        logging.getLogger().setLevel(logging.CRITICAL+1)

    return args