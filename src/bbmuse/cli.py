import logging
import sys

import argparse
from bbmuse import __version__ as prog_version

from bbmuse.engine.project import BbMuseProject
from bbmuse.util.visualization import plot_dependency_graph

logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    args = process_args()

    if args.editor:
        start_editor(args)
    else:
        start_headless(args)    

def start_headless(args):
    logger.info("Starting in headless mode.")

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

    if args.verify_build:
        logger.info("Build ended without errors.")
        plot_dependency_graph(project)
    else:
        logger.info("Run project..")
        try:
            project.run(quit_after=args.quit_after, run_mode=args.mode)
        except Exception:
            logger.exception("Failure while running project.")
            sys.exit(1)
    
    logger.info("bbmuse exited normally from headless mode.")

def start_editor(args):
    logger.info("Starting gui editor..")
    try:
        import bbmuse.editor
    except Exception:
        logger.error("GUI is not implemented yet.")

def process_args():
    parser = argparse.ArgumentParser(prog="bbmuse", description="BlackBoard MUSic Engine")
    parser.add_argument("dir", nargs='?', default=None, type=str, help="Path to desired working directory. If none is given, editor opens.")
    parser.add_argument("-e", "--editor", action="store_true", help="Start the editor instead of running the project.")
    #parser.add_argument("-p", "--project-manager", action="store_true", help="Start the Project Manager, even if a project is auto-detected.")

    parser.add_argument("--mode", default="NORMAL", type=str.upper, choices=['DEBUG', 'NORMAL', 'PERFORM'])

    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug messages.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Show warning and error messages. Overwrites --verbose.")
    parser.add_argument("--silent", action="store_true", help="Show no messages. Overwrites --quiet and --verbose")

    parser.add_argument("--verify-build", action="store_true", help="Verify if project can be build without errors and creates a plot of the dependency graph. Will not run afterwards.")
    parser.add_argument("--quit-after", type=int, default=-1, help="Quit after the given time in seconds.")
    
    parser.add_argument('--version', action='version', version=f"%(prog)s {prog_version}")
    args = parser.parse_args()

    # if no directory is provided, enable gui mode
    if args.dir is None:
        args.editor = True

    match args.mode:
        case "DEBUG": args.mode = -1
        case "NORMAL": args.mode = 0
        case "PERFORM": args.mode = 1

    if args.verbose or args.mode < 0:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    if args.silent:
        logging.getLogger().setLevel(logging.CRITICAL+1)
        sys.stdout = None

    logger.debug("Args: %s", args)

    return args
