import logging
import sys

import argparse
from bbmuse import __version__ as prog_version

from bbmuse.engine.project import BbMuseProject
from bbmuse.learn.session import Session

logging.basicConfig(format="%(levelname)s\t%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    args = process_args()
    
    try:
        project = BbMuseProject(".")
    except Exception:
        logger.exception("Init project failed. 'bblearn' is supposed to be used at the root of a valid bbmuse project.")
        sys.exit(1)
    
    session = Session(project, args)

def process_args():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", help="Show debug messages.")
    common.add_argument("--silent", action="store_true", help="Show no messages. Overwrites --verbose")
    
    parser = argparse.ArgumentParser(prog="bblearn", description="Learning features for bbmuse", parents=[common])
    
    parser.add_argument('--version', action='version', version=f"%(prog)s {prog_version}")
    
    subparsers = parser.add_subparsers(dest="command")
    
    sub_enable = subparsers.add_parser("arm", help='Arm modules (arming means to enable recording mode for the specified modules)', parents=[common])
    sub_enable.add_argument('modules', nargs="*", help="Path or name of modules")
    
    sub_disable = subparsers.add_parser("disarm", help='Disarm modules (disarming means to disable recording mode for the specified modules)', parents=[common])
    sub_disable.add_argument('modules', nargs="*", help="Path or name of modules")
    
    sub_status = subparsers.add_parser("status", help='TODO write help', parents=[common])
    sub_status.add_argument('modules', nargs='*', help="Path or name of modules. If none is given, prints a summary of all modules.")
    
    sub_listen = subparsers.add_parser("listen", help='Starts bbmuse and collects data of armed modules.', parents=[common])
    
    sub_clone = subparsers.add_parser("clone", help='Train a model to mimic a specific module based on previously collected data.', parents=[common])
    sub_clone.add_argument('module', nargs=1, help="Path or name of a module")
    sub_clone.add_argument("--backbone", default=None, type=str, help="Path to a backbone py file")

    sub_sculpt = subparsers.add_parser("sculpt", help='Refine a trained model based on heuristic constraints and human feedback.', parents=[common])
    #sub_sculpt.add_argument('module', nargs=1, help="Path or name of a module")
    
    sub_apply = subparsers.add_parser("apply", help='TODO write help', parents=[common])
    sub_apply.add_argument('module', nargs=1, help="Path or name of a module")
    
    sub_restore = subparsers.add_parser("restore", help='TODO write help', parents=[common])
    sub_restore.add_argument('module', nargs=1, help="Path or name of a module")
    
    #sub_restore = subparsers.add_parser("help")
    
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=logging.DEBUG, force=True)
    if args.silent:
        logging.getLogger().setLevel(logging.CRITICAL+1)
        sys.stdout = None
        sys.stderr = None

    logger.debug("Args: %s", args)
    
    if args.command is None: # or args.command == "help":
        parser.parse_args(["--help"])
        sys.exit(0)
    
    return args
