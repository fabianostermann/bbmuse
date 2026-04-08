import logging
import sys

import argparse
from bbmuse import __version__ as prog_version

from bbmuse.engine.project import BbMuseProject
from bbmuse.learn.session import Session

logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=logging.INFO)
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
    common.add_argument("-q", "--quiet", action="store_true", help="Show warning and error messages. Overwrites --verbose.")
    common.add_argument("--silent", action="store_true", help="Show no messages. Overwrites --quiet and --verbose")
    
    parser = argparse.ArgumentParser(prog="bblearn", description="Learning features for bbmuse", parents=[common])
    
    parser.add_argument('--version', action='version', version=f"%(prog)s {prog_version}")
    
    subparsers = parser.add_subparsers(dest="command")
    
    sub_enable = subparsers.add_parser("enable", help='TODO write help', parents=[common])
    sub_enable.add_argument('var', nargs=1, type=str, help="The variable name to set")
    sub_enable.add_argument('module_path', nargs=1, help="Path to module")
    
    sub_disable = subparsers.add_parser("disable", help='TODO write help', parents=[common])
    sub_disable.add_argument('var', type=str, nargs=1, help="The variable name to set")
    sub_disable.add_argument('module_path', nargs=1, help="Path to module")
    
    sub_status = subparsers.add_parser("status", help='TODO write help', parents=[common])
    sub_status.add_argument('module_path', nargs='?', help="Path to module")
    
    sub_run = subparsers.add_parser("run", help='TODO write help', parents=[common])
    
    sub_train = subparsers.add_parser("train", help='TODO write help', parents=[common])
    sub_train.add_argument('module_path', nargs=1, help="Path to module")
    
    sub_apply = subparsers.add_parser("apply", help='TODO write help', parents=[common])
    sub_apply.add_argument('module_path', nargs=1, help="Path to module")
    
    sub_restore = subparsers.add_parser("restore", help='TODO write help', parents=[common])
    sub_restore.add_argument('module_path', nargs=1, help="Path to module")
    
    #sub_restore = subparsers.add_parser("help")
    
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    if args.silent:
        logging.getLogger().setLevel(logging.CRITICAL+1)
        sys.stdout = None

    logger.debug("Args: %s", args)
    
    if args.command is None: # or args.command == "help":
        parser.parse_args(["--help"])
        sys.exit(0)
    
    return args
