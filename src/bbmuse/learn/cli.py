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
        logging.shutdown()
        sys.exit(1)
    
    try:
        session = Session(project, args)
    except Exception:
        logger.exception("The bblearn session failed with exception.")
        logging.shutdown()
        sys.exit(1)

    logging.shutdown()
    sys.exit(0)

def process_args():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", help="Show debug messages.")
    common.add_argument("--silent", action="store_true", help="Show no messages. Overwrites --verbose")
    
    parser = argparse.ArgumentParser(prog="bblearn", description="Learning features for bbmuse", parents=[common])
    
    parser.add_argument('--version', action='version', version=f"%(prog)s {prog_version}")
    
    subparsers = parser.add_subparsers(dest="command")
    
    sub_enable = subparsers.add_parser("arm", help='To arm a module means it will be recorded during listening.', parents=[common])
    sub_enable.add_argument('modules', nargs="*", help="Path or name of modules")
    
    sub_disable = subparsers.add_parser("disarm", help='To disarm a module means it will NOT be recorded during listening.', parents=[common])
    sub_disable.add_argument('modules', nargs="*", help="Path or name of modules")
    
    sub_status = subparsers.add_parser("status", help='Prints a summary of available modules, records, and trained models.', parents=[common])
    sub_status.add_argument('modules', nargs='*', help="Path or name of modules. If none is given, prints a summary of all modules.")
    sub_status.add_argument('-s', "--short", action="store_true", help="Give the output in the short-format.")
    
    sub_listen = subparsers.add_parser("listen", help='Starts bbmuse and collects data of armed modules.', parents=[common])
    sub_listen.add_argument('--dry-run', action="store_true", help="Do not write to disk.")
    sub_listen.add_argument("--quit-after", type=float, default=-1, help="Quit after the given time in seconds.")
    sub_listen.add_argument("--tag", type=str, default=None, help="A string tag that is appended to the filepath.")

    sub_clone = subparsers.add_parser("clone", help='Train a model to mimic a specific module based on previously collected data.', parents=[common])
    sub_clone.add_argument('module', nargs=1, help="Path or name of a module")
    sub_clone.add_argument("--backbone", default=None, type=str, help="Path to a backbone py file")
    sub_clone.add_argument("--device", default=None, type=str, help="Torch device to use (e.g. 'cuda' or 'cpu')")
    sub_clone.add_argument('--dry-run',action="store_true", help="Do not write to disk.")
    sub_clone.add_argument("--epochs", type=int, default=20, help="Number of epochs to train.")
    sub_clone.add_argument("--tag", type=str, default=None, help="A string tag that is appended to the filepath.")

    # sub_sculpt = subparsers.add_parser("sculpt", help='Refine one model based on heuristic constraints and human feedback.', parents=[common])
    # sub_sculpt.add_argument('module', nargs=1, help="Path or name of a module")
    # sub_sculpt.add_argument("--device", default=None, type=str, help="Torch device to use (e.g. 'cuda' or 'cpu')")
    # sub_sculpt.add_argument('--dry-run',action="store_true", help="Do not write to disk.")
    # sub_sculpt.add_argument("--tag", type=str, default=None, help="A string tag that is appended to the filepath.")

    # sub_sculpt_rr = subparsers.add_parser("sculpt-rr", help='Run sculpt on multiple models in round robin mode.', parents=[common])
    # sub_sculpt_rr.add_argument('modules', nargs='*', help="Path or name of modules.")
    # sub_sculpt_rr.add_argument("--device", default=None, type=str, help="Torch device to use (e.g. 'cuda' or 'cpu')")
    # sub_sculpt_rr.add_argument('--dry-run',action="store_true", help="Do not write to disk.")
    # sub_sculpt_rr.add_argument("--tag", type=str, default=None, help="A string tag that is appended to the filepath.")
    # sub_sculpt_rr.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override a run() parameter, e.g. --set bc_coef=0.3")

    sub_sculpt_sim = subparsers.add_parser("sculpt-sim", help='Run sculpt on multiple models in simultaneous mode.', parents=[common])
    sub_sculpt_sim.add_argument('modules', nargs='*', help="Path or name of modules.")
    sub_sculpt_sim.add_argument("--device", default=None, type=str, help="Torch device to use (e.g. 'cuda' or 'cpu')")
    sub_sculpt_sim.add_argument('--dry-run',action="store_true", help="Do not write to disk.")
    sub_sculpt_sim.add_argument("--tag", type=str, default=None, help="A string tag that is appended to the filepath.")
    sub_sculpt_sim.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override a run() parameter, e.g. --set bc_coef=0.3")
    sub_sculpt_sim.add_argument("--clone", type=str, default=None, help="Determine the clone to load by complete path name, id, or tag.")

    sub_apply = subparsers.add_parser("apply", help='Construct a BbMuse module file that wraps a neural model but runs as native BbMuse module.', parents=[common])
    sub_apply.add_argument('module', nargs=1, help="Path or name of a module")
    sub_apply.add_argument('--list', action='store_true', help="List all available models")
    sub_apply.add_argument('--clone', metavar='<id>', type=str, help="Use clone model with specified ID")
    sub_apply.add_argument('--sculpt', metavar='<id>', type=str, help="Use sculpt model with specified ID")

    sub_restore = subparsers.add_parser("restore", help='Restore the original module file from an applied one.', parents=[common])
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
