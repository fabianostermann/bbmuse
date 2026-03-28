import logging
import sys

import argparse
from bbmuse import __version__ as prog_version

logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    args = process_args()
    
    print("Not yet implemented.")
    sys.exit(1)

def process_args():
    parser = argparse.ArgumentParser(prog="bblearn", description="Learning features for bbmuse")
    
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug messages.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Show warning and error messages. Overwrites --verbose.")
    parser.add_argument("--silent", action="store_true", help="Show no messages. Overwrites --quiet and --verbose")
    
    parser.add_argument('--version', action='version', version=f"%(prog)s {prog_version}")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    if args.silent:
        logging.getLogger().setLevel(logging.CRITICAL+1)

    logger.debug("Args: %s", args)
    
    return args
