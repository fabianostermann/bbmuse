import logging
import sys

import argparse
from bbmuse.engine.project import BbMuseProject

logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(prog="bbmuse", description="BlackBoard MUSic Engine")
    parser.add_argument("dir", nargs='?', default=".", help="Path to project directory")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    logger.info("Init project..")
    try:
        project = BbMuseProject(args.dir)
    except Exception:
        logger.exception("Making project failed.")
        sys.exit(1)

    logger.info("Build project..")
    try:
        project.build()
    except Exception:
        logger.exception("Building project failed.")
        sys.exit(1)

    logger.info("Run project..")
    try:
        project.run(limit=args.limit)
    except Exception:
        logger.exception("Failure while running project.")
        sys.exit(1)
    
    logger.info("Bbmuse exits normally.")
