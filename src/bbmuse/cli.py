import argparse
from bbmuse.engine.project import BbMuseProject

def main():
    parser = argparse.ArgumentParser(prog="bbmuse", description="BlackBoard MUSic Engine")
    parser.add_argument("dir", nargs='?', default=".", help="Path to project directory")
    args = parser.parse_args()
    
    print("CLI: Make project..")
    project = BbMuseProject(args.dir)

    print("CLI: Run..")
    project.run()
    
    print("---end of main()---")