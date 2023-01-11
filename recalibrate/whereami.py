import os
from pathlib import Path
from pprint import pprint
TOP = os.path.dirname(os.path.abspath(__file__))
ROOT = Path(TOP).parent.absolute()


if __name__=='__main__':
    print(ROOT)