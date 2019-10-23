import os
from pathlib import Path

os.environ['CODE_DIR'] = os.environ.get(
    'PT_CODE_DIR', str((Path(__file__).parent / '..').resolve()))
os.environ['DATA_DIR'] = os.environ.get(
    'PT_DATA_DIR', str((Path(__file__).parent / '..' / 'data').resolve()))

