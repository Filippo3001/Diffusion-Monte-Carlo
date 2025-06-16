import os
import sys

print(f'PYTHONPATH:')
for string in os.environ.get('PYTHONPATH'):
    print (string)

print(f'sys.path:')
for string in sys.path:
    print(string)