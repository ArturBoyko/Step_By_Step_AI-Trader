# bin/make_dataset.py
import sys, os
CUR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(CUR), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dataset_builder import main

if __name__ == "__main__":
    main()
