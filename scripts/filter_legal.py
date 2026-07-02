from belirtec.config import load_config
from belirtec.contamination import filter_legal

if __name__ == "__main__":
    filter_legal(load_config())
