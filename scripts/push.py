from belirtec.config import load_config
from belirtec.push import push_all

if __name__ == "__main__":
    push_all(load_config())
