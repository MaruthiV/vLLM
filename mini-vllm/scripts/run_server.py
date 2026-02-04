import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from mini_vllm.entrypoints.openai.api_server import main

if __name__ == "__main__":
    main()
