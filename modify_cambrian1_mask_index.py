import os
import argparse
import re

# file_path = "/cpfs04/user/maosong/VLMEvalKit/vlmeval/vlm/cambrian/model/language_model/cambrian_phi3.py"
file_path = "/cpfs04/user/maosong/VLMEvalKit/vlmeval/vlm/cambrian/model/language_model/cambrian_llama.py"
# file_path = "/cpfs04/user/maosong/VLMEvalKit/backup.py"
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mask_index", type=int, nargs="+", default=[0, 2, 3])
    args = parser.parse_args()

    with open(file_path, "r") as f:
        content = f.read()

    pattern = re.compile(r"self.masked_index = \[[0-9, ]+\]")
    content = re.sub(pattern, f"self.masked_index = {args.mask_index}", content)
    
    with open(file_path, "w") as f:
        f.write(content)


if __name__ == "__main__":
    main()