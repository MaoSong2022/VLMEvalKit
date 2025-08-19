import os
import argparse
import re

file_path = "/cpfs04/user/maosong/VLMEvalKit/vlmeval/vlm/eagle/model/multimodal_encoder/multi_backbone_channel_concatenation_encoder.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mask_index", type=int, nargs="*", default=[0, 2, 3])
    args = parser.parse_args()

    with open(file_path, "r") as f:
        content = f.read()

    pattern = re.compile(r"self.mask_vision_tower_index = \[(?:[0-9, ]*)\]")
    content = re.sub(
        pattern, f"self.mask_vision_tower_index = {args.mask_index}", content
    )

    with open(file_path, "w") as f:
        f.write(content)


if __name__ == "__main__":
    main()
