import os 
import json
import argparse


CONDITION = ["uncover", "cover1", "cover2"]

def parse_argument():
    """
    Parse arguments from command line

    Returns
    -------
    ArgumentParser
        Object for argument parser

    """
    parser = argparse.ArgumentParser("ABC")
    parser.add_argument('-m', '--mapping-file', type=str, 
                        help='Path to input data folder', default="val_SLP_path.json")
    parser.add_argument('-i', '--input-file', type=str, 
                        help='Path to output data folder', default="../data/coco/annotations/person_keypoints_slp_val.json")
    parser.add_argument('-o', '--output-folder', type=str, 
                        help='Path to output data folder', default="../data/coco/annotations/")
    return parser

def main():
    parser = parse_argument()
    args = parser.parse_args()
    mapping_file = args.mapping_file
    input_file = args.input_file
    output_folder = args.output_folder

    mapping = json.load(open(mapping_file))
    try:
        annotations = json.load(open(input_file))['annotations']
    except:
        annotations = json.load(open(input_file))

    uncover = {}
    uncover['annotations'] = list()

    cover1 = {}
    cover1['annotations'] = list()

    cover2 = {}
    cover2['annotations'] = list()

    for anno in annotations:
        image_id = anno['image_id']
        for k in mapping.keys():
            path = mapping[k]
            # print(path)
            _image_id = int(path.split("/")[-1].replace(".png", ""))
            # print(_image_id)
            if image_id == _image_id:
                con = k.split("/")[7]
                if con == "uncover":
                    uncover['annotations'].append(anno)
                    break
                elif con == "cover1":
                    cover1['annotations'].append(anno)
                    break
                elif con == "cover2":
                    cover2['annotations'].append(anno)
                    break
    with open(os.path.join(output_folder, "uncover_val.json"), "w") as f:
        json.dump(uncover, f)
    with open(os.path.join(output_folder, "cover1_val.json"), "w") as f:
        json.dump(cover1, f)
    with open(os.path.join(output_folder, "cover2_val.json"), "w") as f:
        json.dump(cover2, f)

if __name__ == "__main__":
    main()