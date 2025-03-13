from pycocotools.coco import COCO
import json
import argparse

def merge_coco_annotations(json_files, output_file):
    merged_images = []
    merged_annotations = []
    
    # Initialize max IDs
    max_img_id = 0
    max_ann_id = 0
    
    for annotation_file in json_files:
        with open(annotation_file, 'r') as f:
            coco = json.load(f)
            
            # Find the maximum image ID in the dataset
            max_img_id_in_dataset = max([img['id'] for img in coco['images']])
            
            # Update image IDs in the dataset
            for img in coco['images']:
                img['id'] += max_img_id
                
            # Update annotation IDs and image IDs in the dataset
            for ann in coco['annotations']:
                ann['id'] += max_ann_id
                ann['image_id'] += max_img_id
            
            # Update max IDs for the next dataset
            max_img_id += max_img_id_in_dataset + 1
            
            # Adjust max_ann_id if needed
            max_ann_id = max(max_ann_id, max([ann['id'] for ann in coco['annotations']])) + 1
            
            # Add images and annotations to merged lists
            merged_images.extend(coco['images'])
            merged_annotations.extend(coco['annotations'])
    
    # Merge annotations
    coco_merged = {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": coco["categories"]  # Assuming categories are the same for all datasets
    }
    
    # Save the merged annotations to a new file
    with open(output_file, 'w') as outfile:
        json.dump(coco_merged, outfile)

# Path: tools/merge_coco.py
# Get arguments from command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge COCO JSON files")
    parser.add_argument("--ann-paths", nargs="+", help="List of paths to COCO JSON files to merge", required=True)
    parser.add_argument("--output-path", help="Output file path for merged annotations", required=True)
    args = parser.parse_args()

    # Merge COCO JSON files
    merge_coco_annotations(args.json_files, args.output_file)
    print("Merged COCO JSON files saved to", args.output_file)