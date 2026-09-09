import json
import os

annotation_file = r'd:/dlpro/archive (1)/vinbigdata-cxr-ad-coco/annotations/instances_train.json'

if not os.path.exists(annotation_file):
    print(f"File not found: {annotation_file}")
    exit(1)

with open(annotation_file, 'r') as f:
    data = json.load(f)

print("Keys:", data.keys())
if 'categories' in data:
    print("Categories:")
    for cat in data['categories']:
        print(f"  {cat['id']}: {cat['name']}")

print(f"Number of images: {len(data['images'])}")
print(f"Number of annotations: {len(data['annotations'])}")
