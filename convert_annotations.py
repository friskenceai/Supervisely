import tarfile
import os
import cv2
import json
import shutil
import sys

def extract_tar(tar_path, output_dir, delete_tar=False):
    os.makedirs(output_dir, exist_ok=True)
    with tarfile.open(tar_path, 'r') as tar:
        print(f"Extracting '{tar_path}' to '{output_dir}'.")
        tar.extractall(path=output_dir)
    if delete_tar:
        print(f"Deleting tar file: '{tar_path}'.")
        os.remove(tar_path)
    return output_dir

def list_folders(directory):
    return [os.path.join(directory, item) for item in os.listdir(directory) 
            if os.path.isdir(os.path.join(directory, item))]

def find_video_and_annotation_paths(folder):
    video_path = os.path.join(folder, 'video')
    annotation_path = os.path.join(folder, 'ann')

    video_file_path = next((os.path.join(video_path, f) for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))), None)
    annotation_file_path = next((os.path.join(annotation_path, f) for f in os.listdir(annotation_path) if os.path.isfile(os.path.join(annotation_path, f))), None)

    return video_file_path, annotation_file_path

def convert_to_yolo(exterior, img_width, img_height):
    x_min, y_min = exterior[0]
    x_max, y_max = exterior[1]
    center_x = (x_min + x_max) / 2 / img_width
    center_y = (y_min + y_max) / 2 / img_height
    bbox_width = (x_max - x_min) / img_width
    bbox_height = (y_max - y_min) / img_height
    return center_x, center_y, bbox_width, bbox_height

def create_yolo_labels(video_path, annotation_path, output_base_path):
    os.makedirs(os.path.join(output_base_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_base_path, 'labels'), exist_ok=True)

    with open(annotation_path) as f:
        annotations = json.load(f)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = annotations['size']['height']
    width = annotations['size']['width']

    class_mapping = {obj['classTitle']: idx for idx, obj in enumerate(annotations['objects'])}

    for frame in annotations['frames']:
        frame_index = frame['index']
        if frame_index < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, image = cap.read()
            if ret:
                frame_path = os.path.join(output_base_path, 'images', f"frame_{frame_index:04d}.jpg")
                cv2.imwrite(frame_path, image)

                label_path = os.path.join(output_base_path, 'labels', f"frame_{frame_index:04d}.txt")
                with open(label_path, 'w') as label_file:
                    for figure in frame['figures']:
                        geometry = figure['geometry']['points']['exterior']
                        center_x, center_y, bbox_width, bbox_height = convert_to_yolo(geometry, width, height)

                        object_key = figure['objectKey']
                        class_id = class_mapping.get(next(obj['classTitle'] for obj in annotations['objects'] if obj['key'] == object_key), -1)
                        if class_id != -1:
                            label_file.write(f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}\n")

    cap.release()

def compress_folder(folder_path):
    print(f"Compressing folder '{folder_path}' to '{folder_path}.zip'.")
    shutil.make_archive(folder_path, 'zip', folder_path)

def main(tar_path, output_dir, separate_folders=False, delete_extracted=False, delete_yolo_annotation=False):
    extracted_dir = extract_tar(tar_path, output_dir)

    croot = os.path.join(extracted_dir, os.listdir(extracted_dir)[0])

    folder_paths = list_folders(croot)

    total_videos = len(folder_paths)
    for idx, folder in enumerate(folder_paths, start=1):
        video_file_path, annotation_file_path = find_video_and_annotation_paths(folder)

        if video_file_path and annotation_file_path:
            print(f"Processing Video {idx} of {total_videos}")

            # Determine output path
            if separate_folders:
                folder_name = os.path.basename(folder)
                output_base_path = os.path.join('yolo_annotation', folder_name)
            else:
                output_base_path = 'yolo_annotation'

            create_yolo_labels(video_file_path, annotation_file_path, output_base_path)

    # Compress the yolo annotation folder
    compress_folder('yolo_annotation')

    # Cleanup
    if delete_extracted:
        print(f"Deleting extracted folder: '{extracted_dir}'.")
        shutil.rmtree(extracted_dir)

    if delete_yolo_annotation:
        print(f"Deleting yolo_annotation folder after compression.")
        shutil.rmtree('yolo_annotation')

# Example usage from command line
if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python your_script.py <tar_path> <output_dir> <separate_folders> <delete_extracted> <delete_yolo_annotation>")
    else:
        tar_path = sys.argv[1]
        output_dir = sys.argv[2]
        separate_folders = sys.argv[3].lower() == 'true'  # Convert to boolean
        delete_extracted = sys.argv[4].lower() == 'true'  # Convert to boolean
        delete_yolo_annotation = sys.argv[5].lower() == 'true'  # Convert to boolean
        main(tar_path, output_dir, separate_folders, delete_extracted, delete_yolo_annotation)
