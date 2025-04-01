
import imageio.v2 as imageio 
import os
import re
from tqdm import tqdm
import fnmatch
import numpy as np


def create_video_from_images(image_folder, output_video_path, fps=5):
    """
    Combines PNG images from a folder into a video, sorting them by numerical order.

    Args:
        image_folder (str): Path to the folder containing PNG images.
        output_video_path (str): Path to save the output video.
        fps (int): Frames per second for the video.
    """
    # 获取文件夹下所有 PNG 文件
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    # 按文件名中的数字排序
    images.sort(key=lambda f: int(re.sub(r'\D', '', f)))

    if not images:
        raise ValueError("No PNG files found in the specified folder.")

    # 读取第一张图片以获取视频帧尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = imageio.imread(first_image_path)
    height, width, _ = frame.shape

    with imageio.get_writer(output_video_path, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p') as writer:
        for image in tqdm(images, desc="Processing images", unit="frame"):
            image_path = os.path.join(image_folder, image)
            frame = imageio.imread(image_path)
            writer.append_data(frame)
    
    print(f"Video saved to {output_video_path}")
    
    
    
def create_video_from_images(image_folder, output_video_path, fps=5):
    """
    Combines PNG images from a folder into a video, sorting them by numerical order.

    Args:
        image_folder (str): Path to the folder containing PNG images.
        output_video_path (str): Path to save the output video.
        fps (int): Frames per second for the video.
    """
    # 获取文件夹下所有 PNG 文件
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    # 按文件名中的数字排序
    images.sort(key=lambda f: int(re.sub(r'\D', '', f)))

    if not images:
        raise ValueError("No PNG files found in the specified folder.")

    # 读取第一张图片以获取视频帧尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = imageio.imread(first_image_path)
    height, width, _ = frame.shape

    with imageio.get_writer(output_video_path, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p') as writer:
        for image in tqdm(images, desc="Processing images", unit="frame"):
            image_path = os.path.join(image_folder, image)
            frame = imageio.imread(image_path)
            writer.append_data(frame)
    
    print(f"Video saved to {output_video_path}")
    

def create_video_from_two_folders(init_dir, folder2, output_video_path, fps=5):
    """
    Combines PNG images from two folders into a video, sorting them by numerical order,
    and concatenates images from the two folders side by side (horizontally).

    Args:
        init_dir (str): Path to the first folder containing PNG images.
        folder2 (str): Path to the second folder containing PNG images.
        output_video_path (str): Path to save the output video.
        fps (int): Frames per second for the video.
    """
    # 获取两个文件夹的所有 PNG 文件
    
    init_pattern="cam*[0-9].png"
    images1 = [img for img in os.listdir(init_dir) if fnmatch.fnmatch(img, init_pattern)]
    images2 = [img for img in os.listdir(folder2) if img.endswith(".png")]

    # 按文件名中的数字排序
    images1.sort(key=lambda f: int(re.sub(r'\D', '', f)))
    images2.sort(key=lambda f: int(re.sub(r'\D', '', f)))

    if not images1 or not images2:
        raise ValueError("No PNG files found in one or both of the specified folders.")

    # 确保两个文件夹的图片数量一致
    if len(images1) != len(images2):
        raise ValueError("The number of images in the two folders must be the same.")

    # 读取第一张图片以获取视频帧尺寸
    first_image1_path = os.path.join(init_dir, images1[0])
    first_image2_path = os.path.join(folder2, images2[0])

    img1 = imageio.imread(first_image1_path)
    img2 = imageio.imread(first_image2_path)

    # 检查两张图片的高度是否一致，不一致需要调整
    if img1.shape[0] != img2.shape[0]:
        raise ValueError("The images in the two folders must have the same height.")

    # 创建视频写入器
    with imageio.get_writer(output_video_path, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p') as writer:
        for img1_name, img2_name in tqdm(zip(images1, images2), desc="Processing images", unit="frame", total=len(images1)):
            img1_path = os.path.join(init_dir, img1_name)
            img2_path = os.path.join(folder2, img2_name)

            img1 = imageio.imread(img1_path)
            img2 = imageio.imread(img2_path)

            # 检查图片高度是否一致，不一致需要调整
            if img1.shape[0] != img2.shape[0]:
                raise ValueError(f"Height mismatch: {img1_path} and {img2_path}")

            # 水平拼接两张图片
            concatenated_image = np.hstack((img1, img2))

            # 添加到视频
            writer.append_data(concatenated_image)

    print(f"Video saved to {output_video_path}")



    
if __name__ == "__main__":
    init_folder = "/home/quyansong/Project/GSdragger/logs/face_new/init"
    image_folder = "/home/quyansong/Project/GSdragger/logs/face_new/optim_999"
    output_video_path = "/home/quyansong/Project/testfolder/output_video.mp4" 
    fps = 5  # 帧率
    # # create_video_from_images(image_folder, output_video_path, fps)
    create_video_from_two_folders(init_folder, image_folder, output_video_path, fps)