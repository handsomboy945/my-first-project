import os
import json
import cv2
import numpy as np
import time
from multiprocessing import Pool, cpu_count
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm  # 用于显示进度条

def compute_similarity(args):
    """计算两幅图像的相似度得分"""
    test_img_path, ref_img_path = args
    try:
        # 读取并预处理图像
        test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
        ref_img = cv2.imread(ref_img_path, cv2.IMREAD_GRAYSCALE)
        
        if test_img is None or ref_img is None:
            print(f"无法读取图像: {test_img_path} 或 {ref_img_path}")
            return (test_img_path, ref_img_path, -1)
        
        # 调整图像尺寸为相同大小
        height = min(test_img.shape[0], ref_img.shape[0])
        width = min(test_img.shape[1], ref_img.shape[1])
        test_img = cv2.resize(test_img, (width, height))
        ref_img = cv2.resize(ref_img, (width, height))
        
        # 计算结构相似性指数 (SSIM)
        score, _ = ssim(test_img, ref_img, full=True)
        return (test_img_path, ref_img_path, score)
    except Exception as e:
        print(f"处理 {test_img_path} 和 {ref_img_path} 时出错: {str(e)}")
        return (test_img_path, ref_img_path, -1)

def find_best_reference(dataset_path, class_name):
    """
    为每个测试图像找到最相似的正样本参考图像（仅返回最佳匹配）
    """
    print(f"\n{'='*50}")
    print(f"开始处理类别: {class_name}")
    print(f"{'='*50}")
    
    # 1. 收集所有训练集正样本图像
    train_good_path = os.path.join(dataset_path, class_name, "train", "good")
    if not os.path.exists(train_good_path):
        print(f"错误: 训练集目录不存在 - {train_good_path}")
        return {}
    
    ref_images = []
    for f in os.listdir(train_good_path):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            ref_images.append(os.path.join(train_good_path, f))
    
    if not ref_images:
        print(f"警告: 类别 {class_name} 的训练集没有找到任何图像")
        return {}
    
    # 2. 收集所有测试图像（包括正常和异常）
    test_images = []
    test_root = os.path.join(dataset_path, class_name, "test")
    
    if not os.path.exists(test_root):
        print(f"错误: 测试集目录不存在 - {test_root}")
        return {}
    
    test_types = os.listdir(test_root)
    
    for test_type in test_types:
        test_dir = os.path.join(test_root, test_type)
        if os.path.isdir(test_dir):
            for f in os.listdir(test_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    test_images.append(os.path.join(test_dir, f))
    
    if not test_images:
        print(f"警告: 类别 {class_name} 的测试集没有找到任何图像")
        return {}
    
    print(f"测试图像数量: {len(test_images)}")
    print(f"参考图像数量: {len(ref_images)}")
    
    # 3. 创建所有可能的测试-参考对
    pairs = [(test_img, ref_img) for test_img in test_images for ref_img in ref_images]
    print(f"总比较对数: {len(pairs)}")
    
    # 4. 使用多进程并行计算相似度
    results = {}
    start_time = time.time()
    
    # 使用多进程池并行计算
    with Pool(processes=cpu_count()) as pool:
        # 使用tqdm显示进度条
        results_list = list(tqdm(pool.imap(compute_similarity, pairs), total=len(pairs)))
    
    # 整理结果
    for test_path, ref_path, score in results_list:
        if test_path not in results:
            results[test_path] = []
        results[test_path].append((ref_path, score))
    
    # 5. 为每个测试图像找到最高分的参考图像（仅返回最佳匹配）
    aligned_ref_paths = {}
    for test_path, ref_scores in results.items():
        # 按相似度排序（从高到低）
        ref_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 只选择最相似的参考图像
        if ref_scores and ref_scores[0][1] > 0:  # 确保有有效结果
            aligned_ref_paths[test_path] = [ref_scores[0][0]]
    
    elapsed_time = time.time() - start_time
    print(f"完成 {class_name} 类的相似度计算，耗时: {elapsed_time:.2f}秒")
    print(f"成功匹配 {len(aligned_ref_paths)}/{len(test_images)} 个测试图像")
    
    return aligned_ref_paths

def generate_all_classes(dataset_path, output_dir, overwrite=False):
    """
    为所有类别生成对齐映射文件
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有类别
    classes = [d for d in os.listdir(dataset_path) 
              if os.path.isdir(os.path.join(dataset_path, d))]
    
    print(f"找到 {len(classes)} 个类别: {', '.join(classes)}")
    
    for class_name in classes:
        print(f"\n{'='*50}")
        print(f"开始处理类别: {class_name}")
        print(f"{'='*50}")
        
        # 检查输出文件路径
        output_path = os.path.join(output_dir, f"{class_name}.json")
        
        # 如果文件已存在且不覆盖，则跳过
        if os.path.exists(output_path) and not overwrite:
            print(f"文件已存在: {output_path}，跳过（使用 overwrite=True 强制重新生成）")
            continue
        
        # 生成该类别的对齐映射
        aligned_ref_paths = find_best_reference(dataset_path, class_name)
        
        if not aligned_ref_paths:
            print(f"警告: 未生成 {class_name} 类的映射，跳过保存")
            continue
        
        # 转换为相对路径
        relative_ref_paths = {}
        for test_path, ref_paths in aligned_ref_paths.items():
            # 获取相对于数据集根目录的相对路径
            rel_test_path = os.path.relpath(test_path, dataset_path)
            rel_ref_paths = [os.path.relpath(p, dataset_path) for p in ref_paths]
            
            # 统一使用正斜杠路径分隔符
            rel_test_path = rel_test_path.replace('\\', '/')
            rel_ref_paths = [p.replace('\\', '/') for p in rel_ref_paths]
            
            relative_ref_paths[rel_test_path] = rel_ref_paths
        
        # 保存为JSON文件
        with open(output_path, 'w') as f:
            json.dump(relative_ref_paths, f, indent=4)
        
        print(f"已保存 {class_name} 的对齐映射到 {output_path}")
        print(f"映射包含 {len(relative_ref_paths)} 个测试图像")

if __name__ == "__main__":
    # 配置路径
    dataset_path = "./dataset/mvtec_loco_anomaly_detection"  # MVTec数据集根目录
    output_dir = "./LLMtry/annotations/mvtec"     # 输出目录
    
    # 重新生成所有类别，每个测试图像只保留最佳匹配
    generate_all_classes(
        dataset_path=dataset_path,
        output_dir=output_dir,
        overwrite=True  # 强制覆盖现有文件
    )
    
    print("\n所有类别处理完成！")