import os
import json
import base64
import time
import re
from PIL import Image
import dashscope
from http import HTTPStatus
import sys
import numpy as np
import cv2
import random
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

#全局配置
os.environ["DASHSCOPE_API_KEY"] = "sk-1e6d5dd3c6a94151ab15f67cd0b281a8"
BASE_DIR = "./LLMtry"
DATASET_DIR = "./dataset/mvtec_loco_anomaly_detection" 
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations", "mvtec")
TEST_IMAGES_DIR = os.path.join(DATASET_DIR, "{class_name}", "test")

#MVTEC类别（不同数据集需要对应修改）
SELECTED_CLASSES = [d for d in os.listdir(DATASET_DIR) 
              if os.path.isdir(os.path.join(DATASET_DIR, d))]
SELECTED_CLASSES = SELECTED_CLASSES

CLASSES_DIR = [os.path.join(DATASET_DIR, d) for d in os.listdir(DATASET_DIR) 
              if os.path.isdir(os.path.join(DATASET_DIR, d))]

anomaly_types = ['logical_anomalies', 'structural_anomalies']

def encode_image(image_path):
    """将图像编码为Base64字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# def extract_response(text):
#     """从模型响应中提取<think>和<answer>部分"""
#     try:
        
#         think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
#         answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        
#         think_content = think_match.group(1).strip() if think_match else "未提取到推理过程"
#         answer_content = answer_match.group(1).strip() if answer_match else "未提取到答案"
        
        
#         is_anomaly = "yes" in answer_content.lower()
#         bboxes = []
        
#         if is_anomaly:
            
#             bbox_matches = re.findall(r'\{\s*"bbox_2d"\s*:\s*\[([\d\.,\s]+)\]\s*,\s*"label"\s*:\s*"([^"]+)"\s*\}', answer_content)
#             for match in bbox_matches:
#                 coords = [float(x.strip()) for x in match[0].split(',')]
#                 bboxes.append({
#                     "bbox_2d": coords,
#                     "label": match[1]
#                 })
        
#         return {
#             "think": think_content,
#             "answer": answer_content,
#             "is_anomaly": is_anomaly,
#             "bboxes": bboxes
#         }
#     except Exception as e:
#         print(f"解析响应出错: {str(e)}")
#         return {
#             "think": "解析失败",
#             "answer": "解析失败",
#             "is_anomaly": False,
#             "bboxes": []
#         }

def extract_response(text):
    """从模型响应中提取<think>和<answer>部分，并支持四级置信度"""
    try:
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        
        think_content = think_match.group(1).strip() if think_match else "未提取到推理过程"
        answer_content = answer_match.group(1).strip() if answer_match else "未提取到答案"
        
        answer_lower = answer_content.lower()
        if "yes" in answer_lower:
            anomaly_level = "yes"
        elif "possible" in answer_lower:
            anomaly_level = "possible"
        elif "uncertain" in answer_lower:
            anomaly_level = "uncertain"
        else:
            anomaly_level = "no"
        
        bboxes = []
        if anomaly_level in ["yes", "possible", "uncertain"]:
            bbox_matches = re.findall(r'\{\s*"bbox_2d"\s*:\s*\[([\d\.,\s]+)\]\s*,\s*"label"\s*:\s*"([^"]+)"\s*\}', answer_content)
            for match in bbox_matches:
                coords = [float(x.strip()) for x in match[0].split(',')]
                bboxes.append({
                    "bbox_2d": coords,
                    "label": match[1]
                })
        is_anomaly = anomaly_level == "yes"
        return {
            "think": think_content,
            "answer": answer_content,
            "is_anomaly": is_anomaly,  
            "bboxes": bboxes
        }
    except Exception as e:
        print(f"解析响应出错: {str(e)}")
        return {
            "think": "解析失败",
            "answer": "解析失败",
            "is_anomaly": "no",
            "bboxes": []
        }

def get_text(base64_ref_image, 
            base64_image, 
            prompt):
    """使用DashScope API获取模型响应"""
    completed = False
    max_retries = 5
    retry_count = 0
    
    while not completed and retry_count < max_retries:
        time.sleep(3)  
        try:
            response = dashscope.MultiModalConversation.call(
                model='qwen-vl-plus',
                messages=[
                    {
                        "role": "system",
                        "content":f"You are a professional industrial image inspector, particularly skilled at identifying abnormal (defective) areas in images. \
                        The following product may occur these anomalies (defects): {anomaly_types}. You must accurately identify these anomalies and strictly \
                        avoid treating normal differences that are not enough to form defects as anomalies."

                    },
                    {
                        "role": "user",
                        "content": [
                            {"image": f"data:image/png;base64,{base64_ref_image}"},
                            {"image": f"data:image/png;base64,{base64_image}"},
                            {"text": prompt}
                        ]
                    }
                ],
                top_p=0.95
            )
            
            if response.status_code == HTTPStatus.OK:
                
                text_content = ""
                for content in response.output.choices[0].message.content:
                    if 'text' in content:
                        text_content += content['text']
                return text_content
            else:
                print(f"API请求失败: code={response.code}, message={response.message}")
                retry_count += 1
                print(f"重试中 ({retry_count}/{max_retries})...")
        except Exception as e:
            print(f"API调用异常: {str(e)}")
            retry_count += 1
            print(f"重试中 ({retry_count}/{max_retries})...")
    
    print(f"达到最大重试次数 ({max_retries})，跳过当前图像")
    return "API请求失败"

def resize_image(image_path, target_size=448):
    """智能调整图像大小，保持宽高比"""
    try:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        if width > height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"图像调整大小失败: {str(e)}")
        return None

def calculate_metrics(true_labels, pred_labels):
    """计算评估指标"""
    n = min(len(true_labels), len(pred_labels))
    true_labels = true_labels[:n]
    pred_labels = pred_labels[:n]
    
    if not true_labels:
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "false_positive_rate": 0,
            "confusion_matrix": None
        }
    
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    
    cm = confusion_matrix(true_labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "confusion_matrix": cm
    }

def merge_masks(mask_dir):
    """合并一个文件夹下所有png图片的白色区域，输出合并后的二值mask（numpy数组）"""
    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith('.png')]
    merged_mask = None
    for f in mask_files:
        mask_path = os.path.join(mask_dir, f)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        # 二值化，确保白色为255
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        if merged_mask is None:
            merged_mask = mask_bin
        else:
            merged_mask = cv2.bitwise_or(merged_mask, mask_bin)
    return merged_mask

def get_fixed_reference_images(class_name, dataset_dir):
    """
    读取指定的参考图片并encode为base64
    """
    # 正常图片
    normal_dir = os.path.join(dataset_dir, class_name, "train", "good")
    normal_imgs = []
    for fname in ["000.png", "001.png", "002.png"]:
        img_path = os.path.join(normal_dir, fname)
        if os.path.exists(img_path):
            normal_imgs.append(encode_image(img_path))
        else:
            print(f"警告: 正常图片不存在: {img_path}")

    # 逻辑异常图片
    logical_dir = os.path.join(dataset_dir, class_name, "test", "logical_anomalies")
    logical_imgs = []
    for fname in ["000.png", "010.png", "080.png"]:
        img_path = os.path.join(logical_dir, fname)
        if os.path.exists(img_path):
            logical_imgs.append(encode_image(img_path))
        else:
            print(f"警告: 逻辑异常图片不存在: {img_path}")

    # 结构异常图片
    structural_dir = os.path.join(dataset_dir, class_name, "test", "structural_anomalies")
    structural_imgs = []
    for fname in ["040.png", "062.png", "084.png"]:
        img_path = os.path.join(structural_dir, fname)
        if os.path.exists(img_path):
            structural_imgs.append(encode_image(img_path))
        else:
            print(f"警告: 结构异常图片不存在: {img_path}")

    return normal_imgs, logical_imgs, structural_imgs

def process_class(class_name):
    """处理单个MVTec类别"""
    print(f"\n{'='*50}")
    print(f"开始处理类别: {class_name}")
    print(f"{'='*50}")
    
    
    align_ref_path = os.path.join(ANNOTATIONS_DIR, f"{class_name}.json")
    print(f"读取对齐参考映射: {align_ref_path}")
    
    if not os.path.exists(align_ref_path):
        print(f"警告: 对齐参考文件不存在 - {align_ref_path}")
        return None, None
    
    try:
        with open(align_ref_path, 'r') as f:
            aligned_ref_paths = json.load(f)
        print(f"成功读取 {len(aligned_ref_paths)} 个测试图像的对齐映射")
    except json.JSONDecodeError:
        print(f"错误: 无法解析JSON文件 - {align_ref_path}")
        return None, None
    
    
    all_results = []
    true_labels = []  
    pred_labels = []  
    
    
    test_root = os.path.join(DATASET_DIR, class_name, "test")
    ground_truth_root = os.path.join(DATASET_DIR, class_name, "ground_truth")  
    
    if not os.path.exists(test_root):
        print(f"错误: 测试目录不存在 - {test_root}")
        return None, None
    
    defect_types = [d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))]

    # 从ground truth中提取参考图片与参考故障
    # logical_gt_ref_path = os.path.join(ground_truth_root, "logical_anomalies", "000")
    # structural_gt_ref_path = os.path.join(ground_truth_root, "structural_anomalies", "000")
    # logical_gt_mask = merge_masks(logical_gt_ref_path)
    # structural_gt_mask = merge_masks(structural_gt_ref_path)
    # cv2.imwrite("temp_logical_mask.png", logical_gt_mask)
    # cv2.imwrite("temp_structural_mask.png", structural_gt_mask)
    # logical_gt_ref_mask = encode_image("temp_logical_mask.png")
    # structural_gt_ref_mask = encode_image("temp_structural_mask.png")
    # logical_gt_ref = encode_image(os.path.join(test_root, "logical_anomalies", "000.png"))
    # structural_gt_ref = encode_image(os.path.join(test_root, "logical_anomalies", "000.png")) 

    # logical_anomaly_img_rel_path = os.path.join(class_name, 'test', 'logical_anomalies', '000.png').replace('\\', '/')
    # logical_normal_ref_img_rel_path = aligned_ref_paths[logical_anomaly_img_rel_path][0]
    # logical_normal_ref_img_full_path = os.path.join(DATASET_DIR, logical_normal_ref_img_rel_path)
    # logical_normal_ref_image = encode_image(logical_normal_ref_img_full_path)

    # structural_anomaly_img_rel_path = os.path.join(class_name, 'test', 'structural_anomalies', '000.png').replace('\\', '/')
    # structural_normal_ref_img_rel_path = aligned_ref_paths[structural_anomaly_img_rel_path][0]
    # structural_normal_ref_img_full_path = os.path.join(DATASET_DIR, structural_normal_ref_img_rel_path)
    # structural_normal_ref_image = encode_image(structural_normal_ref_img_full_path)

    for defect_type in defect_types:

        defect_dir = os.path.join(test_root, defect_type)
        if not os.path.isdir(defect_dir):
            continue
            
        image_files = [f for f in os.listdir(defect_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"处理缺陷类型: {defect_type}, 图像数量: {len(image_files)}")
        
        is_normal = (defect_type.lower() == "good")
        
        for img_idx, img_file in enumerate(image_files):
            
            test_image_path = os.path.join(defect_dir, img_file)
            print(f"  处理图像 [{img_idx+1}/{len(image_files)}]: {img_file}")
            
            # 编码测试图像（无标注框）
            if not os.path.exists(test_image_path):
                print(f"错误: 测试图像不存在 - {test_image_path}")
                continue
                
            try:
                base64_image = encode_image(test_image_path)
            except Exception as e:
                print(f"编码测试图像失败: {str(e)}")
                continue
            
            ori_image_path = os.path.join(class_name, 'test', defect_type, img_file).replace('\\', '/')
            
            if ori_image_path not in aligned_ref_paths:
                print(f"警告: 未找到 {ori_image_path} 的对齐参考")
                continue
            
            ref_image_paths = aligned_ref_paths[ori_image_path]
            ref_image_path = ref_image_paths[0]  
            
            full_ref_path = os.path.join(DATASET_DIR, ref_image_path)
            
            if not os.path.exists(full_ref_path):
                print(f"错误: 参考图像不存在 - {full_ref_path}")
                continue
            
            resized_ref = resize_image(full_ref_path)
            if resized_ref is None:
                print(f"参考图像调整大小失败: {full_ref_path}")
                continue
                
            try:
                resized_ref.save("temp_ref.png")
                base64_ref_image = encode_image("temp_ref.png")
            except Exception as e:
                print(f"参考图像处理失败: {str(e)}")
                continue
        

            prompt = (
                "The first image is a normal reference sample of the second image.\
                The second image is a test image that may contain defects."
                "Please compare these images in detail and determine whether there are any anomalies (defects) in the second image.\
                Please be cautious when making your decision. Only mark the image as anomalous if you have clear and sufficient evidence of a defect."
                f"If there are anomalies, return their locations in the form of coordinates and their types.\n" \
                "Anomalies can be structural damage or logical differences, your comparison and analysis can include but not be limited to the following aspects: \n" \
                "- Structural anomalies: damage to object structure, destruction of surface integrity, structure deformation, surface contamination, appearance of foreign objects, etc.\n" \
                "- Logical anomalies: object occurs some components missing, some wrong components, components appearing in wrong positions, incorrect component quantities and composition, etc.\n" \
                "Requirements (strictly follow): \n" \
                "- You must compare the two images in detail and then tell us how you identified the abnormal area(s) and why it is/they are abnormal compared to other areas in the second image.\n" \
                "- You first need to think about the reasoning and analysis process in the mind and then provide us with the answer.\n" \
                "- You must reason out (strictly follow) the coordinate location of the abnormal area in the reasoning process.\n" \
                "Output format (strictly follow): \n" \
                "<think>your detailed comparative reasoning and analysis process here</think>\
                <answer>Answer with one of the following: \"Yes\" (high confidence anomaly), \"Possible\" (possible anomaly), \"Uncertain\" (uncertain anomaly), or \"No\" (no anomaly). \
                If \"Yes\", \"Possible\", or \"Uncertain\", continue to output all locations, the format should be like {'bbox_2d': [x1, y1, x2, y2], 'label': '<anomaly type>'}</answer>."
            )
            
            # 调用API获取模型响应
            raw_response = get_text(base64_ref_image, 
                                    base64_image, 
                                    prompt)
            
            parsed_response = extract_response(raw_response)
            
            dict_item = {
                "class": class_name,
                "defect_type": defect_type,
                "image_name": img_file,
                "image_path": test_image_path,
                "ref_image": ref_image_path,
                "true_label": "normal" if is_normal else "anomaly",
                "raw_response": raw_response,
                "think": parsed_response["think"],
                "answer": parsed_response["answer"],
                "predicted_label": "normal" if not parsed_response["is_anomaly"] else "anomaly",
                "predicted_bboxes": parsed_response["bboxes"]
            }
            
            all_results.append(dict_item)
            
            # 收集评估数据 (0=正常, 1=异常)
            true_labels.append(0 if is_normal else 1)
            pred_labels.append(0 if not parsed_response["is_anomaly"] else 1)
            
            print("-" * 50)
            print(f"真实标签: {'正常' if is_normal else '异常'}")
            print(f"预测标签: {'正常' if not parsed_response['is_anomaly'] else '异常'}")
            print(f"推理过程: {parsed_response['think'][:100]}..." if parsed_response['think'] else "无推理过程")
            print(f"答案部分: {parsed_response['answer'][:100]}..." if parsed_response['answer'] else "无答案部分")
            print("-" * 50)
    
    metrics = calculate_metrics(true_labels, pred_labels) if true_labels else None
    
    return all_results, metrics

def save_results(class_name, all_results, metrics):
    """保存处理结果"""
    if all_results is None:
        print(f"类别 {class_name} 无结果可保存")
        return
    
    output_dir = os.path.join(BASE_DIR, "results", "mvtec")
    os.makedirs(output_dir, exist_ok=True)

    if metrics and isinstance(metrics.get("confusion_matrix"), np.ndarray):
        metrics["confusion_matrix"] = metrics["confusion_matrix"].tolist()
    
    detailed_path = os.path.join(output_dir, f"{class_name}_detailed_results.json")
    with open(detailed_path, 'w') as f:
        json.dump({
            "class": class_name,
            "metrics": metrics,
            "results": all_results
        }, f, ensure_ascii=False, indent=4)
    
    if metrics:
        report_path = os.path.join(output_dir, f"{class_name}_evaluation_report.txt")
        with open(report_path, 'w') as f:
            f.write(f"{'='*50}\n")
            f.write(f"类别: {class_name} 评估报告\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"总样本数: {len(all_results)}\n")
            f.write(f"准确率: {metrics['accuracy']:.4f}\n")
            f.write(f"精确率: {metrics['precision']:.4f}\n")
            f.write(f"召回率: {metrics['recall']:.4f}\n")
            f.write(f"F1分数: {metrics['f1']:.4f}\n")
            f.write(f"误判率: {metrics['false_positive_rate']:.4f}\n\n")
            if metrics['confusion_matrix']:
                f.write(str(np.array(metrics['confusion_matrix'])))
        
        print("\n" + "=" * 50)
        print(f"类别: {class_name} 评估报告")
        print("=" * 50)
        print(f"总样本数: {len(all_results)}")
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1']:.4f}")
        print(f"误判率: {metrics['false_positive_rate']:.4f}")
        if metrics['confusion_matrix']:
            print(np.array(metrics['confusion_matrix']))
    
    print(f"详细结果保存到: {detailed_path}")
    if metrics:
        print(f"评估报告保存到: {report_path}")

def validate_api_key():
    """验证API密钥是否设置"""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误: 未设置DASHSCOPE_API_KEY环境变量")
        print("请设置环境变量，例如: export DASHSCOPE_API_KEY='your-api-key'")
        return False
    
    dashscope.api_key = api_key
    print("成功设置DashScope API密钥")
    return True

def generate_overall_report():
    """生成整体评估报告"""
    output_dir = os.path.join(BASE_DIR, "results", "mvtec")
    overall_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "false_positive_rate": []
    }
    
    # 收集所有类别的评估指标
    for class_name in SELECTED_CLASSES:
        report_path = os.path.join(output_dir, f"{class_name}_evaluation_report.txt")
        if not os.path.exists(report_path):
            continue
            
        json_path = os.path.join(output_dir, f"{class_name}_detailed_results.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if 'metrics' in data:
                    metrics = data['metrics']
                    overall_metrics["accuracy"].append(metrics.get("accuracy", 0))
                    overall_metrics["precision"].append(metrics.get("precision", 0))
                    overall_metrics["recall"].append(metrics.get("recall", 0))
                    overall_metrics["f1"].append(metrics.get("f1", 0))
                    overall_metrics["false_positive_rate"].append(metrics.get("false_positive_rate", 0))
            except:
                pass
    
    # 计算平均指标
    avg_metrics = {}
    for key, values in overall_metrics.items():
        if values:
            avg_metrics[f"avg_{key}"] = sum(values) / len(values)
    
    report_path = os.path.join(output_dir, "overall_evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"{'='*50}\n")
        f.write(f"MVTec 数据集整体评估报告\n")
        f.write(f"{'='*50}\n\n")
        
        for key, value in avg_metrics.items():
            f.write(f"{key.replace('avg_', '平均').capitalize()}: {value:.4f}\n")
        
        f.write("\n各指标详情:\n")
        for key, values in overall_metrics.items():
            f.write(f"{key.capitalize()}: {[round(v, 4) for v in values]}\n")
    
    print(f"\n整体评估报告保存到: {report_path}")

if __name__ == '__main__':
    # 验证并设置API，API要配置到环境变量里
    if not validate_api_key():
        sys.exit(1)
    
    
    # 处理选定的类别
    for class_name in SELECTED_CLASSES:
        print(f"\n开始处理类别: {class_name}")
        results, metrics = process_class(class_name)
        save_results(class_name, results, metrics)
    
        print(f"\n完成处理: {class_name}")
    
    # 生成整体评估报告
    generate_overall_report()
    
    print("\n所有选定类别处理完成！")