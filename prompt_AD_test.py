import os
import json
import base64
import time
import re
import io
from PIL import Image
import dashscope
from http import HTTPStatus
import sys
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import dashscope

#全局变量
BASE_DIR = None
DATASET_DIR = None
TEST_IMAGES_DIR = None
ANNOTATIONS_DIR = None
SELECTED_CLASSES = None

def encode_image(image_path):
    """将图像编码为Base64字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_response(text):
    """从模型响应中提取<think>和<answer>部分"""
    try:
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        answer_content = answer_match.group(1).strip() if answer_match else "未提取到答案"

        if answer_match:
            think_content = text[:answer_match.start()].strip()
            think_match = re.search(r'<think>(.*?)</think>', think_content, re.DOTALL)
            if think_match:
                think_content = think_match.group(1).strip()
        else:
            think_content = "未提取到推理过程"

        
        is_anomaly = "yes" in answer_content.lower()
        bboxes = []
        
        if is_anomaly:
            bbox_matches = []
            bbox_1 = re.findall(r'\{\s*"bbox_2d"\s*:\s*\[([\d\.,\s]+)\]\s*,\s*"label"\s*:\s*"([^"]+)"\s*\}', text)
            bbox_2 = re.findall(r"\{\s*'bbox_2d'\s*:\s*\[([\d\.,\s]+)\]\s*,\s*'label'\s*:\s*'([^']+)'\s*\}",text)
            bbox_matches.extend(bbox_1)
            bbox_matches.extend(bbox_2)
            for match in bbox_matches:
                coords = [float(x.strip()) for x in match[0].split(',')]
                bboxes.append({
                    "bbox_2d": coords,
                    "label": match[1]
                })
        
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
            "is_anomaly": False,
            "bboxes": []
        }

def get_text(base64_ref_image, base64_image, prompt):
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
                        "content": "You are a professional industrial image inspector, particularly skilled at identifying abnormal (defective) areas in images."
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

def mask_to_bbox(mask_image):
    """
    从mask图像计算真实边界框 (x1, y1, x2, y2格式)
    """
    mask_array = np.array(mask_image)
    binary_mask = (mask_array > 0).astype(np.uint8)
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    x1 = x_indices[0]
    y1 = y_indices[0]
    x2 = x_indices[-1]
    y2 = y_indices[-1]
    return [x1, y1, x2, y2]

def calculate_iou(bbox1, bbox2):
    """
    计算bbox2与bbox1中所有边界框的整体IoU
    
    参数:
        bbox1: 字典列表，每个字典包含一个边界框，格式为 [{'bbox': [x1, y1, x2, y2]}, ...]
        bbox2: 单个边界框，格式为 [x1, y1, x2, y2]
    
    返回:
        float: 整体IoU值，范围 [0, 1]
    """
    if not bbox1 or bbox2 is None:
        return 0.0
    bboxes1 = []
    for item in bbox1:
        try:
            bboxes1.append(item['bbox_2d'])
        except:
            raise ValueError("每个字典必须包含有效的'bbox_2d'键，格式为[x1, y1, x2, y2]")
    if not bboxes1:
        return 0.0
    overall_x1 = min(bbox[0] for bbox in bboxes1)
    overall_y1 = min(bbox[1] for bbox in bboxes1)
    overall_x2 = max(bbox[2] for bbox in bboxes1)
    overall_y2 = max(bbox[3] for bbox in bboxes1)
    x1_inter = max(overall_x1, bbox2[0])
    y1_inter = max(overall_y1, bbox2[1])
    x2_inter = min(overall_x2, bbox2[2])
    y2_inter = min(overall_y2, bbox2[3])
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    overall_area = (overall_x2 - overall_x1) * (overall_y2 - overall_y1)
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = overall_area + bbox2_area - inter_area
    iou = inter_area / union_area
    return iou

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
    
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    cm = confusion_matrix(true_labels, pred_labels).tolist()
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "confusion_matrix": cm
    }

def process_class(class_name, prompt):
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
    
    if not os.path.exists(test_root):
        print(f"错误: 测试目录不存在 - {test_root}")
        return None, None
    
    defect_types = [d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))]
    
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
                resized_ref.save("./temp_image/temp_ref.png")
                base64_ref_image = encode_image("./temp_image/temp_ref.png")
            except Exception as e:
                print(f"参考图像处理失败: {str(e)}")
                continue
            
            # 调用API获取模型响应
            raw_response = get_text(base64_ref_image, base64_image, prompt)
            
            parsed_response = extract_response(raw_response)

            # 加入置信度为后续跳去典型错误做准备
            if 'yes' in parsed_response['answer'].lower():
                confience = 1.0
            elif 'possible' in parsed_response['answer'].lower():
                confience = 0.7
            elif 'uncertain' in parsed_response['answer'].lower():
                confience = 0.4
            elif 'no' in parsed_response['answer'].lower():
                confience = 0.0
            else:
                confience = 0.0
            
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
                "confidence": confience,
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
            print(f"原始回答： {raw_response}" if raw_response else "无回答")
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

class prompt_AD_tester:
    def __init__(self, class_name, base_dir, dataset_dir, annotation_dir, api_key):
        # 设置全局变量
        global BASE_DIR, DATASET_DIR, ANNOTATIONS_DIR
        BASE_DIR = base_dir
        DATASET_DIR = dataset_dir
        ANNOTATIONS_DIR = annotation_dir
        dashscope.api_key = api_key
        self.class_name = class_name

    def prompt_AD(self, prompt):
        # 开始处理类别
        print(f"\n开始处理类别: {self.class_name}")
        results, metrics = process_class(self.class_name, prompt)
        save_results(self.class_name, results, metrics)
        print(f"\n完成处理: {self.class_name}")
        return metrics, results
        

