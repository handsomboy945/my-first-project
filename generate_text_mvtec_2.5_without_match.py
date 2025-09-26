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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

#全局配置
os.environ["DASHSCOPE_API_KEY"] = "sk-1e6d5dd3c6a94151ab15f67cd0b281a8"
BASE_DIR = "./LLMtry"
DATASET_DIR = "./dataset/mvtec_loco_anomaly_detection" 
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "annotations", "mvtec_loco")
TEST_IMAGES_DIR = os.path.join(DATASET_DIR, "{class_name}", "test")

#MVTEC中的数据类别（不同数据集需要对应修改）
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
        is_anomaly = 1-(anomaly_level == "no")
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

def get_text(base64_image, prompt):   #记得在这里修改模型为qwen2.5-vl-7b-instruct
    """使用DashScope API获取模型响应（单张图像）"""
    completed = False
    max_retries = 5
    retry_count = 0
    
    while not completed and retry_count < max_retries:
        time.sleep(3)  
        try:
            response = dashscope.MultiModalConversation.call(
                model='qwen2.5-vl-7b-instruct',
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional industrial image inspector, particularly skilled at identifying abnormal (defective) areas in images."
                    },
                    {
                        "role": "user",
                        "content": [
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
    """智能调整图像大小，保持宽高比 - 兼容不同PIL版本"""
    try:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        if width > height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))
        
        # 处理不同PIL版本的兼容性问题
        try:
            # 新版本PIL (>=9.1.0)
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        except AttributeError:
            # 旧版本PIL (<9.1.0)
            return image.resize((new_width, new_height), Image.LANCZOS)
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

def process_class(class_name):
    """处理单个MVTec类别"""
    print(f"\n{'='*50}")
    print(f"开始处理类别: {class_name}")
    print(f"{'='*50}")
    
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
            
            # 编码测试图像
            if not os.path.exists(test_image_path):
                print(f"错误: 测试图像不存在 - {test_image_path}")
                continue
                
            try:
                base64_image = encode_image(test_image_path)
            except Exception as e:
                print(f"编码测试图像失败: {str(e)}")
                continue
            
            # 构建提示词（在这里对prompt进行修改尝试）
            prompt = (
                "You are a professional industrial image inspector. "
                "Given an industrial image, determine if there are any anomalies (defects) in the image. "
                f"If there are anomalies, return their locations in the form of coordinates and their types.\n" \
                "Anomalies can be structural damage or logical differences, your comparison and analysis can include but not be limited to the following aspects: \n" \
                "- Structural anomalies: damage to object structure, destruction of surface integrity, structure deformation, surface contamination, appearance of foreign objects, etc.\n" \
                "- Logical anomalies: object occurs some components missing, some wrong components, components appearing in wrong positions, incorrect component quantities and composition, etc.\n" \
                "Requirements (strictly follow): \n" \
                "- You must compare the two images in detail and then tell us how you identified the abnormal area(s) and why it is/they are abnormal compared to other areas in the second image.\n" \
                "- Provide detailed reasoning in <think> tags. Then, provide your final answer in <answer> tags. Include all your answer in the <think> and <answer> tags and do not output any extra text outside the tags.\n" \
                "- You must reason out the coordinate location of the abnormal area in the reasoning process.\n" \
                "Output format (strictly follow): \n" \
                "<think>your detailed comparative reasoning and analysis process here</think>\
                <answer>Answer with one of the following: \"Yes\" (high confidence anomaly), \"Possible\" (possible anomaly), \"Uncertain\" (uncertain anomaly), or \"No\" (no anomaly). \
                If \"Yes\", \"Possible\", or \"Uncertain\", continue to output all locations, the format should be like {'bbox_2d': [x1, y1, x2, y2], 'label': '<anomaly type>'}</answer>."
            )
            
            # 调用API获取模型响应
            raw_response = get_text(base64_image, prompt)
            
            parsed_response = extract_response(raw_response)
            
            dict_item = {
                "class": class_name,
                "defect_type": defect_type,
                "image_name": img_file,
                "image_path": test_image_path,
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
            print(f"原始响应: {raw_response}" if raw_response else "无响应")
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
    
    detailed_path = os.path.join(output_dir, f"{class_name}_detailed_results_2.5.json")
    try:
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump({
                "class": class_name,
                "metrics": metrics,
                "results": all_results
            }, f, ensure_ascii=False, indent=4)
    except UnicodeEncodeError as e:
        print(f"保存详细结果时遇到编码错误: {e}")
        # 尝试使用 ensure_ascii=True 来转义非 ASCII 字符
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump({
                "class": class_name,
                "metrics": metrics,
                "results": all_results
            }, f, ensure_ascii=True, indent=4)
        print("已使用 ASCII 转义保存结果")
    
    if metrics:
        report_path = os.path.join(output_dir, f"{class_name}_evaluation_report_2.5.txt")
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
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
        except UnicodeEncodeError as e:
            print(f"保存评估报告时遇到编码错误: {e}")
            # 尝试清理特殊字符后保存
            with open(report_path, 'w', encoding='utf-8') as f:
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
            print("已清理特殊字符后保存报告")
        
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
        report_path = os.path.join(output_dir, f"{class_name}_evaluation_report_2.5.txt")
        if not os.path.exists(report_path):
            continue
            
        json_path = os.path.join(output_dir, f"{class_name}_detailed_results_2.5.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'metrics' in data:
                    metrics = data['metrics']
                    overall_metrics["accuracy"].append(metrics.get("accuracy", 0))
                    overall_metrics["precision"].append(metrics.get("precision", 0))
                    overall_metrics["recall"].append(metrics.get("recall", 0))
                    overall_metrics["f1"].append(metrics.get("f1", 0))
                    overall_metrics["false_positive_rate"].append(metrics.get("false_positive_rate", 0))
            except Exception as e:
                print(f"读取 {class_name} 结果时出错: {e}")
                continue
    
    # 计算平均指标
    avg_metrics = {}
    for key, values in overall_metrics.items():
        if values:
            avg_metrics[f"avg_{key}"] = sum(values) / len(values)
    
    report_path = os.path.join(output_dir, "overall_evaluation_report_2.5.txt")
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*50}\n")
            f.write(f"MVTec 数据集整体评估报告\n")
            f.write(f"{'='*50}\n\n")
            
            for key, value in avg_metrics.items():
                f.write(f"{key.replace('avg_', '平均').capitalize()}: {value:.4f}\n")
            
            f.write("\n各指标详情:\n")
            for key, values in overall_metrics.items():
                f.write(f"{key.capitalize()}: {[round(v, 4) for v in values]}\n")
    except UnicodeEncodeError as e:
        print(f"保存整体报告时遇到编码错误: {e}")
        # 尝试使用 ASCII 编码
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*50}\n")
            f.write(f"MVTec 数据集整体评估报告\n")
            f.write(f"{'='*50}\n\n")
            
            for key, value in avg_metrics.items():
                f.write(f"{key.replace('avg_', '平均').capitalize()}: {value:.4f}\n")
            
            f.write("\n各指标详情:\n")
            for key, values in overall_metrics.items():
                f.write(f"{key.capitalize()}: {[round(v, 4) for v in values]}\n")
        print("已使用 UTF-8 编码保存整体报告")
    
    print(f"\n整体评估报告保存到: {report_path}")

if __name__ == '__main__':
    # 验证并设置API密钥
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