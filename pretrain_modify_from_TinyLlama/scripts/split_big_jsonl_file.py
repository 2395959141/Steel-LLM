import os
import json
import argparse
import glob
from pathlib import Path
from tqdm import tqdm

def get_file_size(file_path):
    """获取文件大小（GB）"""
    return os.path.getsize(file_path) / (1024 * 1024 * 1024)

def split_single_jsonl_file(input_path, output_dir, max_size_gb=5):
    """拆分单个JSONL文件"""
    os.makedirs(output_dir, exist_ok=True)
    max_size_bytes = max_size_gb * 1024 * 1024 * 1024
    input_filename = Path(input_path).stem
    
    # 获取文件总大小用于进度条
    total_size = os.path.getsize(input_path)
    processed_size = 0
    
    file_index = 0
    current_size = 0
    current_file = None
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            # 使用tqdm创建进度条
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"处理 {input_filename}") as pbar:
                for line in f:
                    line_size = len(line.encode('utf-8'))
                    
                    # 如果当前文件不存在或已接近大小限制，则创建新文件
                    if current_file is None or current_size + line_size > max_size_bytes:
                        if current_file is not None:
                            current_file.close()
                        
                        file_index += 1
                        output_path = os.path.join(output_dir, f"{input_filename}_{file_index}.jsonl")
                        current_file = open(output_path, 'w', encoding='utf-8')
                        current_size = 0
                        print(f"开始写入文件: {output_path}")
                    
                    # 写入当前行
                    current_file.write(line)
                    
                    # 更新当前文件大小和进度条
                    current_size += line_size
                    processed_size += line_size
                    pbar.update(line_size)
        
        # 关闭最后一个文件
        if current_file is not None:
            current_file.close()
            print(f"完成拆分 {input_path}，共生成 {file_index} 个文件")
            
        return True
    except Exception as e:
        print(f"处理文件 {input_path} 时出错: {e}")
        if current_file is not None:
            current_file.close()
        return False

def process_directory(input_pattern, output_dir, max_size_gb=5):
    """处理匹配模式的所有JSONL文件"""
    file_paths = glob.glob(input_pattern)
    if not file_paths:
        print(f"没有找到匹配 '{input_pattern}' 的文件")
        return
    
    print(f"找到 {len(file_paths)} 个文件待处理")
    for file_path in file_paths:
        file_size_gb = get_file_size(file_path)
        print(f"处理文件: {file_path} (大小: {file_size_gb:.2f} GB)")
        split_single_jsonl_file(file_path, output_dir, max_size_gb)

def main():
    parser = argparse.ArgumentParser(description='将大型JSONL文件拆分为多个较小的文件')
    parser.add_argument('input', help='输入的JSONL文件路径或匹配模式（如: data/*.jsonl）')
    parser.add_argument('output_dir', help='输出文件的目录')
    parser.add_argument('--max_size_gb', type=float, default=5, help='每个输出文件的最大大小（GB）')
    parser.add_argument('--process_dir', action='store_true', help='处理整个目录中匹配的文件')
    
    args = parser.parse_args()
    
    if args.process_dir or '*' in args.input:
        process_directory(args.input, args.output_dir, args.max_size_gb)
    else:
        split_single_jsonl_file(args.input, args.output_dir, args.max_size_gb)

if __name__ == '__main__':
    main()
