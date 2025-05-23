import glob
import json
import os
import sys
from pathlib import Path
import sys
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
import random
random.seed(666)
import lit_gpt.packed_dataset as packed_dataset
from lit_gpt import Config, Tokenizer
from transformers import AutoTokenizer
import time

filenames_sample = []
# key: name value: dir

def process_line_text(text, tokenizer):
    t1 = time.time()
    text_ids = tokenizer.encode(text)
    print(time.time() - t1)
    return text_ids

def process_jsonl_file(set_name, file_dir, builder, tokenizer, cache_lines_num):
    """可处理万卷zh，万卷en，天工"""
    cache_text = ""
    counter = 0
    with open(file_dir, encoding="utf-8") as f:
        for row in tqdm(f):
            counter += 1
            # 不同数据集存的字段不一样
            try:
                if set_name=="baike" or set_name=="starcode" or set_name == "cc_i3_HQ" or set_name == "chinese_fine_web_edu" or set_name == "industry_corpus2":
                    text = json.loads(row)["text"]
                elif set_name=="wanjuan_zh" or set_name=="wanjuan_en":
                    text = json.loads(row)
                    if "content" in text:
                        text = text["content"]
                    # 有考试题数据
                    elif "q_main" in text:
                        text = text["q_main"]+text["option_a"]+text["option_b"]+\
                            text["option_c"]+text["option_d"]+text["option_e"]+\
                            "标准答案："+text["std_ans"]+"解析："+text["answer_detail"]
                else:
                    raise NameError(f"process_jsonl_file not have {set_name}")
            except Exception as e:
                print(f"file {file_dir} line: {counter} keys:{json.loads(row).keys()} read error: {e}")
                counter -= 1
                continue
            text += "<|im_end|>"
            cache_text += text
            if counter%cache_lines_num==0:
                text_ids = process_line_text(text=cache_text, tokenizer=tokenizer)
                builder.add_array(np.array(text_ids, dtype=builder.dtype))
                cache_text = "" 
        if cache_text!="":
            text_ids = process_line_text(text=cache_text, tokenizer=tokenizer)
            builder.add_array(np.array(text_ids, dtype=builder.dtype))

def multiprocess_data(set_name, file_dir_list, builder, tokenizer, cache_lines_num, process_idx=0):
    try:
        for file_dir in file_dir_list:
            t0 = time.time()
            if set_name in ["baike", "cc_i3_HQ", "chinese_fine_web_edu","industry_corpus2","starcode"]:
                process_jsonl_file(set_name=set_name, file_dir=file_dir, builder=builder,
                        tokenizer=tokenizer, cache_lines_num=cache_lines_num)
            else:
                raise NameError(f"not have this set {set_name}")
            print(f"process {process_idx} end a file {file_dir} cost {time.time()-t0}s")
        print(f"{builder._prefix} have {builder._counter+1} files {builder.all_tokens} tokens")
        if builder._idx/builder._chunk_size > 0.05:
            builder.write_reminder()
        else:
            print(f"remind part is too small {builder._idx/builder._chunk_size}")

    except Exception  as e:
        print(f"multiprocess error:  {str(e)}")

def process_file_range(set_name, file_path, builder, tokenizer, cache_lines_num, start_line, end_line, process_idx):
    """处理文件的指定行范围"""
    try:
        t0 = time.time()
        cache_text = ""
        line_count = 0
        processed_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            # 跳过之前的行
            for _ in range(start_line):
                next(f)
                line_count += 1
            
            # 处理分配的行范围
            for row in tqdm(f, desc=f"Process {process_idx}", total=end_line-start_line):
                line_count += 1
                if line_count > end_line:
                    break
                    
                try:
                    # 提取文本的逻辑与原函数相同
                    if set_name=="baike" or set_name=="starcode" or set_name == "cc_i3_HQ" or set_name == "chinese_fine_web_edu" or set_name == "industry_corpus2":
                        text = json.loads(row)["text"]
                    elif set_name=="wanjuan_zh" or set_name=="wanjuan_en":
                        # 原有逻辑...
                        pass
                    else:
                        raise NameError(f"process_file_range not have {set_name}")
                except Exception as e:
                    print(f"Process {process_idx}: line {line_count} read error: {e}")
                    continue
                
                text += "<|im_end|>"
                cache_text += text
                processed_count += 1
                
                if processed_count % cache_lines_num == 0:
                    text_ids = process_line_text(text=cache_text, tokenizer=tokenizer)
                    builder.add_array(np.array(text_ids, dtype=builder.dtype))
                    cache_text = ""
            
            # 处理剩余内容
            if cache_text != "":
                text_ids = process_line_text(text=cache_text, tokenizer=tokenizer)
                builder.add_array(np.array(text_ids, dtype=builder.dtype))
        
        print(f"Process {process_idx} 完成: 处理了 {processed_count} 行, 耗时 {time.time()-t0}s")
        print(f"{builder._prefix} 共有 {builder._counter+1} 文件, {builder.all_tokens} tokens")
        
        if builder._idx/builder._chunk_size > 0.05:
            builder.write_reminder()
        else:
            print(f"剩余部分过小 {builder._idx/builder._chunk_size}")
            
    except Exception as e:
        print(f"Process {process_idx} 错误: {str(e)}")

def prepare_full(
    source_path: Path, checkpoint_dir: Path, destination_path: Path, chunk_size: int, 
    match: str = "", max_files = 100000000000000, cache_lines_num = 1000, process_num=16
) -> None:
    """
    chunk_size: chunk_size=(block_size + 1) * 1024 （block是序列长度）
    """
    import zstandard as zstd

    destination_path.mkdir(parents=True, exist_ok=True)

    # tokenizer = Tokenizer(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    # "arxiv": "arxiv/arxiv*"
    for set_name, pattern in filename_sets.items():
        # 循环不同的数据来源
        if match and match not in set_name:
            raise NameError("should not skip...")
            continue
        t0 = time.time()
        
        # 先排序再shuffle，保证每次数据处理顺序
        filenames = glob.glob(os.path.join(source_path, pattern), recursive=True)
        process_num = min(process_num, len(filenames))
        filenames = sorted(filenames)
        random.shuffle(filenames)
        print(filenames)
        print(os.path.join(source_path, pattern), type(os.path.join(source_path, pattern))) 
        filenames = filenames[:max_files]
        print(f"{set_name} have { len(filenames)} files...")
        if not filenames:
            raise RuntimeError(f"No files matching {pattern} found at {source_path}.") 
        # 将文件分到不同的进程
        process_filenames_list = [[] for i in range(process_num)]
        for i in range(len(filenames)):
            process_idx = i%process_num
            process_filenames_list[process_idx].append(filenames[i])
        process_filenames_num = [len(l) for l in process_filenames_list]
        print(process_filenames_list)
        print("file nums in each process:", process_filenames_num)
        
        builder_list = []
        for i in range(process_num):
            builder = packed_dataset.PackedDatasetBuilder(
                outdir=destination_path,
                prefix=set_name+"_"+f"process{i}",
                chunk_size=chunk_size,
                # 填充token，用pad，而不是用eos_id
                sep_token=tokenizer.pad_token_id,
                dtype="auto",
                vocab_size=len(tokenizer),
            )
            builder_list.append(builder)

        # 如果只有一个文件但需要多进程处理
        if len(filenames) == 1 and process_num > 1:
            file_path = source_path / filenames[0]
            # 计算文件总行数
            total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
            print(f"文件 {file_path} 共有 {total_lines} 行")
            
            # 计算每个进程处理的行数范围
            lines_per_process = total_lines // process_num
            process_list = []
            
            for process_idx in range(process_num):
                start_line = process_idx * lines_per_process
                end_line = (process_idx + 1) * lines_per_process if process_idx < process_num - 1 else total_lines
                
                process_list.append(mp.Process(
                    target=process_file_range, 
                    args=(set_name, file_path, builder_list[process_idx], tokenizer, 
                          cache_lines_num, start_line, end_line, process_idx)
                ))
            
            [p.start() for p in process_list]
            [p.join() for p in process_list]
        else:
            process_list = []
            for process_idx in range(process_num):
                a_process_filenames = process_filenames_list[process_idx]
                a_process_filenames = [source_path / name for name in a_process_filenames]
                # 参数：(set_name, file_dir_list, builder, tokenizer, cache_lines_num)
                process_list.append(mp.Process(target=multiprocess_data, 
                                               args=(set_name, a_process_filenames, builder_list[process_idx],
                                                     tokenizer, cache_lines_num, process_idx))) 
            [p.start() for p in process_list]
            [p.join() for p in process_list]
                
        print(f"processed set_name {set_name} cost {time.time()-t0}")


filename_sets = {
    "baike": "/DATA/disk2/yuhang/.cache/steel_dataset/step2_split_jsonl/baidu_baike/*.jsonl",
    #"cc_i3_HQ": "/DATA/disk2/yuhang/.cache/steel_dataset/step2_split_jsonl/cc_ic_HQ/*.jsonl",
    #"chinese_fine_web_edu": "/DATA/disk2/yuhang/.cache/steel_dataset/step2_split_jsonl/chinese_fine_web_edu/*.jsonl",
    #"industry_corpus2": "/DATA/disk2/yuhang/.cache/steel_dataset/step2_split_jsonl/industry_corpus2/*.jsonl",
    #"starcode": "/DATA/disk2/yuhang/.cache/steel_dataset/step2_split_jsonl/star_code/*.jsonl"
}
def prepare(
    source_path: Path = Path("/"),
    # tokenizer地址
    checkpoint_dir: Path = Path("/DATA/disk2/yuhang/.cache/modelscope/models/Qwen/Qwen2___5-0___5B-Instruct"),
    destination_path: Path = Path("/DATA/disk2/yuhang/.cache/steel_dataset/step3_final_data/baike"),
    sample: bool = False,
    match: str = "",
    max_files = 10000000000,
    # 序列长度
    block_size = 2048,
    # 一个数据分片有多少个序列
    # blocks_in_a_chunck=1024时，一个chunk是8M大小 
    blocks_in_a_chunck = 1024*200,
    # 读取cache_lines_num行json数据后进行一次保存
    cache_lines_num = 35,
    # 处理数据进程数
    process_num = 60


) -> None:
    """Prepare the SteelLLM dataset. We assume tokenizer has been trained."""
    prepare_fn = prepare_full
    prepare_fn(
        source_path=source_path,
        checkpoint_dir=checkpoint_dir,
        destination_path=destination_path,
        chunk_size=(block_size + 1) * blocks_in_a_chunck,  # block size + 1 for causal, 1024 blocks
        match=match,
        max_files = max_files,
        cache_lines_num = cache_lines_num,
        process_num = process_num
    )


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)
