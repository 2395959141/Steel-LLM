<div align="center">

# 🚀 尝试训练1.4b参数的中英文模型 🚀

![训练中](https://img.shields.io/badge/状态-训练中-blue)
![参数](https://img.shields.io/badge/参数-1.4B-orange)
![数据量](https://img.shields.io/badge/数据量-1.5TB-green)

### 原仓库链接 👉 [steel-llm](https://github.com/zhanshijinwat/Steel-LLM) 👈

</div>

## 📝 项目简介

`tiny_llm_1.4b`参考了steel-llm的从零预训练大模型项目，使用了1.5TB的中英文混合数据进行训练。
训练开始时间：2025/5/15，目前仍在训练中...

## 🔗 学习过程参考开源项目

- [steel-llm](https://github.com/zhanshijinwat/Steel-LLM)
- [minimind](https://github.com/jingyaogong/minimind)
- [MINI_LLM](https://github.com/jiahe7ay/MINI_LLM)

## 📚 数据收集

本项目使用了以下高质量数据集：

| 数据集名称 | 来源 | 描述 |
|------------|------|------|
| Chinese Fineweb Edu | [OpenCSG](https://www.modelscope.cn/datasets/opencsg/chinese-fineweb-edu) | 高质量教育中文预训练语料数据集 |
| 百度百科数据 | [ModelScope](https://www.modelscope.cn/datasets/fq980207/563w_baidubaike) | 包含563万条百度百科数据 |
| StarCoderData | [ModelScope](https://www.modelscope.cn/datasets/swift/starcoderdata) | 编程代码数据集 |
| IndustryCorpus2 | [BAAI](https://www.modelscope.cn/datasets/BAAI/IndustryCorpus2) | 30类行业分类的高质量预训练数据集（仅选用标记为高质量的数据） |

## 📊 训练进度

- [✔] 数据收集与预处理
- [✔] 预训练阶段
- [ ] SFT指令微调
- [ ] RLHF强化学习







