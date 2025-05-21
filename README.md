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


### 2025.5.16

- 训练时间：68小时
- 当前全局步数（Global Step）：55680
- 测试了50k的检查点（checkpoint），模型输出逐渐流畅，仍然不能接上输入的上文语义
- 困惑度（PPL）稳定在17左右
- 测试：

    用户: c++中的虚函数
    
    助手: 在2019年,我们学习了Java,Java作为新一代的编程语言,其主要面向对象、多态、继承、封装、动态规划等方面,通过深入浅出的学习,对Java语言有了一个全新的认识。Java是一种面向过程、面向对象、多态、继承和封装的语言,它具有良好的可移植性,并且支持多种平台,在电子商务、游戏开发、web应用等方面有着广泛的应用前景。

### 2025.5.21

- 训练时间：8天
- 当前全局步数（Global Step）：156000

-对100000 step的检查点文件，进行了初步的sft微调并行进行测试。
-微调数据集：baai_instruct_7m,code_feedback_custom,openhermes_custom

- 测试：


    用户: 介绍一下北京

    助手回答:

    北京是中国的首都，也是中国的第二大城市。北京的地理位置非常重要，它位于中国北方的黄土高原上，是华北地区的一座重要的城市。北京的气候条件也非常适合种植蔬菜和水果。

    北京的经济也非常发达，是中国重要的国际贸易中心之一。北京也是一座现代化的城市，有许多高楼大厦和现代化的商业设施。北京也是中国的重要交通枢纽，可以方便地前往上海、广州等南方城市。

    北京还拥有许多历史文化名城，如故宫、天坛、颐和园等。这些名城都是中国文化遗产的重要代表，吸引了无数游客前来参观。北京还有许多著名的历史建筑，如故宫、天坛等。

    总之，北京是中国的首都，它拥有许多现代化的建筑和文化景点，是中国的美丽城市之一。


