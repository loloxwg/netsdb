# 索引用于重复检测

## 依赖

```shell script
conda create -n deduplication python=3.8
conda activate deduplication
pip install -r requirements.txt
```


## 任务

### 情感文本分类

#### 数据集介绍与预处理

我们在以下三个数据集上训练了几个模型来进行情感分类：[imdb_reviews](https://www.tensorflow.org/datasets/catalog/imdb_reviews)、[yelp_polarity_reviews](https://www.tensorflow.org/datasets/catalog/yelp_polarity_reviews)和[civil_comments](https://www.tensorflow.org/datasets/catalog/civil_comments)。

|  **数据集**  | **输入** |  **标签**  | **训练点数量** | **测试点数量** |
|:------------:|:--------:|:----------:|:--------------:|:--------------:|
|  IMDB Review | 文本句子 | 0表示负面，1表示正面 | 40,000 | 10,000 |
|  Yelp Review | 文本句子 | 0表示负面，1表示正面 | 478,400 | 119,600 |
| Civil Comment | 文本句子 | 0表示负面，1表示正面 | 1,521,755 | 380,439 |

这三个评论数据集都以句子形式存储。我们应用了数据预处理来为每个句子生成情感标签，定义如下：

- yelp_polarity_reviews：原始数据集使用标签`2`表示正面评论，标签`1`表示负面评论。我们将其预处理为`1`表示正面评论，`0`表示负面评论。

- imdb_reviews：原始数据集使用标签`1`表示正面评论，标签`0`表示负面评论。我们保持相同设置。

- civil_comments：原始数据集使用范围从0到1的浮点值来表示评论的毒性。我们将毒性大于`0.5`的评论的标签设置为`1`，否则将其标签设置为`0`。

#### 访问预处理后的数据集

预处理后的数据集可以通过[这里](https://drive.google.com/uc?id=1nYzDDSJGkjCsVQI4gbSdC3DafQ7Ez107)下载。或者，您可以使用[downloader](downloader.py)下载预处理后的数据集。

#### 模型训练

我们用于训练模型的架构是：嵌入层（可训练/不可训练权重）+ 2个全连接层。您可以通过使用我们的[model_trainer_sentiment_task](model_trainer_sentiment_task.py)从头开始训练模型。或者，您可以使用[downloader](downloader.py)下载我们预训练的模型。

我们训练的情感分类模型总结在下表中：

| 模型 | 嵌入层 | 设置 | 下载链接 |
|------|--------|------|----------|
| w2v_wiki500_yelp_embed_nontrainable | [w2v_wiki500](https://tfhub.dev/google/Wiki-words-500/2) | 在yelp_review数据集上训练的文本分类模型，嵌入层不可训练 | [链接](https://drive.google.com/uc?id=1-6T6c5MaaceARapMnPBEj0KP_P1-rg0y) |
| w2v_wiki500_yelp_embed_trainable | [w2v_wiki500](https://tfhub.dev/google/Wiki-words-500/2) | 在yelp_review数据集上训练的文本分类模型，嵌入层可训练 | [链接](https://drive.google.com/uc?id=1GgVaiexh643C7LlVH0qJHqlJxBZo71M0) |
| w2v_wiki500_imdb_embed_nontrainable | [w2v_wiki500](https://tfhub.dev/google/Wiki-words-500/2) | 在imdb_review数据集上训练的文本分类模型，嵌入层不可训练 | [链接](https://drive.google.com/uc?id=1jG2UjS75KG4pOeRKCWYdZd1o3TQlewfy) |
| w2v_wiki500_imdb_embed_trainable | [w2v_wiki500](https://tfhub.dev/google/Wiki-words-500/2) | 在imdb_review数据集上训练的文本分类模型，嵌入层可训练 | [链接](https://drive.google.com/uc?id=1-1I6r7kBQhwyHi5_Xatrwz_VFVM3H8GH) |
| w2v_wiki500_civil_comment_embed_trainable | [w2v_wiki500](https://tfhub.dev/google/Wiki-words-500/2) | 在civil_comment数据集上训练的文本分类模型，嵌入层可训练 | [链接](https://drive.google.com/uc?id=1--bCnYYoe0mXseM783qqkncVMQVQmbhy) |
| nnlm_en_dim_128_yelp_embed_trainable | [nnlm_en_dim_128](https://tfhub.dev/google/nnlm-en-dim128/2) | 在yelp_review数据集上训练的文本分类模型，嵌入层可训练 | [链接](https://drive.google.com/uc?id=1wm6qgWzqUOTTAoesHhv3aMRv1MartY2p) |
| nnlm_en_dim_50_imdb_embed_trainable | [nnlm_en_dim_50](https://tfhub.dev/google/nnlm-en-dim50/2) | 在imdb_review数据集上训练的文本分类模型，嵌入层可训练 | [链接](https://drive.google.com/uc?id=1-0-a0ldTD_BNr1cIZTU56AjJ0ypxjWw1) |
| w2v_wiki250_civil_comment_embd_trainable | [w2v_wiki250](https://tfhub.dev/google/Wiki-words-250/2) | 在civil_comment数据集上训练的文本分类模型，嵌入层可训练 | [链接](https://drive.google.com/uc?id=1Idxvo0Mt3ZSdJCr9XTaswgGT0IEGVU1o) |
| w2v_wiki500_imdb_embed_trainable_up1 | [w2v_wiki500](https://tfhub.dev/google/Wiki-words-500/2) | 在训练数据上再训练一个轮次 | [链接](https://drive.google.com/uc?id=1oyiGXyFdCvzRlRAM_yYkdzNcqnpTEHjx) |
| w2v_wiki500_imdb_embed_trainable_up2 | [w2v_wiki500](https://tfhub.dev/google/Wiki-words-500/2) | 在训练数据上再训练两个轮次 | [链接](https://drive.google.com/uc?id=1nXU5NkOwEXe-hCx6vh2HXgvFwllAK6WF) |

我们用于训练情感分类模型的脚本包装在[model_trainer_sentiment_task.py](model_trainer_sentiment_task.py)中。

您可以在`main`函数中更改训练参数，为特定Python训练模型，然后执行以下命令开始模型训练：

```shell script
python model_trainer_sentiment_task.py
```


### 极端文本（多标签）分类

|  **数据集**  | **词袋特征维度** | **标签数量** | **训练点数量** | **测试点数量** |
|:------------:|:----------------:|:------------:|:--------------:|:--------------:|
|    RCV1-2K    | 47,236           | 2,456        | 623,847        | 155,962        |
|  EURLex-4.3K  | 200,000          | 4,271        | 45,000         | 6,000          |
| AmazonCat-13K | 203,882          | 13,330       | 1,186,239      | 306,782        |
| AmazonCat-14K | 597,540          | 14,588       | 4,398,050      | 1,099,725      |

#### 数据集介绍与预处理

我们使用的数据集托管在[极端分类库](http://manikvarma.org/downloads/XC/XMLRepository.html)上，请在那里查看我们使用的数据集的详细信息。

#### 访问数据集

我们没有预处理数据集。因此，您可以从[极端分类库](http://manikvarma.org/downloads/XC/XMLRepository.html)下载数据集，或使用[downloader](downloader.py)下载数据集。

#### 模型训练

我们用于训练模型的架构是：输入层 + 隐藏层 + 输出层。您可以通过使用我们的[model_trainer_extreme_task](model_trainer_extreme_task.py)从头开始训练模型。或者，您可以使用[downloader](downloader.py)下载我们预训练的模型。

| 模型 | 描述 | 下载链接 |
|------|------|----------|
| eurlex_4.3k_xml | 在eurlex_4.3k数据集上训练的2层全连接神经网络 | [链接](https://drive.google.com/uc?id=1L6jOpxH81feR2R84CQybHOnysZiv1JXJ) |
| rcv1x_2k_xml | 在rcv1x_2k数据集上训练的2层全连接神经网络 | [链接](https://drive.google.com/uc?id=1C6oqP20sRrGAQ3iAcT7Gd8kAarxA5eow) |
| amazoncat_13k_xml | 在amazoncat-13k数据集上训练的2层全连接神经网络 | [链接](https://drive.google.com/uc?id=1nFQFJdTcMNK91k9kISG47az2OtVLKFvH) |
| amazoncat_14k_xml | 在amazoncat-14k数据集上训练的2层全连接神经网络 | [链接](https://drive.google.com/uc?id=1pMe-09R6doxSzCt4UMpgyd9pfBwX7jGm) |

我们用于训练情感分类模型的脚本包装在[model_trainer_extreme_task.py](model_trainer_extreme_task.py)中。

您可以在`main`函数中更改训练参数，为特定Python训练模型，然后执行以下命令开始模型训练：

```shell script
python model_trainer_extreme_task.py
```


## 下载数据集和模型的工具

请在[downloader.py](downloader.py)的`main`函数中更改参数，以下载您需要的数据集/模型。然后执行以下命令：

```shell script
python downloader.py
```


## 运行去重

我们进行实验的方式是：首先从第一个模型创建块的索引。然后在第2、第3和后续模型上执行去重。
例如，我们将对以下五个不同的情感分类模型进行去重：

| 模型名称 | 模型别名 |
|---------|----------|
| w2v_wiki500_yelp_embed_nontrainable | 模型1 |
| w2v_wiki500_imdb_embed_nontrainable | 模型2 |
| w2v_wiki500_imdb_embed_trainable | 模型3 |
| w2v_wiki500_yelp_embed_trainable | 模型4 |
| w2v_wiki500_civil_comment_embed_trainable | 模型5 |

### MISTIQUE精确匹配和近似匹配
这两种方法的实现包装在`run_baseline.py`文件中。

使用两个基线运行去重时，MISTIQUE精确匹配和MISTIQUE近似匹配，您应该在[run_baseline.py](run_baseline.py)中将**method**设置为`exact_match`或`approximate_match`。

运行代码：
```shell script
python run_baseline.py
```


运行代码后，程序将保存一个**csv**文件和两个**npy**文件到磁盘。csv文件是块去重过程的跟踪历史。detector_output的npy是将发送到页面打包的中间结果。*blocks_indexer.npy*用于存储唯一块。

要使用新模型运行，您应该**复制**函数`baseline_run_on_second_model(method)`并将其修改为新模型。您可以参考以下部分附带的示例。

**注意**：第一次运行后，可以注释掉`baseline_run_on_first_model`。之后，我们将调用**Indexer.update()**操作来更新索引。

### 增强对比和提出的方法

这两种方法的实现包装在`run_dedup_w2v_classifier.py`文件中。

使用不同方法运行去重时，增强对比、不带微调的提议方法或带微调的提议方法，您应该在[run_dedup_w2v_classifier.py](run_dedup_w2v_classifier.py)中将**method**设置为`enhanced_pairwise`、`proposed_wo_finetune`或`proposed_w_finetune`。

要使用新模型运行，您应该**复制**函数`run_on_second_model(method)`并将其修改为新模型。例如，我想用模型3运行它。我应该添加以下代码：

```python
def run_on_third_model(method):
    model_path = 'models'

    # 定义块大小
    block_size_x = 10000
    block_size_y = 100

    # 创建索引器
    blocks_indexer = indexer.Indexer(block_size_x=block_size_x, block_size_y=block_size_y)
    blocks_indexer.load('blocks_indexer.npy')

    # 在w2v_wiki500_imdb_embed_trainable上运行去重
    # 加载模型
    m2 = tf.keras.models.load_model(os.path.join(model_path, 'w2v_wiki500_imdb_embed_trainable.h5'), 
                                    custom_objects={'KerasLayer':hub.KerasLayer})
    m2_ms = blocker.block_model_2d(m2, block_size_x=block_size_x, block_size_y=block_size_y)

    # 为imdb加载数据集
    imdb_x, imdb_y = model_trainer.load_dataset(dataset="imdb_reviews")
    imdb_x_train, imdb_x_test, imdb_y_train, imdb_y_test = train_test_split(
                                        imdb_x, imdb_y, test_size=0.2, random_state=0)

    if method == 'enhanced_pairwise':
        # 运行去重（增强对比）
        result_df = deduplicator.deduplicate_model(m2, m2_ms, imdb_x_test, imdb_y_test, 
                                    blocks_indexer, fp=0.01, sim=0.7, stop_acc_drop=0.035, 
                                    use_lsh=False)
    elif method == 'proposed_wo_finetune':
        # 运行去重（不带微调的提议方法）
        result_df = deduplicator.deduplicate_model(m2, m2_ms, imdb_x_test, imdb_y_test, 
                                    blocks_indexer, fp=0.01, sim=0.7, stop_acc_drop=0.035, 
                                    use_lsh=True)
    elif method == 'proposed_w_finetune':
        # 运行去重（带微调的提议方法）
        result_df = deduplicator.deduplicate_model(m2, m2_ms, imdb_x_test, imdb_y_test, 
                                    blocks_indexer, fp=0.01, sim=0.7, stop_acc_drop=0.035, 
                                    use_lsh=True, x_train=imdb_x_train, y_train=imdb_y_train)

    detector_output, num_unique_blocks = deduplicator.generate_detector_output(result_df, m2_ms, blocks_indexer.num_total)
    result_df.to_csv('imdb_embed_trainable_result.csv', index=False)
    np.save('imdb_embed_trainable_output.npy', detector_output)

    # 更新索引
    blocks_indexer.update_index(m2_ms, result_df)

    # 保存到磁盘
    blocks_indexer.save('blocks_indexer.npy')
```


运行代码：
```shell script
python run_dedup_w2v_classifier.py
```


**注意**：第一次运行后，可以注释掉`run_on_first_model`。之后，我们将调用**Indexer.update()**操作来更新索引。

运行代码后，程序将保存一个**csv**文件和两个**npy**文件到磁盘。csv文件是块去重过程的跟踪历史。detector_output的npy是将发送到页面打包的中间结果。*blocks_indexer.npy*用于存储唯一块。

### 使用不同嵌入层对情感分类模型进行去重

我们对以下异构情感分类模型执行去重。代码包装在[run_hetero_model_exp1](run_hetero_model_exp1)中。

| 模型 | 嵌入层大小 |
|:----:|:----------:|
| nnlm_en_dim_128_yelp_embed_trainable | 963812, 128 |
| nnlm_en_dim_50_imdb_embed_trainable | 963812, 50 |
| w2v_wiki250_civil_comment_embd_trainable | 1009375, 250 |
| w2v_wiki500_yelp_embed_trainable | 1009375, 500 |

### 对不同的极端分类模型进行去重

我们对以下异构极端分类模型执行去重。代码包装在[run_hetero_model_exp2](run_hetero_model_exp2)中。

| 模型 | 下载链接（输入层、隐藏层、输出层） |
|:----:|:----------------------------------:|
| eurlex_4.3k_xml | 200000, 2000, 4271 |
| rcv1x_2k_xml | 47236, 5000, 2456 |
| amazoncat_13k_xml | 203882, 1000, 1330 |
| amazoncat_14k_xml | 597540, 1000, 14588 |

### 对情感分类模型和极端分类模型对进行去重

#### nnlm128_yelp + rcv1x_2k_xml

代码包装在[run_hetero_model_exp3](run_hetero_model_exp3)中。

#### wiki500_yelp + eurlex_4.3k_xml

代码包装在[run_hetero_model_exp4](run_hetero_model_exp4)中。

#### wiki500_yelp + amazoncat_13k_xml

代码包装在[run_hetero_model_exp5](run_hetero_model_exp5)中。

### 对更新的模型进行去重

#### w2v_wiki500_imdb_embed_trainable_update1

对在imdb数据集上多训练1个轮次的模型进行去重。我们比较两种不同策略的去重结果：(1)将模型作为新模型去重 (2)只对至少有N个LSH签名发生变化的块进行去重。

代码包装在[run_update.py](run_update.py)中。

#### w2v_wiki500_imdb_embed_trainable_update2

对在imdb数据集上多训练2个轮次的模型进行去重。我们比较两种不同策略的去重结果：(1)将模型作为新模型去重 (2)只对至少有N个LSH签名发生变化的块进行去重。

代码包装在[run_update.py](run_update.py)中。

### 使用归一化进行去重

我们比较不同归一化策略的去重效果：

1. 按层归一化，按层去重
2. 按层归一化，一起去重
3. 不归一化，按层去重
4. 不归一化，一起去重

代码包装在[run_cross_layer_dedup.py](run_cross_layer_dedup.py)中。