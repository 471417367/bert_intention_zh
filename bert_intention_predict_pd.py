# -*- coding:utf-8 -*-
from bert import tokenization
import tensorflow as tf
import numpy as np

bert_config_file = '/opt/chinese_L-12_H-768_A-12/bert_config.json'
max_seq_length = 128


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def convert_single_example(example, label_map, max_seq_length, tokenizer):
    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    label_ids = label_map[example.label]

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids)
    return feature


def main(_):
    # /my_model/ 文件夹下的模型名字
    predict_fn = tf.contrib.predictor.from_saved_model('/opt/my_model/intention_model')

    label_list = ['受理时效', '地点', '材料', '余额', '其他']
    label_map = {}
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    while True:
        query = input("输入你的问句> ")
        predict_example = InputExample(guid='text-0', text_a=query, label='其他')
        feature = convert_single_example(predict_example, label_map, max_seq_length, tokenizer)

        prediction = predict_fn({
            "input_ids": [feature.input_ids],
            "input_mask": [feature.input_mask],
            "segment_ids": [feature.segment_ids],
            "label_ids": [feature.label_ids]
        })
        # 最好print(prediction),根据返回结果自己处理一下。
        index = np.argmax(prediction['probabilities'][0])
        print('{}属于：{}，得分：{}'.format(query, label_list[int(index)], np.max(prediction['probabilities'][0])))


if __name__ == "__main__":
    tf.app.run()
