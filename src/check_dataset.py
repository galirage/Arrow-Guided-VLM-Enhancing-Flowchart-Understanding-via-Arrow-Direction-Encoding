import os
import json
import argparse

def parser():
    parser = argparse.ArgumentParser(description='lllm_args')
    parser.add_argument('--process_name', '-pn', type=str, default='load_pdf', help='process name')
    parser.add_argument('--dataset_dir', '-dsd', type=str, default='../dataset/DocVQA/', help='pdf file to load')
    parser.add_argument('--output_dir', '-od', type=str, default='output/', help='path to output directory')

    return parser.parse_args()


def check_DocVQA_dataset(args):
    # check annotations
    with open(os.path.join(args.dataset_dir, 'qas/train.json'), 'r') as f:
        data = json.load(f)
    # print("type(data), ", type(data)) # dict
    # print("data.keys(), ", data.keys()) # dict_keys(['dataset_name', 'dataset_version', 'dataset_split', 'data'])
    # print("type(data['data']), ", type(data['data'])) # list
    # print("len(data['data']), ", len(data['data'])) # 36230
    # print("type(data['data'][0]), ", type(data['data'][0])) # dict
    # print("data['data'][0].keys(), ", data['data'][0].keys()) # dict_keys(['questionId', 'question', 'doc_id', 'page_ids', 'answers', 'answer_page_idx', 'data_split'])
    # print("data['data'][0]['question'], ", data['data'][0]['question']) # what is the date mentioned in this letter?
    # print("data['data'][0]['answers'], ", data['data'][0]['answers']) # ['1/8/93']
    # print("type(data['data'][0]['answers']), ", type(data['data'][0]['answers'])) # list
    # print("data['data'][0]['answers'][0], ",data['data'][0]['answers'][0]) # 

    for data1 in data['data']:
        question = data1['question']
        answers = data1['answers']
        # print("question: ", question)
        # print("answers: ", answers)
        if 'flow' in question:
            print("question: ", question)
        for answer1 in answers:
            if 'flow' in answer1:
                print("answer: ", answer1)


if __name__ == '__main__':
    """
    usage)
    python check_dataset.py --process_name check_docvqa --dataset_dir ../DocVQA/
    """
    args = parser()
    if args.process_name == 'check_docvqa':
        """
        check DocVQA Dataset
        https://www.docvqa.org/datasets
        """
        check_DocVQA_dataset(args)
    



