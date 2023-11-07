import os
import random
import glob
import logging
from dataclasses import dataclass

from datasets import Dataset, load_dataset
from transformers import DataCollatorWithPadding


logger = logging.getLogger(__name__)


class Processor:
    def __init__(self, tokenizer, data_args) -> None:
        self.tokenizer = tokenizer
        self.data_args = data_args
        
    def __call__(self):
        datasets = {}
        # train set
        if self.data_args.train_dir is not None:
            train_data = self.load_dataset(self.data_args.train_dir, 'train')
            
            if self.data_args.max_train_samples is not None:
                train_data = train_data.select(range(self.data_args.max_train_samples))
            
            datasets['train'] = self.process_fn(train_data)
            
        # validation set
        if self.data_args.valid_dir is not None:
            valid_data = self.load_dataset(self.data_args.valid_dir, 'train')
        
            if self.data_args.max_valid_samples is not None:
                valid_data = valid_data.select(range(self.data_args.max_valid_samples))
                
            datasets['validation'] = self.process_fn(valid_data)
            
        # validation set
        if self.data_args.test_dir is not None:
            test_data = self.load_dataset(self.data_args.test_dir, 'test')
            datasets['test'] = self.process_fn(test_data)
        
        return datasets
    
    def load_dataset(self, data_path:str=None, key:str='train') -> Dataset:
        """ Load datasets function 

        Args:
            data_path (str, optional): folder contain list of input files name. Defaults to None.
            key (str, optional): help dataloader know is train file or test file. 
                                Input file can be train/validation/test. Defaults to 'train'.

        Raises:
            Exception: _description_

        Returns:
            Datasets
        """
        if not os.path.exists(data_path):
            raise ValueError(f'Not found {data_path} path.')
        
        files = glob.glob(os.path.join(data_path, '*'))
        extention = files[0].split('.')[-1]
    
        try:
            data_file = f"{data_path}/*.{extention}"
            
            if self.data_args.streaming:
                datasets = load_dataset(
                    extention, data_files=data_file, split=key, streaming=self.data_args.streaming
                )
            else:
                datasets = load_dataset(
                    extention, data_files=data_file, split=key, num_proc=self.data_args.dataset_num_proc
                )   
                
            return datasets
        except:
            logger.info(f'Error loading dataset {data_path} with {extention} extention')
      
    def tokenize_fn(self, x:str=None, length:int=None, 
                    add_special_token:bool=False):

        return self.tokenizer(x,
                              max_length=None if length is None else length,
                              padding=False, truncation=True,
                              return_token_type_ids=False,
                              return_attention_mask=False,
                              return_special_tokens_mask=add_special_token,
                    
        )
    
    def process_fn(self, datasets:Dataset) -> Dataset:
        """ Processing tokenizer 

        Args:
            datasets (Dataset): _description_

        Returns:
            Dataset tokenized
        """
        
        if self.data_args.streaming:
            datasets = datasets.map(
                lambda example : self.group_qp_fn(example),
                remove_columns=['negatives', 'positive'],
            )
        else:
            datasets = datasets.map(
                lambda example : self.group_qp_fn(example),
                num_proc=self.data_args.dataset_num_proc,
                remove_columns=['negatives', 'positive'],
                desc = 'Group sample process'
            )
        
        return datasets
    
    def group_qp_fn(self, example):
        # query
        query_tokenized = self.tokenize_fn(example['query'], 
                                        length=self.data_args.q_max_len)
        
        group_psg = []
        # positive passages
        if self.data_args.positive_passage_no_shuffle:
            pos_psg = example['positive']
        else: 
            if isinstance(example['positive'], list):
                pos_psg = [x for x in example['positive']]
                random.shuffle(pos_psg)
                pos_psg = pos_psg[0]
            else:
                pos_psg = example['positive']
        group_psg.append(pos_psg)

        # negative passages
        # 1 for positive psg and the left one for negative number
        max_negative_samples = self.data_args.train_n_passages - 1 
        if self.data_args.train_n_passages == 1: # no negative samples
            negs = []
        elif len(example['negatives']) < max_negative_samples:
            negs = random.choices(example['negatives'], k=max_negative_samples)
        elif self.data_args.negative_passage_no_shuffle:
            negs = example['negatives']
        else:
            negs = [x for x in example['negatives']]
            random.shuffle(negs)
            negs = negs[:max_negative_samples]
            
        group_psg.extend(negs)

        group_psg_tokenized = self.tokenize_fn(group_psg, length=self.data_args.p_max_len)
        
        return {'query': query_tokenized, 'passages': group_psg_tokenized}
    
    
@dataclass
class QPCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg] and add padding.
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    q_max_len: int = 512
    p_max_len: int = 512

    def __call__(self, features):

        qq = [{'input_ids' : f['query']["input_ids"]} for f in features]
        pp = [{'input_ids' : f} for example in features for f in example['passages']["input_ids"]]
    
        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(pp[0], list):
            pp = sum(pp, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding = 'max_length',
            max_length=self.q_max_len,
            return_tensors="pt",
        )
        p_collated = self.tokenizer.pad(
            pp,
            padding= 'max_length',
            max_length=self.p_max_len,
            return_tensors="pt",
        )
                
        return q_collated, p_collated