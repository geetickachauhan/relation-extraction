To replicate experiments, you need to first pre-process the data.

1. Data-Preprocessing contains jupyter notebooks to pre-process all the datasets. Each dataset contains an _original.ipynb notebook which handles the conversion of raw data formats to csv files for easy processing. This is when the converter stage of the pipeline is used. The other notebook performs the pre-processing stage in the pipeline to generate the entity blinded, ner blinded etc versions of the data. 

2. The above only generates train and test data. In order to generate cross validated data, utilize the Data-Splitting folder which individually generates the pickled folds for each dataset. 

3. Contextualized Embeddings folder generates the BERT and ELMo embeddings themselves. The process of generating ELMo embeddings is straightforward and highlighted in the respective notebook. However, the process of generating BERT-embeddings is less straightforward. From the pytorch-pretrained-BERT model, go to examples/extract_features.py and follow instructions similar to those provided on the tensorflow version of the website. In short, this is the command you will want to run: 
```
python extract_features.py \
  --input_file=/tmp/input.txt \
  --output_file=/tmp/output.jsonl \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=128 \
  --batch_size=8 
```
In order to make life easier, I also include another markdown file known as commands.md which provides the exact commands to run for generating BERT embeddings for each dataset, as well as the execution commands to run for replication of experiments.

4. The other folders present here are mostly analysis folders. They do not help with running of experiments
   for replication purposes.
