CUDA_VISIBLE_DEVICES=2 python main.py --cross_validate --dataset=i2b2 --border_size=-1 --num_epoches=150 --lr_values 0.001 0.0001 0.00001 --lr_boundaries 60 120 (previously this had border size 50)
CUDA_VISIBLE_DEVICES=2 python main.py --dataset=ddi --border_size=-1 --num_epoches=100 --lr_values 0.001 0.0001 --lr_boundaries 60 --cross_validate
CUDA_VISIBLE_DEVICES=3 python main.py --use_test --dataset=semeval2010 --use_elmo


Commands after running random search: (need to check if these early stops were even used)
CUDA_VISIBLE_DEVICES=3 python main.py --dataset=i2b2 --border_size=-1 --num_epoches=203 --lr_values 0.001 0.00014 --lr_boundaries 101 --filter_sizes=3,4,5 --batch_size=81 --early_stop --patience=40 --cross_validate
Yes early stop was employed and patience was exceeded at 86 epoches. 
CUDA_VISIBLE_DEVICES=3 python main.py --dataset=ddi --border_size=-1 --num_epoches=124 --lr_values 0.001 0.00021 --lr_boundaries 62 --filter_sizes=3,4,5 --batch_size=37 --early_stop --patience=24 --cross_validate
Yes early stop was employed and patience was exceeded at 76 epoches
CUDA_VISIBLE_DEVICES=3 python main.py --dataset=semeval2010 --border_size=1 --num_epoches=283 --lr_values 0.001 0.0005 --lr_boundaries 141 --filter_sizes=3,4,5,6 --batch_size=88 --use_test













Generating BERT for Semeval 2010
export BERT_BASE_DIR=/data/scratch-oc40/geeticka/data/relation_extraction/bert/bert-general/uncased_L-24_H-1024_A-16

export IO_DIR=/data/scratch-oc40/geeticka/data/relation_extraction/semeval2010/pre-processed/original/bert

For tensorflow version: Getting a cuda error in the conda environment that I have bert_embeddings in scratch/conda_envs
python extract_features.py \
  --input_file=$IO_DIR/input-sentences/train_original_border_50.txt \
  --output_file=$IO_DIR/train_original_border_50.json \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=90 \
  --batch_size=8


For pytorch version: (inside the pytorch-pretrained-BERT/examples)

CUDA_VISIBLE_DEVICES=2 python extract_features.py \
  --input_file=$IO_DIR/input-sentences/test_original_border_50.txt \
  --output_file=$IO_DIR/test_original_border_50.json \
--cache_dir=/crimea/geeticka/data/ \
  --bert_model=bert-large-uncased \
  --do_lower_case \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=90 \
  --batch_size=8

Need to convert to a pytorch checkpoint to load from a specific local model
python convert_tf_checkpoint_to_pytorch.py \
--tf_checkpoint_path=$BERT_BASE_DIR/biobert_model.ckpt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--pytorch_dump_path=$BERT_BASE_DIR/pytorch_model.bin


export BERT_BASE_DIR=/crimea/geeticka/data/relation_extraction/bert/bert-biomed/pubmed_pmc_470k
In order to convert BioBERT, needed to do a workaround in the form of https://github.com/dmis-lab/biobert/issues/2
Adding in this part to the modeling.py file that converter uses. 



Need to update the following parameters: max_seq_length (different for different datasets)
Input file, output file, and the export paths 

For i2b2:
To generate the weights from a link to a pytorch bin model
export BERT_BASE_DIR=/data/scratch-oc40/geeticka/data/relation_extraction/bert/bert-clinical-notes/biobert_pretrain_output_all_notes_150000

export IO_DIR=/data/scratch-oc40/geeticka/data/relation_extraction/i2b2/pre-processed/original/bert-CLS

CUDA_VISIBLE_DEVICES=3 python extract_features.py \
  --input_file=$IO_DIR/input-sentences/test_original_border_-1.txt \
  --output_file=$IO_DIR/test_original_border_-1.json \
--cache_dir=/crimea/geeticka/data/ \
  --bert_model=$BERT_BASE_DIR/ \
  --do_lower_case \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=204 \
  --batch_size=8

For DDI: 
To generate the weights from a link to a pytorch bin model
export BERT_BASE_DIR=/data/scratch-oc40/geeticka/data/relation_extraction/bert/bert-biomed/pubmed_pmc_470k

export IO_DIR=/data/scratch-oc40/geeticka/data/relation_extraction/ddi/pre-processed/original/bert-CLS

CUDA_VISIBLE_DEVICES=0 python extract_features.py \
  --input_file=$IO_DIR/input-sentences/test_original_border_-1.txt \
  --output_file=$IO_DIR/test_original_border_-1.json \
--cache_dir=/crimea/geeticka/data/ \
  --bert_model=$BERT_BASE_DIR/ \
  --do_lower_case \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=131 \
  --batch_size=8


