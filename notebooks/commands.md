Default commands:
1. ```CUDA\_VISIBLE\_DEVICES=2 python main.py --cross\_validate --dataset=i2b2 --border\_size=-1 --num\_epoches=150 --lr\_values 0.001 0.0001 0.00001 --lr\_boundaries 60 120 (previously this had border size 50)```
2. ```CUDA\_VISIBLE\_DEVICES=2 python main.py --dataset=ddi --border\_size=-1 --num\_epoches=100 --lr\_values 0.001 0.0001 --lr\_boundaries 60 --cross\_validate```
3. ```CUDA\_VISIBLE\_DEVICES=3 python main.py --use\_test --dataset=semeval2010 --use\_elmo```


Commands after running random search: (need to check if these early stops were even used)
1. ```CUDA\_VISIBLE\_DEVICES=3 python main.py --dataset=i2b2 --border\_size=-1 --num\_epoches=203 --lr\_values 0.001 0.00014 --lr\_boundaries 101 --filter\_sizes=3,4,5 --batch\_size=81 --early\_stop --patience=40 --cross\_validate```
Yes early stop was employed and patience was exceeded at 86 epoches. 
2. ```CUDA\_VISIBLE\_DEVICES=3 python main.py --dataset=ddi --border\_size=-1 --num\_epoches=124 --lr\_values 0.001 0.00021 --lr\_boundaries 62 --filter\_sizes=3,4,5 --batch\_size=37 --early\_stop --patience=24 --cross\_validate```
Yes early stop was employed and patience was exceeded at 76 epoches
3. ```CUDA\_VISIBLE\_DEVICES=3 python main.py --dataset=semeval2010 --border\_size=1 --num\_epoches=283 --lr\_values 0.001 0.0005 --lr\_boundaries 141 --filter\_sizes=3,4,5,6 --batch\_size=88 --use\_test```



Generating BERT for Semeval 2010
```export BERT\_BASE\_DIR=/data/scratch-oc40/geeticka/data/relation\_extraction/bert/bert-general/uncased\_L-24\_H-1024\_A-16```

```export IO\_DIR=/data/scratch-oc40/geeticka/data/relation\_extraction/semeval2010/pre-processed/original/bert```

For tensorflow version: Getting a cuda error in the conda environment that I have bert\_embeddings in scratch/conda\_envs
```
python extract\_features.py \
  --input\_file=$IO\_DIR/input-sentences/train\_original\_border\_50.txt \
  --output\_file=$IO\_DIR/train\_original\_border\_50.json \
  --vocab\_file=$BERT\_BASE\_DIR/vocab.txt \
  --bert\_config\_file=$BERT\_BASE\_DIR/bert\_config.json \
  --init\_checkpoint=$BERT\_BASE\_DIR/bert\_model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max\_seq\_length=90 \
  --batch\_size=8
```


For pytorch version: (inside the pytorch-pretrained-BERT/examples)

```
CUDA\_VISIBLE\_DEVICES=2 python extract\_features.py \
  --input\_file=$IO\_DIR/input-sentences/test\_original\_border\_50.txt \
  --output\_file=$IO\_DIR/test\_original\_border\_50.json \
--cache\_dir=/crimea/geeticka/data/ \
  --bert\_model=bert-large-uncased \
  --do\_lower\_case \
  --layers=-1,-2,-3,-4 \
  --max\_seq\_length=90 \
  --batch\_size=8
```

Need to convert to a pytorch checkpoint to load from a specific local model
```
python convert\_tf\_checkpoint\_to\_pytorch.py \
--tf\_checkpoint\_path=$BERT\_BASE\_DIR/biobert\_model.ckpt \
--bert\_config\_file=$BERT\_BASE\_DIR/bert\_config.json \
--pytorch\_dump\_path=$BERT\_BASE\_DIR/pytorch\_model.bin
```


```export BERT\_BASE\_DIR=/crimea/geeticka/data/relation\_extraction/bert/bert-biomed/pubmed\_pmc\_470k```
In order to convert BioBERT, needed to do a workaround in the form of ```https://github.com/dmis-lab/biobert/issues/2```
Adding in this part to the modeling.py file that converter uses. 



Need to update the following parameters: max\_seq\_length (different for different datasets)
Input file, output file, and the export paths 

For i2b2:
To generate the weights from a link to a pytorch bin model
```export BERT\_BASE\_DIR=/data/scratch-oc40/geeticka/data/relation\_extraction/bert/bert-clinical-notes/biobert\_pretrain\_output\_all\_notes\_150000```

```export IO\_DIR=/data/scratch-oc40/geeticka/data/relation\_extraction/i2b2/pre-processed/original/bert-CLS```

```
CUDA\_VISIBLE\_DEVICES=3 python extract\_features.py \
  --input\_file=$IO\_DIR/input-sentences/test\_original\_border\_-1.txt \
  --output\_file=$IO\_DIR/test\_original\_border\_-1.json \
--cache\_dir=/crimea/geeticka/data/ \
  --bert\_model=$BERT\_BASE\_DIR/ \
  --do\_lower\_case \
  --layers=-1,-2,-3,-4 \
  --max\_seq\_length=204 \
  --batch\_size=8
```

For DDI: 
To generate the weights from a link to a pytorch bin model
```export BERT\_BASE\_DIR=/data/scratch-oc40/geeticka/data/relation\_extraction/bert/bert-biomed/pubmed\_pmc\_470k```

```export IO\_DIR=/data/scratch-oc40/geeticka/data/relation\_extraction/ddi/pre-processed/original/bert-CLS```

```
CUDA\_VISIBLE\_DEVICES=0 python extract\_features.py \
  --input\_file=$IO\_DIR/input-sentences/test\_original\_border\_-1.txt \
  --output\_file=$IO\_DIR/test\_original\_border\_-1.json \
--cache\_dir=/crimea/geeticka/data/ \
  --bert\_model=$BERT\_BASE\_DIR/ \
  --do\_lower\_case \
  --layers=-1,-2,-3,-4 \
  --max\_seq\_length=131 \
  --batch\_size=8
```


