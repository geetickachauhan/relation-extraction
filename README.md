# relation-extraction
How to run (from within scripts folder): python main.py --cross\_validate --dataset=ddi 

Relation Extraction with Semeval 2010 data, i2b2 2010 VA challenge classification data and DDI extraction data.

For Semeval 2010 task 8, evaluation is done based on macro F1 of all classes but not considering "Other"
For the DDI Extraction task, evaluation is done based on macro F1 of all classes (strict evaluation) as well
as macro F1 for relation detection (loose evaluation). We print macro F1 of all classes (5 way with none),
macro F1 of non 'none' classes and macro F1 of the detection. 

There is a hierarchy for storing your files. The suggested format is:
1. In a main folder called relation\_extraction, store the 
The framework:
1. Converters are for converting the data into the original format. Refer to the jupyter notebooks in the
   Data-Preprocessing folder and the ones marked with \_original to know how to use the converter. 
2. Preprocessing is done on top of the original dataframes created. 
