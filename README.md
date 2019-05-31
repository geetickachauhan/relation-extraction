# REflex: Flexible Framework for Relation Extraction in Multiple Domains


![Framework](https://github.com/geetickachauhan/relation-extraction/blob/master/img/framework.png)

REflex is a unifying framework for Relation Extraction, applied on 3 highly used datasets (from the general, biomedical and clinical domains), with the ability to be extendable to new datasets.

REflex has experimental as well as design goals. The experimental goals are in identification of sources of variability in results for the 3 datasets and provide the field with a strong baseline model to compare against for future improvements. The design goals are in identification of best practices for relation extraction and to be a guide for approaching new datasets.

In order to replicate experiments for this work, generate the data beyond the pre-processing stage by going into the notebooks/ folder and following the README.md instructions there. 

The hierarchy of this code is organized as follows:
1. relation_extraction stores the main components of the framework, including converters, pre-processing module and models
2. eval/ contains the evaluation scripts used to evaluate the model
3. scripts/ which contains the scripts to run the model

Refer to the jupyter notebooks in the notebooks/Data-Preprocessing folder and the ones marked with _original to know how to use the converter.

In order to run the model, cd into the scripts/ folder and type ```python main.py --cross_validate --dataset=ddi```

Relation Extraction with Semeval 2010 data, i2b2 2010 VA challenge classification data and DDI extraction data.

For Semeval 2010 task 8, evaluation is done based on macro F1 of all classes but not considering "Other"
For the DDI Extraction task, evaluation is done based on macro F1 of all classes (strict evaluation) as well
as macro F1 for relation detection (loose evaluation). We print macro F1 of all classes (5 way with none),
macro F1 of non 'none' classes and macro F1 of the detection. 
