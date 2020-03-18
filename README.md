# README

There are two main folders in our project:
`
./src\
`
`
./SupportingMaterials
`

You should run command in ./src

We decide to use Cosine Similarity Model which is in model.py. To run this model

`python model.py`

This command will generate predictions on test set and the predictinos are saved in ./SupportingMaterials/samplePrediction.txt

Here are models we tried but not decided to use:

1. RNN

   We implemented an RNN model which is in ./src/train_cad.py. To run this model, you should unzip the zip files in ./src/save/context_pretrain
   
   `python 'utils(2).py'`
   
   `python train_rnn.py --data_path ../SupportingMaterials/ --save save/context_pretrain/ --batch_size 128 --lr_ae 0.0001 --word_ckpt save/context_pretrain/ckpt_epoch30-best@0.163260.pt` 
    
2. Logistic Regression
   
   We implemented a LR model which is in ./src/lr.py. To run this model,
   
   `python lr.py`
   
   Notice: This may take 10 hours to fine-tune the doc2vec model.# ROC-Story-Prediction
