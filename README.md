# differential-subset-pruning
This repository is accompany with the paper: [Differential Subset Pruning of Transformer Heads]().

## Dependencies
- python 3.7.4
- perl 5.18.4
- pytorch 1.7.1+cu101

## BERT on MNLI
Install the Transformer library adapted from [Huggingface](https://github.com/huggingface/transformers):
```
cd transformers
pip install -e .
```
Then install other dependencies:
```
cd examples
pip install -r requirements.txt
```
Download the MNLI dataset:
```
cd pruning
python ../../utils/download_glue_data.py --data_dir data/ --task MNLI
```
### Joint Pruning
#### Joint DSP
```
export TASK_NAME=MNLI
python run_dsp.py \
	--data_dir data/$TASK_NAME/ \
	--model_name_or_path bert-base-uncased \
	--task_name $TASK_NAME \
	--do_train \
	--do_eval \
	--max_seq_length 128 \
	--per_device_train_batch_size 32 \
	--learning_rate 2e-5 \
	--num_train_epochs 3.0 \
	--output_dir dsp_output/ \
	--cache_dir cache/ \
	--save_steps 10000 \
	--num_of_heads 12 \
	--pruning_lr 0.5 \
	--annealing \
	--initial_temperature 1000 \
	--final_temperature 1e-8 \
	--cooldown_steps 25000 \
	--joint_pruning 
```
where `num_of_heads` is the number of attention heads you want to keep unpruned, `pruning_lr`, `initial_temperature`, `final_temperature`, and `cooldown_steps` are the hyperparameters for DSP. 

You can also use straight-through estimator:
```
export TASK_NAME=MNLI
python run_dsp.py \
	--data_dir data/$TASK_NAME/ \
	--model_name_or_path bert-base-uncased \
	--task_name $TASK_NAME \
	--do_train \
	--do_eval \
	--max_seq_length 128 \
	--per_device_train_batch_size 32 \
	--learning_rate 2e-5 \
	--num_train_epochs 3.0 \
	--output_dir dsp_output/ \
	--cache_dir cache/ \
	--save_steps 10000 \
	--num_of_heads 12 \
	--pruning_lr 0.5 \
    --use_ste \
	--joint_pruning 
```
#### Voita et al. 
The baseline of [Voita et al., 2019](https://www.aclweb.org/anthology/P19-1580) can be reproduced by
```
export TASK_NAME=MNLI
python run_voita.py \
	--model_name_or_path bert-base-uncased \
	--task_name $TASK_NAME \
	--do_train \
	--do_eval \
	--data_dir data/$TASK_NAME/ \
	--max_seq_length 128 \
	--per_device_train_batch_size 32 \
	--learning_rate 2e-5 \
	--num_train_epochs 3.0 \
	--output_dir voita_output/\
	--cache_dir cache/ \
	--save_steps 10000 \
	--l0_penalty 0.0015 \
	--joint_pruning \
	--pruning_lr 0.1 
```
where `l0_penalty` is used to indirectly control the number of heads.
### Pipelined Pruning
We need to fine tune BERT on MNLI first:
```
export TASK_NAME=MNLI
python ../text-classification/run_glue.py \
	--model_name_or_path bert-base-uncased \
	--task_name $TASK_NAME \
	--do_train \
	--do_eval \
	--data_dir data/$TASK_NAME/ \
	--max_seq_length 128 \
	--per_device_train_batch_size 32 \
	--learning_rate 2e-5 \
	--num_train_epochs 3.0 \
	--output_dir output/ \
	--cache_dir cache/ \
	--save_steps 10000 
```
#### Pipelined DSP
```
export TASK_NAME=MNLI
python run_dsp.py \
	--data_dir data/$TASK_NAME/ \
	--model_name_or_path output/ \
	--task_name $TASK_NAME \
	--do_train \
	--do_eval \
	--max_seq_length 128 \
	--per_device_train_batch_size 32 \
	--learning_rate 2e-5 \
	--num_train_epochs 1.0 \
	--output_dir dsp_output/pipelined/ \
	--cache_dir cache/ \
	--save_steps 10000 \
	--num_of_heads 12 \
	--pruning_lr 0.5 \
	--annealing \
	--initial_temperature 1000 \
	--final_temperature 1e-8 \
	--cooldown_steps 8333 
```
#### Michel et al.
The baseline of [Michel et al., 2019](https://arxiv.org/abs/1905.10650) can be reproduced by
```
export TASK_NAME=MNLI
python run_michel.py \
	--data_dir data/$TASK_NAME/ \
	--model_name_or_path output/ \
	--task_name $TASK_NAME \
	--output_dir michel_output/ \
	--cache_dir cache/ \
	--tokenizer_name bert-base-uncased \
	--exact_pruning \
    --num_of_heads 12 
```
where `num_of_heads` is the number of heads to prune (mask) at each step.
## Enc-Dec on IWSLT
Install the Transformer library adapted from [Fairseq](https://github.com/pytorch/fairseq):
```
cd fairseq
pip install -e .
```
Download and prepare the IWSLT dataset:
```
cd examples/pruning
./prepare-iwslt.sh
```
Preprocess the dataset:
```
export TEXT=iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/\
    --workers 20
```
### Joint Training
#### Joint DSP
```
export SAVE_DIR=dsp_checkpoints/

python run_dsp.py \
    data-bin/ \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-epoch 60 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir $SAVE_DIR/ \
    --num-of-heads 4 \
    --pruning-lr 0.2 \
    --joint-pruning \
    --annealing \
    --initial-temperature 0.1 \
    --final-temperature 1e-8 \
    --cooldown-steps 15000 

python generate_dsp.py \
	data-bin \
 	-s de -t en \
	--path $SAVE_DIR/checkpoint_best.pt \
	--quiet \
	--batch-size 32 --beam 5 --remove-bpe
```
where `num_of_heads` is the number of attention heads you want to keep unpruned, `pruning-lr`, `initial-temperature`, `final-temperature`, and `cooldown-steps` are the hyperparameters for DSP. 

You can also use straight-through estimator:
```
export SAVE_DIR=dsp_checkpoints/

python run_dsp.py \
    data-bin/ \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-epoch 60 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir $SAVE_DIR/ \
    --num-of-heads 4 \
    --pruning-lr 0.2 \
    --joint-pruning \
    --use-ste 

python generate_dsp.py \
	data-bin \
 	-s de -t en \
	--path $SAVE_DIR/checkpoint_best.pt \
	--quiet \
	--batch-size 32 --beam 5 --remove-bpe
```

#### Voita et al.
The baseline of [Voita et al., 2019](https://www.aclweb.org/anthology/P19-1580) can be reproduced by
```
export SAVE_DIR=voita_checkpoints/

python run_voita.py \
		data-bin/ \
		--arch transformer_iwslt_de_en --share-decoder-input-output-embed \
		--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
		--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
		--dropout 0.3 --weight-decay 0.0001 \
		--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
		--max-tokens 4096 \
		--max-epoch 60 \
		--keep-best-checkpoints 1 \
		--l0-penalty 200 \
		--save-dir $SAVE_DIR/ 

python generate_voita.py \
	data-bin \
	-s de -t en \
	--path $SAVE_DIR/checkpoint_last.pt \
	--quiet \
	--batch-size 32 --beam 5 --remove-bpe
```
where `l0_penalty` is used to indirectly control the number of heads.
### Pipelined Pruning
We need to fine tune the model on IWSLT first:
```
fairseq-train \
    data-bin/ \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-epoch 60 \
    --save-dir checkpoints/ 
```
#### Pipelined DSP
```
export SAVE_DIR=dropout_checkpoints/pipelined/

python run_dsp.py \
    data-bin/ \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --reset-optimizer \
    --restore-file checkpoints/checkpoint_last.pt \
    --max-epoch 61 \
    --save-dir $SAVE_DIR/ \
    --num-of-heads 4 \
    --pruning-lr 0.2 \
    --annealing \
    --initial-temperature 0.1 \
    --final-temperature 1e-8 \
    --cooldown-steps 250 

python generate_dsp.py \
	data-bin \
 	-s de -t en \
	--path $SAVE_DIR/checkpoint_last.pt \
	--quiet \
	--batch-size 32 --beam 5 --remove-bpe
```
#### Michel et al.
The baseline of [Michel et al., 2019](https://arxiv.org/abs/1905.10650) can be reproduced by
```
python run_michel.py \
	data-bin \
 	-s de -t en \
	-a transformer_iwslt_de_en \
 	--restore-file checkpoints/checkpoint_last.pt \
	--optimizer adam --adam-betas '(0.9,0.98)' \
	--reset-optimizer \
	--batch-size 32 --beam 5 --remove-bpe \
	--num-of-heads 4 \
	--exact-pruning 
```
where `num_of_heads` is the number of heads to prune (mask) at each step.