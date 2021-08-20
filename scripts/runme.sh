#CUDA_VISIBLE_DEVICES=0 python -m src.main set.features=true 
#CUDA_VISIBLE_DEVICES=1 python -m src.main set.train=true
CUDA_VISIBLE_DEVICES=0 python -m src.main set.eval=true
CUDA_VISIBLE_DEVICES=0 python -m src.post_proc -val_path=/home/yzj/data/Development_Set/Validation_Set/ -evaluation_file=/home/ydc/DACSE2021/sed-tim-base/workplace/Eval_out_tim.csv -new_evaluation_file=new_eval_output2.csv
