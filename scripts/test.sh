#CUDA_VISIBLE_DEVICES=0 python -m src.main set.features=true 
#CUDA_VISIBLE_DEVICES=1 python -m src.main set.train=true
CUDA_VISIBLE_DEVICES=2 python -m src.main set.test=true
CUDA_VISIBLE_DEVICES=2 python -m src.post_proc -val_path=/home/ydc/DACSE2021/task5/data/Evaluation_Set/ -evaluation_file=/home/ydc/DACSE2021/sed-tim-base/workplace/Eval_out_tim_test.csv -new_evaluation_file=new_eval_output_test.csv
