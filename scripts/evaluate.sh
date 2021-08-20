# file_path = /home/ydc/DACSE2021/task5/sed-tim/workplace/Eval_out_tim.csv
# refer_path = /home/ydc/DACSE2021/task5/data/Development_Set/Validation_Set/
# save_path = ./dict/
python -m evaluation_metrics.evaluation -pred_file=/home/ydc/DACSE2021/sed-tim-base/new_eval_output2.csv -ref_files_path=/home/ydc/DACSE2021/task5/data/Development_Set/Validation_Set/ -team_name=PKU_ADSP -dataset=VAL -savepath=/home/ydc/DACSE2021/sed-tim-base/evaluation_metrics/dict/