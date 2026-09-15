sacct --format=JobID,JobName,State,ExitCode,TimeLimit,Elapsed -S 2026-08-02
python hpc_runner_dl.py --model cnn --rs 42 --data_path dummy_test.csv

# Xóa vĩnh viễn

chmod -R 777 ~/ML_Project

rm -rf ~/ML_Project

1@Newdayyyyyyyy


python code/hpc_runner_dl_ts.py --model lstm --rs 1 --data_path data/coffee.csv

du -sh * | sort -hr
 
scp -r ./ enucc:~/ML_Project/
 
scp code/hpc_runner_dl_testing.py code/nested_cv_dl_testing.py code/submit_file/cnn_coffee.sh ext.login.enucc:~/ML_Project/src

cd C:\Users\40774065\Documents\ML\
 
upload
 


pip install xgboost catboost
 

 
 
scp -r ext.login.enucc:/users/40774065/ML_Project/src/hpc_runner.py /Users/mmm/Research/code



scp -r ext.login.enucc:/users/40774065/ML_Project/src/results/vinegar/ /Users/mmm/Research/results/

scp -r ext.login.enucc:/users/40774065/ML_Project/src/results/coffee/ /Users/mmm/Research/results/

scp -r ext.login.enucc:/users/40774065/ML_Project/src/results/plant_oil /Users/mmm/Research/results/

scp -r ext.login.enucc:/users/40774065/ML_Project/src/results/cacao_butter /Users/mmm/Research/results/

scp -r ext.login.enucc:/users/40774065/ML_Project/src/results/chinese_wine /Users/mmm/Research/results
scp -r ext.login.enucc:/users/40774065/ML_Project/src/results/wine_spoilage /Users/mmm/Research/results/

scp -r ext.login.enucc:/users/40774065/ML_Project/src/results/pork_adulteration_1 /Users/mmm/Research/results/


scp -r ext.login.enucc:/users/40774065/ML_Project/src/results/cuttlefish /Users/mmm/Research/results/

scp -r ext.login.enucc:/users/40774065/ML_Project/src/results/nemipterus_nemathophorus /Users/mmm/Research/results/

scp -r ext.login.enucc:/users/40774065/ML_Project/src/results/octopus /Users/mmm/Research/results/
scp -r ext.login.enucc:/users/40774065/ML_Project/src/results/squid /Users/mmm/Research/results/



scp -r enucc:/users/40774065/ML_Project/src/results/vinegar/node /Users/mmm/Research/results/vinegar

scp -r enucc:/users/40774065/ML_Project/src/results/coffee/node /Users/mmm/Research/results/coffee

scp -r enucc:/users/40774065/ML_Project/src/results/plant_oil/node /Users/mmm/Research/results/plant_oil

scp -r enucc:/users/40774065/ML_Project/src/results/cacao_butter/node /Users/mmm/Research/results/cacao_butter

scp -r enucc:/users/40774065/ML_Project/src/results/chinese_wine/node /Users/mmm/Research/results/chinese_wine

scp -r enucc:/users/40774065/ML_Project/src/results/wine_spoilage/node /Users/mmm/Research/results/wine_spoilage




scp -r enucc:/users/40774065/ML_Project/src/results/pork_adulteration_1/node /Users/mmm/Research/results/pork_adulteration

scp -r enucc:/users/40774065/ML_Project/src/results/cuttlefish/node /Users/mmm/Research/results/cuttlefish

scp -r enucc:/users/40774065/ML_Project/src/results/nemipterus_nemathophorus/node /Users/mmm/Research/results/nemipterus

scp -r enucc:/users/40774065/ML_Project/src/results/octopus/node /Users/mmm/Research/results/octopus

scp -r enucc:/users/40774065/ML_Project/src/results/squid/node /Users/mmm/Research/results/squid

scp login.enucc:/users/40774065/ML_Project/src/cat_ts* login.enucc:/users/40774065/ML_Project/src/ada* login.enucc:/users/40774065/ML_Project/src/gb_ts* login.enucc:/users/40774065/ML_Project/src/gru_ts* login.enucc:/users/40774065/ML_Project/src/knn_ts* login.enucc:/users/40774065/ML_Project/src/l_svc_ts* login.enucc:/users/40774065/ML_Project/src/lstm_ts* login.enucc:/users/40774065/ML_Project/src/mlp_ts* login.enucc:/users/40774065/ML_Project/src/node_ts* login.enucc:/users/40774065/ML_Project/src/rf_ts* login.enucc:/users/40774065/ML_Project/src/saint_ts* login.enucc:/users/40774065/ML_Project/src/svc_ts* login.enucc:/users/40774065/ML_Project/src/tcn_ts* login.enucc:/users/40774065/ML_Project/src/xgb_ts* /Users/mmm/Research/code/submit_code



scp -r enucc:/users/40774065/ML_Project/src/results/vinegar/ /Users/mmm/Research/results/

scp -r enucc:/users/40774065/ML_Project/src/results/coffee/ /Users/mmm/Research/results/

scp -r enucc:/users/40774065/ML_Project/src/results/plant_oil /Users/mmm/Research/results/

scp -r enucc:/users/40774065/ML_Project/src/results/cacao_butter /Users/mmm/Research/results/

scp -r enucc:/users/40774065/ML_Project/src/results/chinese_wine /Users/mmm/Research/results/

scp -r enucc:/users/40774065/ML_Project/src/results/wine_spoilage /Users/mmm/Research/results/

scp -r enucc:/users/40774065/ML_Project/src/results/pork_adulteration_1 /Users/mmm/Research/results/


scp -r enucc:/users/40774065/ML_Project/src/results/cuttlefish /Users/mmm/Research/results/

scp -r enucc:/users/40774065/ML_Project/src/results/nemipterus_nemathophorus /Users/mmm/Research/results/

scp -r enucc:/users/40774065/ML_Project/src/results/octopus /Users/mmm/Research/results/

scp -r enucc:/users/40774065/ML_Project/src/results/squid /Users/mmm/Research/results/


*3
sbatch node_nemi.sh -1
Submitted batch job 153731
sbatch node_squid.sh -1
Submitted batch job 153732
2-5
sbatch node_nemi.sh
Submitted batch job 153876
(base) [40774065@login1[enucc] src]$ sbatch node_squid.sh
Submitted batch job 153877

1
sbatch node_coffee.sh
Submitted batch job 153817
sbatch node_plant_oil.sh
Submitted batch job 153818
sbatch node_cacao_butter.sh
Submitted batch job 153819

sbatch node_chinese_wine.sh
Submitted batch job 153820
sbatch node_cuttlefish.sh
Submitted batch job 153878 1-5
sbatch node_pork.sh
Submitted batch job 153823
sbatch node_octopus.sh
Submitted batch job 153824

=====
vinegar and wine_spoilage
======
1-5
sbatch saint_cacao_butter.sh
Submitted batch job 153825
sbatch saint_nemi.sh 1-5
Submitted batch job 153827 
batch saint_coffee.sh
Submitted batch job 153828
sbatch saint_squid.sh
Submitted batch job 153832
sbatch saint_cuttlefish.sh
Submitted batch job 153834

sbatch saint_pork.sh
Submitted batch job 153836
===========
1-5
sbatch tab_nemi.sh
Submitted batch job 153852
sbatch tab_cacao_butter.sh
Submitted batch job 153837
sbatch tab_coffee.sh
Submitted batch job 153839
sbatch tab_plant_oil.sh
Submitted batch job 153840
sbatch tab_chinese_wine.sh
Submitted batch job 153841
sbatch tab_cuttlefish.sh
Submitted batch job 153842
sbatch tab_octopus.sh
Submitted batch job 153844
sbatch tab_pork.sh 
Submitted batch job 153851

======
RS1
1-3 folds
sbatch tab_wine_spoilage.sh
Submitted batch job 153861
(base) [40774065@login1[enucc] src]$ sbatch tab_vinegar.sh
Submitted batch job 153862
(base) [40774065@login1[enucc] src]$ sbatch node_wine_spoilage.sh
Submitted batch job 153863
(base) [40774065@login1[enucc] src]$ sbatch node_vinegar.sh
Submitted batch job 153864
(base) [40774065@login1[enucc] src]$ sbatch saint_wine_spoilage.sh
Submitted batch job 153865
(base) [40774065@login1[enucc] src]$ sbatch saint_vinegar.sh
Submitted batch job 153866