gpu=3
run_id=24
desc='no_velocity,no_ped_obst,center1_before_transform'
dataset_name='nmap'
kl_weight=100
z_S=64
dropout_map=0
gamma=0
alpha=0.5
pred_len=12
batch_size=64
ckpt_load_iter=0
max_iter=10000
ckpt_save_iter=100
encoder_h_dim=32
attention=0
map_size=180
#pool_dim=$encoder_h_dim
#pool_dim=0
decoder_h_dim=128
mlp_dim=32
#pooling_type='attn'
lr_VAE=1e-3

CUDA_VISIBLE_DEVICES=${gpu} \
	nohup python -u main_map_ae.py \
	--run_id ${run_id} --desc ${desc} --device cuda \
	--batch_size ${batch_size} --map_size ${map_size} \
	--ckpt_save_iter ${ckpt_save_iter} --ckpt_load_iter ${ckpt_load_iter} --max_iter ${max_iter} \
	--print_iter 10 --gamma ${gamma} --alpha ${alpha} \
	--batch_size ${batch_size}  --num_layers 1 \
	--dataset_dir ../datasets --dataset_name ${dataset_name} --delim tab --loader_num_workers 0 \
	--obs_len 8 --pred_len ${pred_len} \
	--encoder_h_dim ${encoder_h_dim} --decoder_h_dim ${decoder_h_dim} \
	--mlp_dim ${mlp_dim} \
	--dropout_map ${dropout_map}  \
	--lr_VAE ${lr_VAE} --beta1_VAE 0.9 --beta2_VAE 0.999 \
	--viz_on True --viz_ll_iter 50 --viz_la_iter 50 --viz_port 8002 \
	> ./log_ae/${dataset_name}.${run_id}bs${batch_size}.map_size${map_size}.drop_out${dropout_map}.gamma${gamma}.alpha${alpha}.txt &
