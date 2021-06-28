gpu=6
run_id=42
desc='normal,tf,ped=1,map-pretrained'
dataset_name='zara2'
kl_weight=100
z_S=64
dropout_mlp=0.1
dropout_rnn=0.25
pred_len=12
batch_size=64
ckpt_load_iter=4200
max_iter=4200
ckpt_save_iter=100
encoder_h_dim=32
#pool_dim=$encoder_h_dim
attention=0
decoder_h_dim=128
mlp_dim=32
lr_VAE=1e-3

CUDA_VISIBLE_DEVICES=${gpu} \
	nohup python -u main.py \
	--run_id ${run_id} --desc ${desc} --device cuda --pool_every_timestep 0 --kl_weight ${kl_weight} \
	--zS_dim ${z_S} \
	--ckpt_save_iter ${ckpt_save_iter} --ckpt_load_iter ${ckpt_load_iter} --max_iter ${max_iter} \
	--print_iter 10 \
	--batch_size ${batch_size}  --num_layers 1 \
	--dataset_dir ../datasets --dataset_name ${dataset_name} --delim tab --loader_num_workers 0 \
	--obs_len 8 --pred_len ${pred_len} \
	--encoder_h_dim ${encoder_h_dim} --decoder_h_dim ${decoder_h_dim} \
	--mlp_dim ${mlp_dim} --attention ${attention} \
	--dropout_mlp ${dropout_mlp} --dropout_rnn ${dropout_rnn} \
	--lr_VAE ${lr_VAE} --beta1_VAE 0.9 --beta2_VAE 0.999 \
	--viz_on True --viz_ll_iter 10 --viz_la_iter 40 --viz_port 8002 \
	> ./elog/${dataset_name}.${run_id}.pred_len${pred_len}.z_S${z_S}.drop_mlp${dropout_mlp}.drop_rnn${dropout_rnn}.encoder_h_dim${encoder_h_dim}.dec_h_dim${decoder_h_dim}.mlp_dim${mlp_dim}.lr_VAE${lr_VAE}.bs${batch_size}.kl_w${kl_weight}.txt &
