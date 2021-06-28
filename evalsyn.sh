gpu=2
run_id=22
desc='normal,withTF,min_ped=1,temp=1.99'
dataset_name='all'
kl_weight=100
z_S=64
dropout_mlp=0.1
dropout_rnn=0.25
pred_len=12
batch_size=32
ckpt_load_iter=10000
max_iter=10000
ckpt_save_iter=100
encoder_h_dim=32
attention=0
decoder_h_dim=128
mlp_dim=32
lr_VAE=1e-3
map_trainable=1

CUDA_VISIBLE_DEVICES=${gpu} \
	nohup python -u main.py \
	--run_id ${run_id} --desc ${desc} --device cuda --pool_every_timestep 0 --kl_weight ${kl_weight} \
	--zS_dim ${z_S} --attention ${attention} \
	--ckpt_save_iter ${ckpt_save_iter} --ckpt_load_iter ${ckpt_load_iter} --max_iter ${max_iter} \
	--print_iter 10 --map_trainable ${map_trainable} \
	--batch_size ${batch_size}  --num_layers 1 \
	--dataset_dir ../datasets/syn_x_cropped --dataset_name ${dataset_name} --delim tab --loader_num_workers 0 \
	--obs_len 8 --pred_len ${pred_len} \
	--encoder_h_dim ${encoder_h_dim} --decoder_h_dim ${decoder_h_dim} \
	--mlp_dim ${mlp_dim} \
	--dropout_mlp ${dropout_mlp} --dropout_rnn ${dropout_rnn} \
	--lr_VAE ${lr_VAE} --beta1_VAE 0.9 --beta2_VAE 0.999 \
	--viz_on True --viz_ll_iter 500 --viz_la_iter 500 --viz_port 8002 \
	> ./eslog/${dataset_name}.${run_id}.pred_len${pred_len}.z_S${z_S}.drop_mlp${dropout_mlp}.drop_rnn${dropout_rnn}.encoder_h_dim${encoder_h_dim}.dec_h_dim${decoder_h_dim}.mlp_dim${mlp_dim}.lr_VAE${lr_VAE}.bs${batch_size}.kl_w${kl_weight}.attn${attention}_${ckpt_load_iter}.map_trainable${map_trainable}_coll3.txt &
