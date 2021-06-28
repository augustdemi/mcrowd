gpu=4
run_id=72
desc='normal,withTF,min_ped=1,temp=0.66,resized64,w22_loaded,fc_fix'
dataset_name='zara1'
kl_weight=100
z_S=64
dropout_mlp=0.1
dropout_rnn=0.25
pred_len=12
batch_size=64
ckpt_load_iter=0
max_iter=20000
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
	nohup python -u main.py \
	--run_id ${run_id} --desc ${desc} --device cuda --pool_every_timestep 0 --kl_weight ${kl_weight} \
	--zS_dim ${z_S} --attention ${attention} --map_size ${map_size} \
	--ckpt_save_iter ${ckpt_save_iter} --ckpt_load_iter ${ckpt_load_iter} --max_iter ${max_iter} \
	--print_iter 10 \
	--batch_size ${batch_size}  --num_layers 1 \
	--dataset_dir ../datasets --dataset_name ${dataset_name} --delim tab --loader_num_workers 0 \
	--obs_len 8 --pred_len ${pred_len} \
	--encoder_h_dim ${encoder_h_dim} --decoder_h_dim ${decoder_h_dim} \
	--mlp_dim ${mlp_dim} \
	--dropout_mlp ${dropout_mlp} --dropout_rnn ${dropout_rnn} \
	--lr_VAE ${lr_VAE} --beta1_VAE 0.9 --beta2_VAE 0.999 \
	--viz_on True --viz_ll_iter 100 --viz_la_iter 100 --viz_port 8002 \
	> ./log/${dataset_name}.${run_id}.pred_len${pred_len}.z_S${z_S}.drop_mlp${dropout_mlp}.drop_rnn${dropout_rnn}.encoder_h_dim${encoder_h_dim}.dec_h_dim${decoder_h_dim}.mlp_dim${mlp_dim}.lr_VAE${lr_VAE}.bs${batch_size}.kl_w${kl_weight}.map_size${map_size}.txt &
