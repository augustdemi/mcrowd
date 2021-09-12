gpu=0
run_id=400
desc='new_pathfinding,u-net'
dataset_name='path'
dataset_dir='/dresden/users/ml1323/crowd/datasets/Trajectories'
kl_weight=100
lg_kl_weight=50
z_S=10
w_dim=20
dropout_mlp=0.3
dropout_rnn=0.25
pred_len=12
batch_size=64
ckpt_load_iter=0
max_iter=20000
ckpt_save_iter=500
map_mlp_dim=256
map_feat_dim=32
encoder_h_dim=64
decoder_h_dim=256
mlp_dim=256
vis_ll=100
lr_VAE=1e-4
ll_prior_w=1

CUDA_VISIBLE_DEVICES=${gpu} \
	nohup python -u main.py \
	--run_id ${run_id} --desc ${desc} --device cuda --pool_every_timestep 0 --kl_weight ${kl_weight} \
	--zS_dim ${z_S} --w_dim ${w_dim} --lg_kl_weight ${lg_kl_weight} \
	--ckpt_save_iter ${ckpt_save_iter} --ckpt_load_iter ${ckpt_load_iter} --max_iter ${max_iter} \
	--print_iter 10 --map_feat_dim ${map_feat_dim} \
	--batch_size ${batch_size}  --num_layers 1 \
	--dataset_dir ${dataset_dir} --dataset_name ${dataset_name} --delim , --loader_num_workers 0 \
	--obs_len 8 --pred_len ${pred_len} \
	--ll_prior_w ${ll_prior_w} \
	--encoder_h_dim ${encoder_h_dim} --decoder_h_dim ${decoder_h_dim} \
	--mlp_dim ${mlp_dim} --map_mlp_dim ${map_mlp_dim} \
	--dropout_mlp ${dropout_mlp} --dropout_rnn ${dropout_rnn} \
	--lr_VAE ${lr_VAE} --beta1_VAE 0.9 --beta2_VAE 0.999 \
	--viz_on True --viz_ll_iter ${vis_ll} --viz_la_iter ${vis_ll} --viz_port 8002 \
	> ./log/${dataset_name}.${run_id}.pred_len${pred_len}.z_S${z_S}.drop_mlp${dropout_mlp}.drop_rnn${dropout_rnn}.encoder_h_dim${encoder_h_dim}.dec_h_dim${decoder_h_dim}.mlp_dim${mlp_dim}.lr_VAE${lr_VAE}.bs${batch_size}.kl_w${kl_weight}.lg_kl_weight${lg_kl_weight}.map_feat_dim${map_feat_dim}.map_mlp${map_mlp_dim}.w_dim${w_dim}.ll_prior_w${ll_prior_w}.txt &
