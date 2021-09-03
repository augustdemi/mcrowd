gpu=3
run_id=305
desc='local_map(not_circle)_train_with_or_wo_TF,no_kl_clamp'
dataset_name='A2E'
dataset_dir='/dresden/users/ml1323/crowd/baseline/HTP-benchmark/Splits'
kl_weight=100
z_S=32
dropout_mlp=0.1
dropout_rnn=0.25
pred_len=12
batch_size=64
ckpt_load_iter=700
max_iter=700
ckpt_save_iter=50
encoder_h_dim=64
map_size=32
map_feat_dim=16
decoder_h_dim=64
mlp_dim=32
map_mlp_dim=64
vis_ll=50
lr_VAE=1e-3
radius_deno=8
ll_prior_w=1

CUDA_VISIBLE_DEVICES=${gpu} \
	nohup python -u main.py \
	--run_id ${run_id} --desc ${desc} --device cuda --pool_every_timestep 0 --kl_weight ${kl_weight} \
	--zS_dim ${z_S} --map_size ${map_size} \
	--ckpt_save_iter ${ckpt_save_iter} --ckpt_load_iter ${ckpt_load_iter} --max_iter ${max_iter} \
	--print_iter 10 --map_feat_dim ${map_feat_dim} \
	--batch_size ${batch_size}  --num_layers 1 \
	--dataset_dir ${dataset_dir} --dataset_name ${dataset_name} --delim , --loader_num_workers 0 \
	--obs_len 8 --pred_len ${pred_len} \
	--radius_deno ${radius_deno} --ll_prior_w ${ll_prior_w} \
	--encoder_h_dim ${encoder_h_dim} --decoder_h_dim ${decoder_h_dim} \
	--mlp_dim ${mlp_dim} --map_mlp_dim ${map_mlp_dim} \
	--dropout_mlp ${dropout_mlp} --dropout_rnn ${dropout_rnn} \
	--lr_VAE ${lr_VAE} --beta1_VAE 0.9 --beta2_VAE 0.999 \
	--viz_on True --viz_ll_iter ${vis_ll} --viz_la_iter ${vis_ll} --viz_port 8002 \
	> ./log/${dataset_name}.${run_id}.pred_len${pred_len}.z_S${z_S}.drop_mlp${dropout_mlp}.drop_rnn${dropout_rnn}.encoder_h_dim${encoder_h_dim}.dec_h_dim${decoder_h_dim}.mlp_dim${mlp_dim}.lr_VAE${lr_VAE}.bs${batch_size}.kl_w${kl_weight}.map_size${map_size}.map_feat_dim${map_feat_dim}.map_mlp${map_mlp_dim}.r_deno${radius_deno}.ll_prior_w${ll_prior_w}.eval.txt &
