gpu=2
run_id=5
desc='TF,bceloss,cropped'
dataset_name='all'
kl_weight=1
z_S=64
dropout_mlp=0.1
dropout_rnn=0.25
dropout_map=0.5
emb_dim=16
pred_len=12
batch_size=64
ckpt_load_iter=0
max_iter=10000
ckpt_save_iter=50
encoder_h_dim=32
decoder_h_dim=128
mlp_dim=32
lr_VAE=1e-3

CUDA_VISIBLE_DEVICES=${gpu} \
	nohup python -u main_map_vae.py \
	--run_id ${run_id} --desc ${desc} --device cuda --kl_weight ${kl_weight} \
	--zS_dim ${z_S} --output_save_iter 50 \
	--ckpt_save_iter ${ckpt_save_iter} --ckpt_load_iter ${ckpt_load_iter} --max_iter ${max_iter} \
	--print_iter 10 \
	--batch_size ${batch_size}  --num_layers 1 \
	--dataset_dir ../datasets/map --dataset_name ${dataset_name} --delim tab --loader_num_workers 0 \
	--obs_len 8 --pred_len ${pred_len} \
	--encoder_h_dim ${encoder_h_dim} --decoder_h_dim ${decoder_h_dim} --emb_dim ${emb_dim} \
	--mlp_dim ${mlp_dim} \
	--dropout_mlp ${dropout_mlp} --dropout_rnn ${dropout_rnn} --dropout_map ${dropout_map} \
	--lr_VAE ${lr_VAE} --beta1_VAE 0.9 --beta2_VAE 0.999 \
	--viz_on True --viz_ll_iter 10 --viz_la_iter 40 --viz_port 8002 \
	> ./mlog/${dataset_name}.${run_id}.pred_len${pred_len}.z_S${z_S}.drop_mlp${dropout_mlp}.drop_rnn${dropout_rnn}.drop_map${dropout_map}.enc_h_dim${encoder_h_dim}.dec_h_dim${decoder_h_dim}.mlp_dim${mlp_dim}.emb_dim${emb_dim}.lr_VAE${lr_VAE}.bs${batch_size}.kl_w${kl_weight}.txt &
