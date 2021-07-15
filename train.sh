gpu=3
run_id=18
desc='runid1,with_map,no_relation_attn'
dataset_name='univ'
kl_weight=100
latent_dim=64
emb_size=512
d_ff=2048
layers=6
heads=8
dropout=0.1
pred_len=12
batch_size=32
ckpt_load_iter=0
max_iter=40000
ckpt_save_iter=200
map_size=180
lr_VAE=1e-3

CUDA_VISIBLE_DEVICES=${gpu} \
	nohup python -u main.py \
	--run_id ${run_id} --desc ${desc} --device cuda --kl_weight ${kl_weight} \
	--latent_dim ${latent_dim} --emb_size ${emb_size} --d_ff ${d_ff} \
	--layers ${layers} --heads ${heads} --dropout ${dropout} \
	--map_size ${map_size} \
	--ckpt_save_iter ${ckpt_save_iter} --ckpt_load_iter ${ckpt_load_iter} --max_iter ${max_iter} \
	--print_iter 10 \
	--batch_size ${batch_size} \
	--dataset_dir ../datasets --dataset_name ${dataset_name} --delim tab --loader_num_workers 0 \
	--obs_len 8 --pred_len ${pred_len} \
	--lr_VAE ${lr_VAE} --beta1_VAE 0.9 --beta2_VAE 0.999 \
	--viz_on True --viz_ll_iter 400 --viz_la_iter 400 --viz_port 8002 \
	> ./log_transformer/${dataset_name}.${run_id}.pred_len${pred_len}.latent_dim${latent_dim}.emb_size${emb_size}.d_ff${d_ff}.layers${layers}.heads${heads}.dropout${dropout}.lr_VAE${lr_VAE}.bs${batch_size}.kl_w${kl_weight}.map_size${map_size}.txt &
