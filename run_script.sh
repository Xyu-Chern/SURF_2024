# python main/script.py --path "PointMaze_UMazeDense-v3_PPO&RND_v2"  --k 16 &
# python main/script.py --path "PointMaze_UMazeDense-v3_PPO_v2" --k 16 &
# python main/script.py --path "PointMaze_UMazeDense-v3_RND_v2" --k 16 &
# python main/script.py --path "PointMaze_UMazeDense-v3_random_v2" --k 16 &

python main/script.py --name "PointMaze_UMazeDense-v3_random_v2_gamma_0p9995" --gamma 0.9995 &
python main/script.py --name "PointMaze_UMazeDense-v3_random_v2_gamma_0p9" --gamma 0.9 &
python main/script.py --name "PointMaze_UMazeDense-v3_random_v2_lr_0p01" --v_lr 0.01 --a_lr 0.01 --q_lr 0.01 &
python main/script.py --name "PointMaze_UMazeDense-v3_random_v2_lr_0p0001"  --v_lr 0.0001 --a_lr 0.0001 --q_lr 0.0001 &
python main/script.py --name "PointMaze_UMazeDense-v3_random_v2_tau_0p9"  --tau 0.9 &
python main/script.py --name "PointMaze_UMazeDense-v3_random_v2_tau_0p1"  --tau 0.1 &
python main/script.py --name "PointMaze_UMazeDense-v3_random_v2_normal"  &