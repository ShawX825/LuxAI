#nohup python embedding.py --map 12 --device 'cuda:0' > 12.log 2>&1 &
#nohup python embedding.py --map 16 --device 'cuda:1' > 16.log 2>&1 &
#nohup python embedding.py --map 24 --device 'cuda:2' > 24.log 2>&1 &
#nohup python embedding.py --map 32 --device 'cuda:3' > 32.log 2>&1 &
#nohup python main.py --device "cuda:0" --map 12 --epoch 10 --train_steps 15 --lr 0.001 --transfer 2 > DAGGER_MAP12_5.log 2>&1 &
#nohup python main.py --device "cuda:1" --map 16 --epoch 10 --train_steps 15 --lr 0.001 --transfer 2 > DAGGER_MAP16_7.log 2>&1 &
#nohup python main.py --device "cuda:2" --map 24 --epoch 10 --train_steps 15 --lr 0.001 --transfer 2 > DAGGER_MAP24_7.log 2>&1 &
#nohup python main.py --device "cuda:3" --map 32 --epoch 10 --train_steps 10 > DAGGER_MAP32.log 2>&1 &
nohup python main.py --device "cuda:0" --map 12 --epoch 500 --train_steps 10 > DAGGER_MAP12_6.log 2>&1 &
nohup python main.py --device "cuda:1" --map 16 --epoch 500 --train_steps 10 > DAGGER_MAP16_8.log 2>&1 &
nohup python main.py --device "cuda:2" --map 24 --epoch 500 --train_steps 10 > DAGGER_MAP24_8.log 2>&1 &
nohup python main.py --device "cuda:3" --map 32 --epoch 200 --train_steps 10 > DAGGER_MAP32_1.log 2>&1 &
