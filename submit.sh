cp imitation_learning/model_checkpoints/IL_OPT3_24/best_acc.pth imitation_learning/submission/24.pth
cp imitation_learning/model_checkpoints/IL_OPT3_12/best_acc.pth imitation_learning/submission/12.pth
cp imitation_learning/model_checkpoints/IL_OPT3_16/best_acc.pth imitation_learning/submission/16.pth
cp imitation_learning/model_checkpoints/IL_OPT8_32/best_acc.pth imitation_learning/submission/32.pth
cd imitation_learning/submission
tar -czvf IL1103_2.tar.gz  12.pth 16.pth 24.pth 32.pth main.py lux/.
kaggle competitions submit -c lux-ai-2021 -f IL1103_2.tar.gz -m "Opt 3(12,16,24,TB,RL), Opt 8(32, TB), best acc, another submission"
