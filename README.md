- To run imitation learning, use ``` nohup python main.py``` under working directory `imitaion_learning`. Results will be saved under fold `model_checkpoint` along with model checkpoints. The fold will be named after the current time-stamp. You can also tweak command arguments to customize training phase. For example,

  - ```nohup python main.py --load_prop 1.0 --epoch 30 --option 3 --map 12 --device 'cuda:0' > IL_OPT3_MAP12_TB.log 2>&1 &```

  For details of or to modify command arguments, please see `imitation_learning/main.py`

- Code structures, configuration, command arguments, etc remain similar for Deep Q-learning.

- To submit via kaggle API, you need to ensure your API token file exists in `~/.kaggle/kaggle.json`. Then run command:
  - ```kaggle competitions submit -c lux-ai-2021 -f FILE_NAME -m MESSAGE

- Experiments for Imitation Learning and DQN are conducted with Python 3.7, PyTorch 1.7, CUDA 11.0, kaggle-enviroments 1.8.12, gym 0.19.0, 