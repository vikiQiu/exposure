import sys
import os
from util import load_config, GPU

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpu = GPU().choose_gpu()
print('Using the automatically choosed GPU %s' % gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

from net import GAN


def main():
  config_name = sys.argv[1]
  cfg = load_config(config_name)
  cfg.name = sys.argv[1] + '/' + sys.argv[2] # example/test
  net = GAN(cfg, restore=False)
  net.train()


if __name__ == '__main__':
  main()
