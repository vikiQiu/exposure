import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from util import load_config
from util import load_config, GPU

# gpu = GPU().choose_gpu()
# print('Using the automatically choosed GPU %s' % gpu)
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu

from net import GAN


def evaluate():
    if len(sys.argv) < 4:
        print(
            "Usage: python3 evaluate.py [config suffix] [model name] [image files name1] [image files name2] ..."
        )
        exit(-1)
    if len(sys.argv) == 4:
        print(
            " Note: Process a single image at a time may be inefficient - try multiple inputs)"
        )
    print("(TODO: batch pro cessing when images have the same resolution)")
    print()
    print("Initializing...")
    config_name = sys.argv[1]
    import shutil
    shutil.copy('models/%s/%s/scripts/config_%s.py' %
                (config_name, sys.argv[2], config_name), 'config_tmp.py')
    cfg = load_config('tmp')
    cfg.name = sys.argv[1] + '/' + sys.argv[2]
    net = GAN(cfg, restore=True)
    net.restore(20000)  # restore the model
    spec_files = sys.argv[3:]
    file_dir = sys.argv[3]
    print('processing files {}', spec_files)
    net.eval_on_mit5k_small(file_dir=file_dir, step_by_step=False)


if __name__ == '__main__':
    evaluate()
