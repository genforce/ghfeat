# python 3.6
"""Extracts Generative Hierarchical Feature (GH-Feat) from give images."""

import os
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from dnnlib import tflib

from utils.logger import setup_logger
from utils.visualizer import adjust_pixel_range
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image, load_image, resize_image

from dnnlib import EasyDict


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('model_path', type=str,
                      help='Path to the pre-trained model.')
  parser.add_argument('image_list', type=str,
                      help='List of images to extract ghfeat.')
  parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`./results/ghfeat/${IMAGE_LIST}` '
                           'will be used by default.')
  parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size. (default: 4)')
  parser.add_argument('--viz_size', type=int, default=256,
                      help='Image size for visualization. (default: 256)')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  assert os.path.exists(args.image_list)
  image_list_name = os.path.splitext(os.path.basename(args.image_list))[0]
  output_dir = args.output_dir or f'results/ghfeat/{image_list_name}'
  logger = setup_logger(output_dir, 'extract_feature.log', 'inversion_logger')

  logger.info(f'Loading model.')
  tflib.init_tf({'rnd.np_random_seed': 1000})
  with open(args.model_path, 'rb') as f:
    E, _, _, Gs = pickle.load(f)

  # Get input size.
  image_size = E.input_shape[2]
  assert image_size == E.input_shape[3]

  G_args = EasyDict(func_name='training.networks_stylegan.G_synthesis')
  G_style_mod = tflib.Network('G_StyleMod', resolution=image_size, label_size=0, **G_args)
  Gs_vars_pairs = {name: tflib.run(val) for name, val in Gs.components.synthesis.vars.items()}
  for g_name, g_val in G_style_mod.vars.items():
    tflib.set_vars({g_val: Gs_vars_pairs[g_name]})

  # Build graph.
  logger.info(f'Building graph.')
  sess = tf.get_default_session()
  input_shape = E.input_shape
  input_shape[0] = args.batch_size
  x = tf.placeholder(tf.float32, shape=input_shape, name='real_image')
  ghfeat = E.get_output_for(x, is_training=False)
  x_rec = G_style_mod.get_output_for(ghfeat, randomize_noise=False)

  # Load image list.
  logger.info(f'Loading image list.')
  image_list = []
  with open(args.image_list, 'r') as f:
    for line in f:
      image_list.append(line.strip())

  # Extract GH-Feat from images.
  logger.info(f'Start feature extraction.')
  headers = ['Name', 'Original Image', 'Encoder Output']
  viz_size = None if args.viz_size == 0 else args.viz_size
  visualizer = HtmlPageVisualizer(
      num_rows=len(image_list), num_cols=len(headers), viz_size=viz_size)
  visualizer.set_headers(headers)

  images = np.zeros(input_shape, np.uint8)
  names = ['' for _ in range(args.batch_size)]
  features = []
  for img_idx in tqdm(range(0, len(image_list), args.batch_size), leave=False):
    # Load inputs.
    batch = image_list[img_idx:img_idx + args.batch_size]
    for i, image_path in enumerate(batch):
      image = resize_image(load_image(image_path), (image_size, image_size))
      images[i] = np.transpose(image, [2, 0, 1])
      names[i] = os.path.splitext(os.path.basename(image_path))[0]
    inputs = images.astype(np.float32) / 255 * 2.0 - 1.0
    # Run encoder.
    outputs = sess.run([ghfeat, x_rec], {x: inputs})
    features.append(outputs[0][0:len(batch)])
    outputs[1] = adjust_pixel_range(outputs[1])
    for i, _ in enumerate(batch):
      image = np.transpose(images[i], [1, 2, 0])
      save_image(f'{output_dir}/{names[i]}_ori.png', image)
      save_image(f'{output_dir}/{names[i]}_enc.png', outputs[1][i])
      visualizer.set_cell(i + img_idx, 0, text=names[i])
      visualizer.set_cell(i + img_idx, 1, image=image)
      visualizer.set_cell(i + img_idx, 2, image=outputs[1][i])

  # Save results.
  os.system(f'cp {args.image_list} {output_dir}/image_list.txt')
  np.save(f'{output_dir}/ghfeat.npy', np.concatenate(features, axis=0))
  visualizer.save(f'{output_dir}/reconstruction.html')


if __name__ == '__main__':
  main()
