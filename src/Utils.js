import fs from "node:fs";
import path from "path";
import { Readable } from "stream";
import * as tf from "@tensorflow/tfjs";
import { execSync } from "node:child_process";
import { getEncoder } from "./Encoder";
import * as npy from "npyjs"

const model_sizes = ["124M", "355M", "774M", "1558M"];
const files = [
  "checkpoint",
  "encoder.json",
  "hparams.json",
  "model.ckpt.data-00000-of-00001",
  "model.ckpt.index",
  "model.ckpt.meta",
  "vocab.bpe",
];
const endpoint = "https://openaipublic.blob.core.windows.net/gpt-2/models";
const download_gpt2_files = async (model_size, model_dir) => {
  if (!model_sizes.includes(model_size)) {
    throw `Error: Provided model size does not exist`;
  }
  for await (const file of files) {
    const filepath = path.join(file);
    const stream = new fs.createWriteStream(filepath);
    const req = await fetch(`${endpoint}/${model_size}/${filename}`);
    const body = req.body;
    Readable.fromWeb(body).pipe(stream);
  }
};

const set_in_nested_dict = (d, keys, val) => {
  if (!keys) return val;
  if (!Object.keys(d).includes(keys[0])) {
    d[keys[0]] = {};
  }
  d[keys[0]] = set_in_nested_dict(d[keys[0]], keys.slice(1), val);
  return d;
};


const _tf_conv_weights = (path) => {
/**
 * There were probably better ways to implement this
 * however this seemed like the easiest fix for the
 * inavailability of .ckpt
 */
console.log('Converting weights, this may take a minute...');
execSync(`python3 -c "
import tensorflow as tf
import numpy as np

def save_weights_to_npz(model_dir, output_path):
    ckpt_path = tf.train.latest_checkpoint(model_dir)
    weights = {}
    for name, _ in tf.train.list_variables(ckpt_path):
        array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        weights[name.replace('/', '_')] = array  # npz keys can't have /
    
    np.savez(output_path, **weights)
    print(f"Saved to {output_path}")

save_weights_to_npz('${path}', '${path}/weights.npz')   
"`)
console.log('Done!');
}

const tf_train_list_variables = (tf_ckpt_path) => {

const output = execSync(`python3 -c "
import tensorflow as tf
import json

ckpt_path = '${tf_ckpt_path}'
vars = [[name, shape] for name, shape in tf.train.list_variables(ckpt_path)]
print(json.dumps(vars))
"`).toString();
return JSON.parse(output)
}

let weight_file = null

const tf_train_load_variable = async (tf_ckpt_path, name) => {
    if (!fs.existsSync(`${tf_ckpt_path}/weights.npz`)) {
      _tf_conv_weights(tf_ckpt_path);
    }
    
    if (!weight_file) {
      weight_file = await npy.load(`${tf_ckpt_path}/weights.npz`);
    }
  
    const key = name.replace(/\//g, '_');
    return weight_file[key];
}

const load_gpt2_params_from_tf_ckpt = (tf_ckpt_path, hparams) => {


  const n_layers = hparams["n_layer"];
  const params = { blocks: Array.from({ length: n_layers }, () => ({})) };
  for (const [lbl ] of tf_train_list_variables(tf_ckpt_path)) {
      arr = tf_train_load_variable(tf_ckpt_path, lbl)
      const name = lbl.slice("model/".length)
      if (name.startsWith('h')) {
            // ?
            const m = name.match(/^h([0-9]+)\/(.*)/);
            const n = int(m[1])
            const subName = m[2]
            set_in_nested_dict(params['blocks'][n], subName.split('/'), arr)
      } else {
        set_in_nested_dict(params, name.split('/'), arr)
      }
  }
};

function latestCheckpoint(modelDir) {
    const content = fs.readFileSync(path.join(modelDir, 'checkpoint'), 'utf8');
    const match = content.match(/model_checkpoint_path:\s*"([^"]+)"/);
    if (!match) return null;
    const ckptPath = match[1];
    return path.isAbsolute(ckptPath) ? ckptPath : path.join(modelDir, ckptPath);
  }

const load_encoder_hparams_and_params = async (model_size, models_dir) => {
  if (!model_sizes.includes(model_size)) {
    throw `Error: Provided model size does not exist`;
  }
  const model_dir = path.join(models_dir, model_size);
  const tf_ckpt_path = latestCheckpoint(model_dir);

  if (!tf_ckpt_path) {
    if (!fs.existsSync(model_dir)) fs.mkdirSync(model_dir);
    await download_gpt2_files(model_size, model_dir);
    tf_ckpt_path = latestCheckpoint(model_dir);
  }

  const encoder = getEncoder(model_size, models_dir);
  const target_pth = path.join(model_dir, "hparams.json");
  const hparams = JSON.parse(fs.readFileSync(target_pth, "utf-8"));
  const params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, hparams);

  return { encoder, hparams, params };
};

export { download_gpt2_files, load_encoder_hparams_and_params };
