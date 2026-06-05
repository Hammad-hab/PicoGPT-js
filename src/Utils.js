import fs from "node:fs"
import path from "path"
import { Readable } from 'stream'

const model_sizes = ["124M", "355M", "774M", "1558M"]
const files =  [
    "checkpoint",
    "encoder.json",
    "hparams.json",
    "model.ckpt.data-00000-of-00001",
    "model.ckpt.index",
    "model.ckpt.meta",
    "vocab.bpe",
]
const endpoint = "https://openaipublic.blob.core.windows.net/gpt-2/models"
const download_gpt2_files = async (model_size, model_dir) => {
    if (!model_sizes.includes(model_size)) {
        throw `Error: Provided model size does not exist`
    }
    for await (const file of files) {
        const filepath = path.join(file)
        const stream = new fs.createWriteStream(filepath)
        const req = await fetch(`${endpoint}/${model_size}/${filename}`)
        const body = req.body
        Readable.fromWeb(body).pipe(stream);
    }
}

export default download_gpt2_files;